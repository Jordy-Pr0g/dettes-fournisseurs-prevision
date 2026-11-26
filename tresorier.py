"""
Vue du trésorier : ce qu'il doit savoir un lundi matin.
- la prévision des sorties d'argent pour les 8 prochaines semaines ;
- les alertes : factures à regarder en priorité, classées par urgence et par montant.

La vue ne regarde que ce qui est connu à la date choisie : une facture payée après cette date est encore ouverte,
et une validation postérieure n'est pas encore faite. Aucun paiement n'est déclenché : le trésorier décide.

Entrée : transformed_dataset.csv, ou resultats_pipeline.xlsx (onglet transformed_dataset)
Sorties : resultats_tresorier.xlsx, graphique_tresorier.png, alertes_tresorier.csv, prevision_8_semaines.csv

Usage :
    results = run_tresorier("transformed_dataset.csv", as_of="2026-06-29")
"""
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from modele import add_features, build_model, load_data, to_matrix, week_start
from prevision import credit_note_distribution

DEMO_DATE = "2026-06-29"          # le lundi qui suit les dernières factures du fichier
HORIZON_WEEKS = 8
BACKTEST_MONDAYS = ("2026-02-02", "2026-06-22")
BACKTEST_HORIZONS = (0, 1, 3)     # 0 = la semaine qui commence, 1 = la suivante, 3 = dans trois semaines

# [SUGGESTION] Seuils proposés, absents du Data Contract : à valider avec un trésorier
RISK_THRESHOLD = 0.5              # probabilité d'au moins une semaine de retard
DUE_SOON_DAYS = 7
RISK_WINDOW_DAYS = 14
BLOCKING_ANOMALIES = ["ecart_prix", "ecart_quantite", "bc_manquant"]

ALERTS = {1: "Échéance dépassée", 2: "Échéance proche, pas validée", 3: "Anomalie à résoudre", 4: "Risque de retard élevé"}
EXCEL_COLOR, MODEL_COLOR, INK, MUTED, GRID = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def prepare(path):
    """Prépare les factures saisies en 2026 et leurs probabilités, avec le modèle entraîné sur les saisies d'avant 2026."""
    data = add_features(load_data(path))
    data["approval_date"] = pd.to_datetime(data.approval_date, errors="coerce")   # jamais utilisée par le modèle
    usable = data[data.usable_for_model]
    train = usable[usable.split == "train"]
    model = build_model().fit(to_matrix(train), train.week_offset)
    offsets, credit_proba = credit_note_distribution(data[data.split == "train"])

    # Un doublon repéré au contrôle (anomaly_type) n'entre jamais dans la prévision : ce fait est connu à la saisie
    scope = data[(data.split == "test") & (data.anomaly_type != "doublon")].reset_index(drop=True)
    classes = np.arange(offsets.min(), offsets.max() + 1)
    proba = np.zeros((len(scope), len(classes)))
    is_invoice = (scope.doc_type == "facture").to_numpy()
    predicted = model.predict_proba(to_matrix(scope[is_invoice]))
    positions = np.searchsorted(classes, model.classes_)
    proba[np.ix_(np.flatnonzero(is_invoice), positions)] = predicted
    proba[~is_invoice] = credit_proba
    return scope, proba, classes


def open_mask(scope, as_of):
    """Saisie à cette date et pas encore payée à cette date."""
    return ((scope.posting_date <= as_of) & (scope.payment_date.isna() | (scope.payment_date > as_of))).to_numpy()


def weekly_forecast(rows, proba, classes, as_of, horizon):
    """Sorties prévues par semaine à partir de la date. Comme la facture est encore ouverte,
    les semaines déjà passées sont impossibles : on répartit le montant sur les semaines restantes."""
    current = week_start(pd.Series([as_of])).iloc[0]
    amounts = rows.amount_ttc.to_numpy()
    due_gap = ((week_start(rows.due_date) - current).dt.days // 7).to_numpy()

    relative = due_gap[:, None] + classes[None, :]
    possible = proba * (relative >= 0)
    total = possible.sum(axis=1)
    model = np.zeros(horizon)
    for i in range(len(rows)):
        if total[i] == 0:                 # plus aucune semaine prévue : le paiement est attendu tout de suite
            model[0] += amounts[i]
            continue
        for j in np.flatnonzero(possible[i]):
            if relative[i, j] < horizon:
                model[relative[i, j]] += amounts[i] * possible[i, j] / total[i]

    def single_date(gaps):            # Excel et la règle : toute la facture sur une semaine, jamais dans le passé
        out = np.zeros(horizon)
        np.add.at(out, np.clip(gaps, 0, None)[gaps < horizon], amounts[gaps < horizon])
        return out

    excel = single_date(due_gap)
    rule_dates = rows.due_date + pd.to_timedelta(rows.supplier_hist_median_delay.round(), unit="D")
    rule = single_date(((week_start(rule_dates) - current).dt.days // 7).to_numpy())

    paid_gap = ((week_start(rows.payment_date) - current).dt.days // 7)
    real = np.array([amounts[(paid_gap == h).to_numpy()].sum() for h in range(horizon)])
    weeks = [current + pd.Timedelta(days=7 * h) for h in range(horizon)]
    return pd.DataFrame({"semaine_du": weeks, "prevision_modele": model.round(0), "prevision_excel": excel.round(0),
                         "prevision_regle_simple": rule.round(0), "paye_en_realite": real.round(0)})


def build_alerts(scope, proba, classes, as_of):
    """Une ligne par facture à regarder, avec son alerte la plus urgente."""
    is_open = open_mask(scope, as_of)
    rows = scope[is_open & (scope.doc_type == "facture").to_numpy()].copy()
    rows["proba_retard"] = proba[is_open & (scope.doc_type == "facture").to_numpy()][:, classes >= 1].sum(axis=1).round(3)
    not_approved = rows.approval_date > as_of
    days_left = (rows.due_date - as_of).dt.days

    rows["priorite"] = np.select(
        [days_left < 0,
         not_approved & (days_left <= DUE_SOON_DAYS),
         not_approved & rows.anomaly_type.isin(BLOCKING_ANOMALIES),
         (rows.proba_retard >= RISK_THRESHOLD) & (days_left <= RISK_WINDOW_DAYS)],
        [1, 2, 3, 4], default=0)
    alerts = rows[rows.priorite > 0].copy()
    alerts["alerte"] = alerts.priorite.map(ALERTS)
    alerts["jours_avant_echeance"] = days_left[alerts.index]
    alerts["validee"] = (~not_approved[alerts.index]).map({True: "Oui", False: "Non"})
    alerts = alerts.sort_values(["priorite", "amount_ttc"], ascending=[True, False])
    return alerts[["alerte", "invoice_id", "supplier_name", "supplier_category", "amount_ttc", "due_date", "jours_avant_echeance",
                   "validee", "anomaly_type", "proba_retard", "approver_id", "payment_date"]], len(rows)


def backtest(scope, proba, classes):
    """Rejoue chaque lundi de 2026 : quelle erreur si le trésorier avait utilisé la vue ce jour-là ?"""
    rows = []
    for monday in pd.date_range(*BACKTEST_MONDAYS, freq="7D"):
        is_open = open_mask(scope, monday)
        table = weekly_forecast(scope[is_open], proba[is_open], classes, monday, max(BACKTEST_HORIZONS) + 1)
        for h in BACKTEST_HORIZONS:
            row = table.iloc[h]
            rows.append(dict(lundi=monday, horizon=h, reel=row.paye_en_realite, Modèle=abs(row.prevision_modele - row.paye_en_realite),
                             Excel=abs(row.prevision_excel - row.paye_en_realite), Règle=abs(row.prevision_regle_simple - row.paye_en_realite)))
    detail = pd.DataFrame(rows)
    summary = detail.groupby("horizon").agg(reel_moyen=("reel", "mean"), erreur_modele=("Modèle", "mean"), erreur_excel=("Excel", "mean"),
                                            erreur_regle_simple=("Règle", "mean"), lundis=("lundi", "count"),
                                            modele_bat_excel=("Modèle", lambda s: int((s < detail.loc[s.index, "Excel"]).sum())),
                                            modele_bat_regle=("Modèle", lambda s: int((s < detail.loc[s.index, "Règle"]).sum()))).round(0).astype(int).reset_index()
    summary["horizon"] = summary.horizon.map({0: "Semaine qui commence", 1: "Semaine suivante", 3: "Dans 3 semaines"})
    return summary


def chart_forecast(forecast, as_of, path):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    x = np.arange(len(forecast))
    ax.bar(x - 0.19, forecast.prevision_excel / 1000, width=0.34, color=EXCEL_COLOR, label="Excel (paiement à l'échéance)")
    ax.bar(x + 0.19, forecast.prevision_modele / 1000, width=0.34, color=MODEL_COLOR, label="Modèle")
    ax.set_xticks(x, [d.strftime("%d/%m") for d in forecast.semaine_du])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Milliers d'euros", color=MUTED, fontsize=10)
    ax.set_xlabel("Semaine du", color=MUTED, fontsize=10)
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="upper right")
    ax.set_title(f"Sorties d'argent prévues pour les factures ouvertes au {as_of:%d/%m/%Y}", loc="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return fig


def export_excel(path, tabs, image_path):
    with pd.ExcelWriter(path) as writer:
        for name, frame in tabs.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                longest = frame[col].astype(str).str.len().max() if len(frame) else 0
                sheet.column_dimensions[get_column_letter(i)].width = min(max(len(col), int(longest)) + 2, 60)
                is_date = pd.api.types.is_datetime64_any_dtype(frame[col])
                for row in range(2, len(frame) + 2):
                    if is_date:
                        sheet.cell(row, i).number_format = "DD/MM/YYYY"
                    elif pd.api.types.is_numeric_dtype(frame[col]) and frame[col].abs().max() > 1000:
                        sheet.cell(row, i).number_format = "#,##0"
        sheet = writer.book.create_sheet("graphique")
        picture = Image(image_path)
        ratio = 1000 / picture.width
        picture.width, picture.height = 1000, int(picture.height * ratio)
        sheet.add_image(picture, "A1")


def run_tresorier(path, as_of=DEMO_DATE, out_folder=None):
    out_folder = out_folder or os.path.dirname(os.path.abspath(path))
    as_of = pd.Timestamp(as_of)
    scope, proba, classes = prepare(path)

    is_open = open_mask(scope, as_of)
    forecast = weekly_forecast(scope[is_open], proba[is_open], classes, as_of, HORIZON_WEEKS)
    alerts, n_open = build_alerts(scope, proba, classes, as_of)
    by_alert = (alerts.groupby("alerte", sort=False).agg(factures=("invoice_id", "count"), montant_ttc=("amount_ttc", "sum")).round(0).reset_index())
    tests = backtest(scope, proba, classes)

    png = os.path.join(out_folder, "graphique_tresorier.png")
    fig = chart_forecast(forecast, as_of, png)
    alerts.to_csv(os.path.join(out_folder, "alertes_tresorier.csv"), index=False, date_format="%Y-%m-%d")
    forecast.to_csv(os.path.join(out_folder, "prevision_8_semaines.csv"), index=False, date_format="%Y-%m-%d")
    export_excel(os.path.join(out_folder, "resultats_tresorier.xlsx"),
                 {"prevision_8_semaines": forecast, "resume_alertes": by_alert, "alertes": alerts, "test_chaque_lundi": tests}, png)

    print(f"[TRÉSORIER] Vue au {as_of:%d/%m/%Y} : {n_open} factures ouvertes, {len(alerts)} alertes")
    print("\n" + by_alert.to_string(index=False))
    print("\n" + forecast.to_string(index=False))
    print("\nTest chaque lundi :\n" + tests.to_string(index=False))
    return dict(forecast=forecast, alerts=alerts, by_alert=by_alert, tests=tests, n_open=n_open, as_of=as_of, fig=fig, scope=scope, proba=proba, classes=classes)


if __name__ == "__main__":
    matplotlib.use("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if a.lower().endswith((".csv", ".xlsx", ".xls"))]
    run_tresorier(args[0] if args else "transformed_dataset.csv")
