"""
Étape 7 du workflow : PRÉVISION
Calcule les sorties d'argent prévues chaque semaine, à partir des probabilités de l'étape 6.

Par rapport à l'aperçu de l'étape 6, le périmètre s'élargit :
- les avoirs (montants négatifs) sont répartis selon leur décalage historique ;
- les factures en litige restent prévues, car on ignore à la saisie qu'elles ne seront pas payées ;
- les doublons repérés et annulés dès le contrôle ne sont jamais prévus.

Entrée : transformed_dataset.csv, ou resultats_pipeline.xlsx (onglet transformed_dataset)
Sorties : prevision_hebdomadaire.csv, resume_prevision.csv, resultats_prevision.xlsx

Usage :
    results = run_prevision("transformed_dataset.csv")
"""
import os
import sys
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from modele import OFFSET_MIN, OFFSET_MAX, PREVIEW_WEEKS, add_features, build_model, load_data, to_matrix, week_start

EXCEL, RULE, MODEL = "Excel (paiement à l'échéance)", "Règle simple (retard habituel du fournisseur)", "Modèle"


def forecast_scope(data):
    """Tout ce qu'un trésorier voit à la saisie en 2026, sauf les doublons déjà annulés."""
    return data[(data.split == "test") & (data.payment_status != "annulee")].copy()


def spread(rows, proba, offsets):
    """Une ligne par facture et par semaine possible : montant x probabilité."""
    due_week = week_start(rows.due_date)
    parts = [pd.DataFrame({"week": (due_week + pd.to_timedelta(7 * int(off), unit="D")).values,
                           "amount": rows.amount_ttc.values * proba[:, j],
                           "posting_date": rows.posting_date.values})
             for j, off in enumerate(offsets)]
    return pd.concat(parts)


def point(rows, dates):
    """Toute la facture sur une seule semaine."""
    return pd.DataFrame({"week": week_start(dates).values, "amount": rows.amount_ttc.values,
                         "posting_date": rows.posting_date.values})


def credit_note_distribution(train):
    """Décalage historique des avoirs, appris sur la période d'entraînement."""
    offsets = np.arange(OFFSET_MIN, OFFSET_MAX + 1)
    credits = train[(train.doc_type == "avoir") & train.week_offset.notna()]
    return offsets, credits.week_offset.value_counts(normalize=True).reindex(offsets, fill_value=0).to_numpy()


def error_summary(table, name):
    window = (table.index >= PREVIEW_WEEKS[0]) & (table.index <= PREVIEW_WEEKS[1])
    t = table[window]
    return [dict(methode=m, vue=name, erreur_moyenne_eur_semaine=round((t["Réel"] - t[m]).abs().mean()),
                 erreur_en_pct_du_reel=round((t["Réel"] - t[m]).abs().sum() / t["Réel"].sum() * 100, 1),
                 total_prevu_sur_reel=round(t[m].sum() / t["Réel"].sum(), 3)) for m in (EXCEL, RULE, MODEL)]


def run_prevision(path, out_folder=None):
    out_folder = out_folder or os.path.dirname(os.path.abspath(path))
    data = add_features(load_data(path))
    usable = data[data.usable_for_model]
    train = usable[usable.split == "train"]
    scope = forecast_scope(data)
    invoices, credits = scope[scope.doc_type == "facture"], scope[scope.doc_type == "avoir"]
    print(f"[PRÉVISION] {len(invoices)} factures, dont {int((invoices.payment_status == 'en_litige').sum())} en litige, et {len(credits)} avoirs")

    # Même modèle que l'étape 6, entraîné uniquement sur les factures saisies avant 2026
    model = build_model().fit(to_matrix(train), train.week_offset)
    offsets, credit_proba = credit_note_distribution(data[data.split == "train"])

    rule_delay = scope.supplier_hist_median_delay.fillna(train.days_late.median()).round()
    long = {
        EXCEL: point(scope, scope.due_date),
        RULE: point(scope, scope.due_date + pd.to_timedelta(rule_delay, unit="D")),
        MODEL: pd.concat([spread(invoices, model.predict_proba(to_matrix(invoices)), model.classes_),
                          spread(credits, np.tile(credit_proba, (len(credits), 1)), offsets)]),
    }
    paid = scope[scope.payment_status == "payee"]
    real = paid.groupby(week_start(paid.payment_date)).amount_ttc.sum().rename("Réel")

    # Vue 1 : toutes les factures saisies. Vue 2 : seulement celles connues au moins 7 jours avant le début de la semaine.
    all_known = pd.concat([real] + [df.groupby("week").amount.sum().rename(m) for m, df in long.items()], axis=1).fillna(0)
    one_week = pd.concat([real] + [df[df.posting_date + pd.Timedelta(days=7) <= df.week].groupby("week").amount.sum().rename(m)
                                   for m, df in long.items()], axis=1).fillna(0)
    summary = pd.DataFrame(error_summary(all_known, "Toutes les factures saisies")
                           + error_summary(one_week, "Prévision faite 7 jours avant la semaine"))

    window = (all_known.index >= PREVIEW_WEEKS[0]) & (all_known.index <= PREVIEW_WEEKS[1])
    weekly = all_known[window].round(0).rename_axis("semaine_du").reset_index()
    for m in (EXCEL, RULE, MODEL):
        weekly[f"Écart {m}"] = (weekly[m] - weekly["Réel"]).round(0)

    weekly.to_csv(os.path.join(out_folder, "prevision_hebdomadaire.csv"), index=False, date_format="%Y-%m-%d")
    summary.to_csv(os.path.join(out_folder, "resume_prevision.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out_folder, "resultats_prevision.xlsx")) as writer:
        for name, frame in {"resume_prevision": summary, "prevision_hebdomadaire": weekly}.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                longest = frame[col].astype(str).str.len().max()
                sheet.column_dimensions[get_column_letter(i)].width = min(max(len(col), int(longest)) + 2, 48)
                is_date = pd.api.types.is_datetime64_any_dtype(frame[col])
                for row in range(2, len(frame) + 2):
                    if is_date:
                        sheet.cell(row, i).number_format = "DD/MM/YYYY"
                    elif pd.api.types.is_float_dtype(frame[col]) and frame[col].abs().max() > 100:
                        sheet.cell(row, i).number_format = "#,##0"

    print(f"[PRÉVISION] 2 fichiers CSV et resultats_prevision.xlsx écrits dans {out_folder}")
    print(f"\n{int(window.sum())} semaines, {weekly['Réel'].mean():,.0f} EUR réels en moyenne :\n" + summary.to_string(index=False))
    return dict(weekly=weekly, summary=summary, all_known=all_known, one_week=one_week, scope=scope, model=model)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if a.lower().endswith((".csv", ".xlsx", ".xls"))]
    run_prevision(args[0] if args else "transformed_dataset.csv")
