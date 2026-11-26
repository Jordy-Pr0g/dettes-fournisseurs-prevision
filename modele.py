"""
Étape 6 du workflow : MODÈLE
Pour chaque facture, prédit la probabilité d'être payée chaque semaine par rapport à son échéance,
uniquement avec ce qu'on sait à la saisie.

Pourquoi des probabilités et pas une seule date : les paiements sont soit à l'heure, soit très en retard.
Une date « moyenne » déplace tout le montant vers une semaine où il n'est pas payé.

Entrée : transformed_dataset.csv, ou resultats_pipeline.xlsx (onglet transformed_dataset)
Sorties : predictions_factures.csv, metriques_factures.csv, apercu_erreur_hebdo.csv, importance_variables.csv

Usage :
    results = run_modele("transformed_dataset.csv")
"""
import os
import sys
import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
OFFSET_MIN, OFFSET_MAX = -2, 10   # semaines par rapport à la semaine d'échéance (au-delà : regroupé)
PREVIEW_WEEKS = ("2026-02-02", "2026-06-22")   # lundis de la première et de la dernière semaine mesurées

# Variables connues au moment de la saisie (voir Data Passport)
CATEGORICAL = ["supplier_id", "supplier_category", "site", "reception_channel",
               "payment_terms", "payment_method", "approver_id", "anomaly_type"]
NUMERIC = ["amount_ht", "terms_days", "days_posting_to_due", "due_weekday", "due_day", "posting_month",
           "supplier_past_late_rate", "supplier_past_mean_delay", "supplier_past_count"]
BOOLEAN = ["supplier_strategic", "needs_director", "has_anomaly", "posted_in_august", "due_at_quarter_end", "has_po"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN

# Connues seulement après coup : ne doivent jamais servir à prédire.
# L'historique du pipeline est exclu aussi : sur l'entraînement, il contient le retard de la facture elle-même.
FORBIDDEN = ["approval_date", "payment_status", "payment_date", "days_late", "is_late", "late_over_30",
             "payment_week", "is_duplicate", "duplicate_of", "week_offset",
             "supplier_hist_late_rate", "supplier_hist_median_delay"]
assert not set(FEATURES) & set(FORBIDDEN), "variable interdite dans le modèle"


def week_start(dates):
    return dates.dt.to_period("W-SUN").dt.start_time


def load_data(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name="transformed_dataset")
    else:
        df = pd.read_csv(path, low_memory=False)
    for col in ["invoice_date", "posting_date", "due_date", "payment_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"[MODÈLE] {len(df)} lignes chargées")
    return df


def add_supplier_history(df):
    """Le passé de chaque fournisseur, tel que connu le jour de la saisie :
    seuls les paiements antérieurs à posting_date comptent, donc aucune information du futur."""
    out = df.copy()
    for col in ["supplier_past_late_rate", "supplier_past_mean_delay", "supplier_past_count"]:
        out[col] = np.nan
    paid = out[out.usable_for_model].sort_values("payment_date")
    for supplier, rows in out.groupby("supplier_id"):
        history = paid[paid.supplier_id == supplier]
        delays = history.days_late.to_numpy(dtype=float)
        n_known = history.payment_date.searchsorted(rows.posting_date, side="left")
        cum_delay = np.concatenate([[0.0], np.cumsum(delays)])
        cum_late = np.concatenate([[0.0], np.cumsum(delays > 0)])
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[rows.index, "supplier_past_mean_delay"] = cum_delay[n_known] / n_known
            out.loc[rows.index, "supplier_past_late_rate"] = cum_late[n_known] / n_known
        out.loc[rows.index, "supplier_past_count"] = n_known
    return out


def add_features(df):
    out = add_supplier_history(df)
    out["due_weekday"] = out.due_date.dt.weekday      # les paiements partent le jeudi
    out["due_day"] = out.due_date.dt.day              # une campagne part aussi en fin de mois
    out["posting_month"] = out.posting_date.dt.month
    out["has_po"] = out.po_number.notna()
    weeks = (week_start(out.payment_date) - week_start(out.due_date)).dt.days // 7
    out["week_offset"] = weeks.clip(OFFSET_MIN, OFFSET_MAX)   # la cible : décalage en semaines
    return out


def build_model():
    # Réglage choisi sur une validation interne (saisies de juillet à décembre 2025), pas sur le test
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    prepare = ColumnTransformer([("categories", encoder, CATEGORICAL)], remainder="passthrough")
    classifier = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
                                                random_state=RANDOM_STATE)
    return Pipeline([("prepare", prepare), ("classifier", classifier)])


def to_matrix(df):
    X = df[FEATURES].copy()
    X[BOOLEAN] = X[BOOLEAN].astype(float)
    return X


def invoice_metrics(name, actual, predicted):
    gap = np.abs(actual - predicted)
    return dict(methode=name, semaine_exacte=round((gap == 0).mean(), 3),
                a_une_semaine_pres=round((gap <= 1).mean(), 3), ecart_moyen_semaines=round(gap.mean(), 2))


def weekly_preview(test, proba, classes):
    """Aperçu de l'objectif du projet : erreur moyenne par semaine, en euros (détaillé aux étapes 7 et 8)."""
    due_week = week_start(test.due_date)
    rule_dates = test.due_date + pd.to_timedelta(test.supplier_hist_median_delay.round(), unit="D")
    model_parts = [pd.DataFrame({"week": (due_week + pd.to_timedelta(7 * int(off), unit="D")).values,
                                 "amount": test.amount_ttc.values * proba[:, j]})
                   for j, off in enumerate(classes)]
    series = {
        "Réel": test.groupby(week_start(test.payment_date)).amount_ttc.sum(),
        "Excel (paiement à l'échéance)": test.groupby(due_week).amount_ttc.sum(),
        "Règle simple (retard habituel du fournisseur)": test.groupby(week_start(rule_dates)).amount_ttc.sum(),
        "Modèle": pd.concat(model_parts).groupby("week").amount.sum(),
    }
    table = pd.DataFrame(series).fillna(0)
    table = table[(table.index >= PREVIEW_WEEKS[0]) & (table.index <= PREVIEW_WEEKS[1])]
    rows = [dict(methode=m, erreur_moyenne_eur_semaine=round((table["Réel"] - table[m]).abs().mean()),
                 erreur_en_pct_du_reel=round((table["Réel"] - table[m]).abs().sum() / table["Réel"].sum() * 100, 1))
            for m in table.columns if m != "Réel"]
    return pd.DataFrame(rows), len(table), table["Réel"].mean()


def run_modele(path, out_folder=None):
    out_folder = out_folder or os.path.dirname(os.path.abspath(path))
    data = add_features(load_data(path))
    usable = data[data.usable_for_model]
    train, test = usable[usable.split == "train"], usable[usable.split == "test"].copy()
    print(f"[MODÈLE] entraînement : {len(train)} factures saisies avant 2026 | test : {len(test)} factures saisies en 2026")

    model = build_model()
    model.fit(to_matrix(train), train.week_offset)
    classes = model.classes_
    proba = model.predict_proba(to_matrix(test))

    # Prédiction par facture
    test["predicted_week_offset"] = classes[proba.argmax(axis=1)]
    test["predicted_payment_week"] = week_start(test.due_date) + pd.to_timedelta(7 * test.predicted_week_offset, unit="D")
    test["proba_late_1_week_plus"] = proba[:, classes >= 1].sum(axis=1).round(3)
    proba_columns = {f"p_semaine_{int(c):+d}": proba[:, j].round(3) for j, c in enumerate(classes)}
    predictions = pd.concat([test[["invoice_id", "supplier_id", "supplier_name", "supplier_category", "amount_ttc",
                                   "posting_date", "due_date", "anomaly_type", "predicted_week_offset",
                                   "predicted_payment_week", "proba_late_1_week_plus", "week_offset", "payment_date"]],
                             pd.DataFrame(proba_columns, index=test.index)], axis=1)

    # Qualité par facture, sur le test
    actual = test.week_offset.to_numpy()
    rule_weeks = ((week_start(test.due_date + pd.to_timedelta(test.supplier_hist_median_delay.round(), unit="D"))
                   - week_start(test.due_date)).dt.days // 7).clip(OFFSET_MIN, OFFSET_MAX).to_numpy()
    metrics = pd.DataFrame([
        invoice_metrics("Excel (paiement à l'échéance)", actual, np.zeros(len(test))),
        invoice_metrics("Règle simple (retard habituel du fournisseur)", actual, rule_weeks),
        invoice_metrics("Modèle", actual, test.predicted_week_offset.to_numpy()),
    ])

    # Le score de risque sert-il à repérer les factures qui partiront en retard ?
    late = actual >= 1
    top = test.proba_late_1_week_plus >= test.proba_late_1_week_plus.quantile(0.8)
    risk = dict(part_retard_ensemble=round(late.mean(), 3), part_retard_top_20_pct=round(late[top.to_numpy()].mean(), 3))

    weekly, n_weeks, mean_real = weekly_preview(test, proba, classes)

    result = permutation_importance(model, to_matrix(test), test.week_offset, n_repeats=3,
                                    random_state=RANDOM_STATE, scoring="neg_log_loss")
    importance = (pd.DataFrame({"variable": FEATURES, "importance": result.importances_mean.round(4)})
                  .sort_values("importance", ascending=False))

    predictions.to_csv(os.path.join(out_folder, "predictions_factures.csv"), index=False, date_format="%Y-%m-%d")
    metrics.to_csv(os.path.join(out_folder, "metriques_factures.csv"), index=False)
    weekly.to_csv(os.path.join(out_folder, "apercu_erreur_hebdo.csv"), index=False)
    importance.to_csv(os.path.join(out_folder, "importance_variables.csv"), index=False)
    excel_path = os.path.join(out_folder, "resultats_modele.xlsx")
    tabs = {"apercu_erreur_hebdo": weekly, "metriques_factures": metrics, "importance_variables": importance,
            "predictions_factures": predictions}
    with pd.ExcelWriter(excel_path, datetime_format="DD/MM/YYYY") as writer:
        for name, frame in tabs.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                longest = frame[col].dropna().astype(str).str.len().max()
                width = max(len(col), 0 if pd.isna(longest) else int(longest)) + 2
                sheet.column_dimensions[get_column_letter(i)].width = min(width, 60)
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    for row in range(2, len(frame) + 2):
                        sheet.cell(row, i).number_format = "DD/MM/YYYY"

    print(f"[MODÈLE] 4 fichiers CSV et resultats_modele.xlsx écrits dans {out_folder}")
    print("\nPar facture :\n" + metrics.to_string(index=False))
    print(f"\nFactures payées avec au moins une semaine de retard : {risk['part_retard_ensemble']:.1%} sur l'ensemble, "
          f"{risk['part_retard_top_20_pct']:.1%} parmi les 20 % jugées les plus risquées")
    print(f"\nAperçu par semaine ({n_weeks} semaines, {mean_real:,.0f} EUR réels en moyenne) :\n" + weekly.to_string(index=False))
    return dict(model=model, metrics=metrics, weekly=weekly, risk=risk, importance=importance,
                predictions=predictions, train=train, test=test)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if a.lower().endswith((".csv", ".xlsx", ".xls"))]
    run_modele(args[0] if args else "transformed_dataset.csv")
