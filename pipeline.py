"""
Pipeline de données : SOURCE → VALIDATE → CLEAN → TRANSFORM → OUTPUT
Projet : prévision des décaissements fournisseurs (Distrilor SAS, données synthétiques)

Source de vérité : Data Contract = Data_Passport_factures_fournisseurs.xlsx
Les règles marquées [SUGGESTION] ne figurent pas dans le Data Contract : elles sont proposées, pas imposées.

Usage :
    from pipeline import run_pipeline
    results = run_pipeline("factures_fournisseurs_synthetique.csv")
ou en ligne de commande :
    python pipeline.py factures_fournisseurs_synthetique.csv
"""
import os
import sys
import pandas as pd
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# Data Contract (repris du Data Passport, version 1.0)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "invoice_id", "supplier_invoice_ref", "doc_type", "supplier_id", "supplier_name",
    "supplier_category", "supplier_strategic", "site", "po_number", "po_amount_ht",
    "invoice_date", "reception_channel", "reception_date", "posting_date", "payment_terms",
    "due_date", "amount_ht", "vat_rate", "amount_ttc", "payment_method", "approver_id",
    "anomaly_type", "approval_date", "payment_status", "payment_date", "days_late",
    "is_duplicate", "duplicate_of",
]
DATE_COLUMNS = ["invoice_date", "reception_date", "posting_date", "due_date", "approval_date", "payment_date"]
NUMERIC_COLUMNS = ["po_amount_ht", "amount_ht", "vat_rate", "amount_ttc", "days_late"]

ALLOWED = {
    "doc_type": ["facture", "avoir"],
    "supplier_category": ["Marchandises", "Transport", "Energie & telecoms", "Interim",
                          "Services generaux", "Prestations intellectuelles", "Maintenance"],
    "site": ["Lyon", "Paris", "Lille", "Nantes"],
    "reception_channel": ["email_pdf", "papier", "facture_electronique"],
    "payment_terms": ["30j_net", "45j_fin_de_mois", "60j_net", "15j_prelevement"],
    "vat_rate": [0.055, 0.10, 0.20],
    "payment_method": ["virement", "prelevement"],
    "anomaly_type": ["aucune", "ecart_prix", "ecart_quantite", "bc_manquant", "doublon"],
    "payment_status": ["payee", "en_litige", "annulee"],
}
PERIOD = ("2024-07-01", "2026-06-30")
DIRECTOR_THRESHOLD = 10_000
TRAIN_END = "2026-01-01"
# Fenêtre d'évaluation de la baseline : semaines complètes (lundi-dimanche) du 1er semestre 2026.
# Après juin, plus aucune nouvelle facture n'entre dans le fichier : ces semaines fausseraient la mesure.
EVAL_WEEKS = ("2026-01-05", "2026-06-28")

# Champs connus seulement après coup : interdits comme variables explicatives
POST_EVENT_FIELDS = ["approval_date", "payment_status", "payment_date", "days_late", "is_duplicate", "duplicate_of"]


# ---------------------------------------------------------------------------
# Fonctions utilitaires (conventions de calcul)
# ---------------------------------------------------------------------------
def theoretical_due_date(invoice_date, terms):
    """Échéance attendue selon payment_terms.
    Convention retenue pour « 45 jours fin de mois » : date + 45 jours, puis fin du mois."""
    if terms == "30j_net":
        return invoice_date + pd.Timedelta(days=30)
    if terms == "60j_net":
        return invoice_date + pd.Timedelta(days=60)
    if terms == "15j_prelevement":
        return invoice_date + pd.Timedelta(days=15)
    if terms == "45j_fin_de_mois":
        return (invoice_date + pd.Timedelta(days=45)) + pd.offsets.MonthEnd(0)
    return pd.NaT


def is_payment_run_day(d):
    """Jour de campagne de paiement : jeudi ou dernier jour ouvré du mois."""
    last_bday = d + pd.offsets.BMonthEnd(0)
    return d.weekday() == 3 or d == last_bday


# ---------------------------------------------------------------------------
# 1. SOURCE
# ---------------------------------------------------------------------------
def load_source(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name="Donnees", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    print(f"[SOURCE] {len(df)} lignes, {df.shape[1]} colonnes")
    print(df.iloc[:3, :6].to_string())  # aperçu : 3 lignes, 6 premières colonnes
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        raise ValueError(f"Colonnes attendues absentes : {missing}")
    if extra:
        print(f"[SOURCE] Colonnes non prévues par le contrat (ignorées) : {extra}")
    return df[EXPECTED_COLUMNS].copy()


def to_typed(df):
    """Conversion des types selon le Data Contract (texte → dates, nombres, booléens)."""
    out = df.copy()
    for c in DATE_COLUMNS:
        # [:10] : Excel renvoie "2024-07-01 00:00:00", le CSV "2024-07-01"
        out[c] = pd.to_datetime(out[c].str[:10], format="%Y-%m-%d", errors="coerce")
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["supplier_strategic", "is_duplicate"]:
        out[c] = out[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return out


# ---------------------------------------------------------------------------
# 2. VALIDATE : chaque règle renvoie un masque des lignes en échec
# ---------------------------------------------------------------------------
def build_rules(d):
    ttc_calc = (d.amount_ht * (1 + d.vat_rate)).round(2)
    due_calc = pd.Series([theoretical_due_date(i, t) for i, t in zip(d.invoice_date, d.payment_terms)], index=d.index)
    delay_cap = d.payment_terms.map({"45j_fin_de_mois": 76}).fillna(60)
    main_name = d.groupby("supplier_id").supplier_name.transform(lambda s: s.value_counts().index[0])
    virement_paid = (d.payment_method == "virement") & d.payment_date.notna()
    run_day = d.payment_date.apply(lambda x: is_payment_run_day(x) if pd.notna(x) else True)
    status_needs_date = d.payment_status == "payee"
    late_calc = (d.payment_date - d.due_date).dt.days

    return [
        ("R01", "invoice_id unique et non vide", d.invoice_id.isna() | d.invoice_id.duplicated(keep=False)),
        ("R02", "supplier_invoice_ref non vide", d.supplier_invoice_ref.isna() | (d.supplier_invoice_ref.str.strip() == "")),
        ("R03", "doc_type dans la liste", ~d.doc_type.isin(ALLOWED["doc_type"])),
        ("R04", "avoir => montant négatif, facture => montant positif",
         ((d.doc_type == "avoir") & (d.amount_ht >= 0)) | ((d.doc_type == "facture") & (d.amount_ht <= 0))),
        ("R05", "supplier_id au format FRN-NNN", ~d.supplier_id.fillna("").str.fullmatch(r"FRN-\d{3}")),
        ("R06", "un seul supplier_name par supplier_id (lignes au nom divergent)", d.supplier_name != main_name),
        ("R07", "supplier_category exactement dans la liste", ~d.supplier_category.isin(ALLOWED["supplier_category"])),
        ("R08", "site dans la liste", ~d.site.isin(ALLOWED["site"])),
        ("R09", "po_amount_ht renseigné si et seulement si po_number renseigné", d.po_number.notna() != d.po_amount_ht.notna()),
        ("R10", "Marchandises : bon de commande obligatoire (sauf anomalie bc_manquant)",
         (d.supplier_category == "Marchandises") & d.po_number.isna() & (d.anomaly_type != "bc_manquant")),
        ("R11", "invoice_date valide et dans la période", d.invoice_date.isna() | ~d.invoice_date.between(*PERIOD)),
        ("R12", "reception_channel dans la liste", ~d.reception_channel.isin(ALLOWED["reception_channel"])),
        ("R13", "reception_date renseignée et >= invoice_date", d.reception_date.isna() | (d.reception_date < d.invoice_date)),
        ("R14", "posting_date >= reception_date et jour ouvré",
         d.posting_date.isna() | (d.posting_date < d.reception_date) | (d.posting_date.dt.weekday >= 5)),
        ("R15", "payment_terms dans la liste", ~d.payment_terms.isin(ALLOWED["payment_terms"])),
        ("R16", "due_date cohérente avec payment_terms et sous le plafond légal",
         d.due_date.isna() | (d.due_date != due_calc) | ((d.due_date - d.invoice_date).dt.days > delay_cap)),
        ("R17", "vat_rate dans {0.055 ; 0.10 ; 0.20}", ~d.vat_rate.round(3).isin(ALLOWED["vat_rate"])),
        ("R18", "amount_ttc = amount_ht x (1 + vat_rate) à 0,01 près", d.vat_rate.isna() | ((d.amount_ttc - ttc_calc).abs() > 0.011)),
        ("R19", "prelevement si et seulement si 15j_prelevement",
         (d.payment_method == "prelevement") != (d.payment_terms == "15j_prelevement")),
        ("R20", "APP-DIR obligatoire si amount_ht > 10 000 EUR", (d.amount_ht > DIRECTOR_THRESHOLD) & (d.approver_id != "APP-DIR")),
        ("R21", "anomaly_type dans la liste", ~d.anomaly_type.isin(ALLOWED["anomaly_type"])),
        ("R22", "approval_date >= posting_date", d.approval_date.isna() | (d.approval_date < d.posting_date)),
        ("R23", "payment_status dans la liste et payment_date vide si et seulement si en_litige ou annulee",
         ~d.payment_status.isin(ALLOWED["payment_status"]) | (status_needs_date != d.payment_date.notna())),
        ("R24", "virement : payment_date >= approval_date et jour de campagne (jeudi ou dernier jour ouvré)",
         virement_paid & ((d.payment_date < d.approval_date) | ~run_day)),
        ("R25", "days_late = payment_date - due_date", d.payment_date.notna() & (d.days_late != late_calc)),
        ("R26", "is_duplicate => duplicate_of renseigné et existant",
         (d.is_duplicate == True) & (d.duplicate_of.isna() | ~d.duplicate_of.isin(d.invoice_id))),
    ]


def suggestion_rules(d):
    """[SUGGESTION] Règle hors Data Contract : même fournisseur, même date et même montant = doublon probable.
    On garde la pièce saisie en premier et on signale les suivantes,
    sauf celles que le contrôle actuel a déjà repérées (anomaly_type = doublon, statut annulee)."""
    key = ["supplier_id", "invoice_date", "amount_ht"]
    ordered = d.sort_values(["posting_date", "invoice_id"])
    later_copy = ordered.duplicated(key, keep="first").reindex(d.index)
    probable = later_copy & (d.doc_type == "facture") & (d.anomaly_type != "doublon")
    return [("S01", "[SUGGESTION] doublon probable non repéré par le contrôle actuel", probable)]


def validate(d, stage):
    rows = []
    for rule_id, label, mask in build_rules(d) + suggestion_rules(d):
        mask = mask.fillna(True)
        failed = d.loc[mask, "invoice_id"].tolist()
        rows.append(dict(stage=stage, rule_id=rule_id, rule=label, failures=len(failed),
                         affected_rows=", ".join(failed[:15]) + (" ..." if len(failed) > 15 else "")))
    report = pd.DataFrame(rows)
    print(f"[VALIDATE {stage}] {int((report.failures > 0).sum())} règles en échec, "
          f"{int(report.failures.sum())} échecs au total")
    return report


# ---------------------------------------------------------------------------
# 3. CLEAN : corrections explicables + quarantaine
# ---------------------------------------------------------------------------
# Règles dont l'échec rend la ligne inutilisable sans intervention humaine
BLOCKING_RULES = ["R01", "R03", "R04", "R11", "R15", "R16", "R17", "R18", "R23", "R25", "S01"]
# Règles signalées mais non bloquantes pour la prévision (ligne conservée avec un indicateur)
FLAG_RULES = {"R10": "flag_po_manquant", "R13": "flag_reception_manquante", "R09": "flag_po_incoherent"}


def clean(d):
    log = []
    d = d.copy()

    def note(action, n, detail):
        log.append(dict(action=action, rows_affected=int(n), detail=detail))

    # a) doublons exacts (lignes strictement identiques)
    n0 = len(d)
    d = d.drop_duplicates()
    note("Suppression des doublons exacts", n0 - len(d), "lignes strictement identiques")

    # b) textes : espaces superflus
    for c in ["supplier_invoice_ref", "supplier_name", "supplier_category", "site", "approver_id"]:
        before = d[c].copy()
        d[c] = d[c].str.strip()
        note(f"Suppression des espaces dans {c}", (before != d[c]).sum(), "espaces en début ou fin")

    # c) libellés de catégorie : correspondance insensible à la casse avec la liste du contrat
    canon = {c.lower(): c for c in ALLOWED["supplier_category"]}
    before = d.supplier_category.copy()
    d["supplier_category"] = d.supplier_category.str.lower().map(canon).fillna(d.supplier_category)
    note("Standardisation de supplier_category", (before != d.supplier_category).sum(),
         "casse harmonisée sur la liste du Data Contract")

    # d) un seul nom par fournisseur : on retient le nom le plus fréquent
    main_name = d.groupby("supplier_id").supplier_name.agg(lambda s: s.value_counts().index[0])
    before = d.supplier_name.copy()
    d["supplier_name"] = d.supplier_id.map(main_name)
    note("Harmonisation de supplier_name", (before != d.supplier_name).sum(), "nom le plus fréquent du supplier_id")

    # e) TVA manquante : recalculée à partir de TTC / HT si le résultat est un taux autorisé
    miss = d.vat_rate.isna() & d.amount_ht.ne(0)
    implied = (d.amount_ttc / d.amount_ht - 1).round(3)
    fixable = miss & implied.isin(ALLOWED["vat_rate"])
    d.loc[fixable, "vat_rate"] = implied[fixable]
    note("Recalcul de vat_rate manquant", fixable.sum(), "vat_rate = amount_ttc / amount_ht - 1 (taux autorisé)")

    # f) indicateurs pour les règles non bloquantes
    rules = dict((r[0], r[2].fillna(True)) for r in build_rules(d) + suggestion_rules(d))
    for rule_id, flag in FLAG_RULES.items():
        d[flag] = rules[rule_id]
        note(f"Indicateur {flag}", d[flag].sum(), f"règle {rule_id} en échec : ligne conservée et signalée")

    # g) quarantaine : lignes en échec sur une règle bloquante après corrections
    reasons = pd.Series("", index=d.index)
    for rule_id in BLOCKING_RULES:
        reasons[rules[rule_id]] += rule_id + " "
    to_quarantine = reasons.str.len() > 0
    quarantine = d[to_quarantine].assign(quarantine_reason=reasons[to_quarantine].str.strip())
    clean_df = d[~to_quarantine].copy()
    note("Mise en quarantaine", to_quarantine.sum(), "échec d'au moins une règle bloquante : " + ", ".join(BLOCKING_RULES))

    print(f"[CLEAN] {len(clean_df)} lignes propres, {len(quarantine)} en quarantaine")
    return clean_df, quarantine, pd.DataFrame(log), n0 - len(d.drop_duplicates())


# ---------------------------------------------------------------------------
# 4. TRANSFORM : variables du projet
# ---------------------------------------------------------------------------
def transform(d):
    t = d.copy()
    # Durée de crédit contractuelle
    t["terms_days"] = (t.due_date - t.invoice_date).dt.days
    # Temps restant entre la saisie et l'échéance pour valider et payer
    t["days_posting_to_due"] = (t.due_date - t.posting_date).dt.days
    # Signature du directeur requise
    t["needs_director"] = t.amount_ht > DIRECTOR_THRESHOLD
    # Anomalie détectée au rapprochement
    t["has_anomaly"] = t.anomaly_type != "aucune"
    # Validation plus lente en août
    t["posted_in_august"] = t.posting_date.dt.month == 8
    # Échéance dans les 7 derniers jours d'un trimestre (rétention de trésorerie possible)
    t["due_at_quarter_end"] = t.due_date.dt.month.isin([3, 6, 9, 12]) & (t.due_date.dt.day >= 24)
    # Découpage chronologique
    t["split"] = (t.posting_date < TRAIN_END).map({True: "train", False: "test"})
    # Cibles
    t["is_late"] = t.days_late > 0
    t["late_over_30"] = t.days_late > 30
    # Semaines : prévision baseline (échéance) et décaissement réel (paiement)
    t["due_week"] = t.due_date.dt.to_period("W-SUN").astype(str)
    t["payment_week"] = t.payment_date.dt.to_period("W-SUN").astype(str).where(t.payment_date.notna())

    # Historique fournisseur calculé uniquement sur la période d'entraînement (pas de fuite de données)
    hist = t[(t.split == "train") & (t.doc_type == "facture") & (t.payment_status == "payee")]
    late_rate = hist.groupby("supplier_id").is_late.mean()
    med_delay = hist.groupby("supplier_id").days_late.median()
    t["supplier_hist_late_rate"] = t.supplier_id.map(late_rate).fillna(hist.is_late.mean()).round(3)
    t["supplier_hist_median_delay"] = t.supplier_id.map(med_delay).fillna(hist.days_late.median())

    # Lignes exploitables pour entraîner / tester un modèle de date de paiement
    t["usable_for_model"] = (t.doc_type == "facture") & (t.payment_status == "payee")

    # Tableau hebdomadaire : décaissements réels vs prévision Excel (baseline Z)
    paid = t[t.payment_status == "payee"]
    real = paid.groupby("payment_week").amount_ttc.sum().rename("actual_outflow")
    base = paid.groupby("due_week").amount_ttc.sum().rename("baseline_forecast")
    weekly = pd.concat([real, base], axis=1).fillna(0).sort_index()
    weekly["baseline_abs_error"] = (weekly.actual_outflow - weekly.baseline_forecast).abs()
    weekly.index.name = "week"
    print(f"[TRANSFORM] {t.shape[1] - d.shape[1]} variables créées, {len(weekly)} semaines dans le tableau baseline")
    return t, weekly.reset_index()


# ---------------------------------------------------------------------------
# 5. OUTPUT
# ---------------------------------------------------------------------------
def export(outputs, folder):
    os.makedirs(folder, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(folder, f"{name}.csv"), index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    # Classeur Excel : un onglet par sortie (Excel en français ouvre mal les CSV à virgules)
    excel_path = os.path.join(folder, "resultats_pipeline.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, frame in outputs.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                letter = get_column_letter(i)
                longest = frame[col].dropna().astype(str).str.len().max()
                width = max(len(col), 0 if pd.isna(longest) else int(longest)) + 2
                sheet.column_dimensions[letter].width = min(width, 60)
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    for cell in sheet[letter][1:]:
                        cell.number_format = "DD/MM/YYYY"
    print(f"[OUTPUT] {len(outputs)} fichiers CSV et resultats_pipeline.xlsx écrits dans {folder}")


def baseline_mean_error(weekly):
    """Erreur moyenne de la prévision Excel sur les semaines de la fenêtre d'évaluation."""
    in_window = (weekly.week.str[:10] >= EVAL_WEEKS[0]) & (weekly.week.str[11:] <= EVAL_WEEKS[1])
    return round(weekly[in_window].baseline_abs_error.mean(), 0)


def run_pipeline(path, out_folder=None):
    out_folder = out_folder or os.path.join(os.path.dirname(os.path.abspath(path)), "sorties_pipeline_v1.1")
    raw = load_source(path)
    typed = to_typed(raw)
    report_before = validate(typed, "avant nettoyage")
    clean_df, quarantine, log, exact_dups = clean(typed)
    if clean_df.empty:
        raise ValueError("Aucune ligne propre : vérifie le fichier d'entrée (colonnes, format des dates).")
    report_after = validate(clean_df.drop(columns=list(FLAG_RULES.values())), "après nettoyage")
    transformed, weekly = transform(clean_df)

    quality = pd.DataFrame([
        ("Lignes d'origine", len(raw)),
        ("Lignes propres", len(clean_df)),
        ("Lignes en quarantaine", len(quarantine)),
        ("Doublons exacts supprimés", exact_dups),
        ("Échecs de validation avant nettoyage", int(report_before.failures.sum())),
        ("Échecs de validation après nettoyage", int(report_after.failures.sum())),
        ("Lignes utilisables pour le modèle", int(transformed.usable_for_model.sum())),
        ("Erreur moyenne baseline Excel (EUR/semaine, du 05/01/2026 au 28/06/2026)", baseline_mean_error(weekly)),
    ], columns=["indicateur", "valeur"])

    export({
        "data_quality_report": quality,
        "validation_report": pd.concat([report_before, report_after]),
        "cleaning_log": log,
        "quarantine": quarantine,
        "weekly_baseline": weekly,
        "clean_dataset": clean_df,
        "transformed_dataset": transformed,
    }, out_folder)

    print("\n===== RÉSUMÉ =====")
    print(f"Lignes d'origine         : {len(raw)}")
    print(f"Lignes propres           : {len(clean_df)}")
    print(f"Lignes en quarantaine    : {len(quarantine)}")
    print(f"Doublons exacts supprimés: {exact_dups}")
    print(f"Échecs de validation     : {int(report_before.failures.sum())} avant, "
          f"{int(report_after.failures.sum())} après nettoyage")
    return dict(clean=clean_df, quarantine=quarantine, validation=report_before, validation_after=report_after,
                log=log, quality=quality, transformed=transformed, weekly=weekly)


if __name__ == "__main__":
    # Terminal Windows : forcer l'UTF-8 pour les accents (inutile et absent dans Colab / Jupyter)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Dans Colab, sys.argv contient des arguments internes (ex. "-f") : on ne garde qu'un vrai fichier de données
    args = [a for a in sys.argv[1:] if a.lower().endswith((".csv", ".xlsx", ".xls"))]
    run_pipeline(args[0] if args else "factures_fournisseurs_synthetique.csv")
