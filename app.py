"""
Application web : prévision des décaissements fournisseurs.
Enchaîne le workflow complet (pipeline, modèle, prévision, comparaison, vue du trésorier).

Lancement : streamlit run app.py
"""
import io
import os
import tempfile

import streamlit as st

st.set_page_config(page_title="Prévision des décaissements", layout="wide")
st.title("Prévision des décaissements fournisseurs")
st.caption("Prévoir la semaine probable de paiement de chaque facture, plutôt que supposer que tout est payé à l'échéance. "
           "L'outil prévoit et alerte, le trésorier décide : aucun paiement n'est déclenché.")

# Les bibliothèques de calcul sont longues à charger : le titre s'affiche d'abord
with st.spinner("Chargement de l'application..."):
    import altair as alt
    import matplotlib

    matplotlib.use("Agg")

    import pandas as pd

    import comparaison
    import pipeline
    import tresorier
    from prevision import EXCEL, MODEL, RULE

DEMO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "factures_fournisseurs_synthetique.csv")
COLORS = {"Réel": "#0b0b0b", "Excel": comparaison.COLORS[EXCEL], "Règle simple": comparaison.COLORS[RULE], "Modèle": comparaison.COLORS[MODEL]}
SHORT = {EXCEL: "Excel", RULE: "Règle simple", MODEL: "Modèle"}


@st.cache_data(show_spinner=False)
def analyse(content, filename):
    """Enchaîne tout le workflow une fois par fichier. Le résultat reste en mémoire pour changer de date sans attendre."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, filename)
        with open(path, "wb") as f:
            f.write(content)
        clean = pipeline.run_pipeline(path, out_folder=folder)
        prepared = os.path.join(folder, "transformed_dataset.csv")
        compared = comparaison.run_comparaison(prepared, out_folder=folder)
        scope, proba, classes = tresorier.prepare(prepared)
        tests = tresorier.backtest(scope, proba, classes)
    return dict(quality=clean["quality"], validation=clean["validation"], log=clean["log"], quarantine=clean["quarantine"],
                methods=compared["methods"], tests=compared["tests"], conclusion=compared["conclusion"], table=compared["table"],
                scope=scope, proba=proba, classes=classes, monday_tests=tests)


def to_excel(tabs):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, datetime_format="DD/MM/YYYY") as writer:
        for name, frame in tabs.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            for i, col in enumerate(frame.columns):
                width = max(len(str(col)), int(frame[col].astype(str).str.len().max() or 0) if len(frame) else 0) + 2
                writer.sheets[name].column_dimensions[chr(65 + i)].width = min(width, 60)
    return buffer.getvalue()


def available_mondays(scope):
    first, last = scope.posting_date.min(), scope.posting_date.max()
    return list(pd.date_range(first + pd.Timedelta(days=7 - first.weekday()), last - pd.Timedelta(days=last.weekday()), freq="7D"))


def short_euros(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} M€".replace(".", ",")
    return f"{value / 1000:.0f} k€"


def chart_forecast(forecast, show_real):
    columns = {"prevision_excel": "Excel", "prevision_modele": "Modèle"}
    if show_real:
        columns["paye_en_realite"] = "Réel"
    long = forecast.rename(columns=columns).melt(id_vars="semaine_du", value_vars=list(columns.values()), var_name="Méthode", value_name="Montant")
    long["Semaine"] = long.semaine_du.dt.strftime("%d/%m")
    order = list(columns.values())
    return (alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Semaine:N", sort=None, title="Semaine du", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Méthode:N", sort=order),
        y=alt.Y("Montant:Q", title="Sorties d'argent (€)", axis=alt.Axis(format="~s")),
        color=alt.Color("Méthode:N", sort=order, scale=alt.Scale(domain=order, range=[COLORS[m] for m in order]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Semaine:N", title="Semaine du"), "Méthode:N", alt.Tooltip("Montant:Q", format=",.0f", title="Montant (€)")],
    ).properties(height=340))


def chart_weeks(table, selected):
    long = table.rename(columns=SHORT).rename_axis("Semaine").reset_index().melt(id_vars="Semaine", value_vars=selected, var_name="Méthode", value_name="Montant")
    return (alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=28)).encode(
        x=alt.X("Semaine:T", title=None, axis=alt.Axis(format="%d/%m")),
        y=alt.Y("Montant:Q", title="Sorties d'argent par semaine (€)", axis=alt.Axis(format="~s")),
        color=alt.Color("Méthode:N", sort=selected, scale=alt.Scale(domain=selected, range=[COLORS[m] for m in selected]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Semaine:T", format="%d/%m/%Y"), "Méthode:N", alt.Tooltip("Montant:Q", format=",.0f", title="Montant (€)")],
    ).properties(height=380))


with st.sidebar:
    st.header("Données")
    source = st.radio("Source", ["Données de démonstration", "Importer un fichier"], label_visibility="collapsed")
    if source == "Importer un fichier":
        uploaded = st.file_uploader("Factures brutes (CSV ou Excel)", type=["csv", "xlsx"])
        st.caption("Le fichier doit respecter le Data Passport : mêmes 28 colonnes et mêmes périodes (saisies avant et pendant 2026).")
        if uploaded is None:
            st.info("Choisis un fichier pour lancer l'analyse.")
            st.stop()
        content, filename = uploaded.getvalue(), os.path.basename(uploaded.name)
    else:
        with open(DEMO_FILE, "rb") as f:
            content, filename = f.read(), os.path.basename(DEMO_FILE)
        st.caption("Entreprise fictive, factures simulées sur 24 mois.")

try:
    with st.spinner("Analyse en cours : contrôle des données, modèle, prévision. Cela prend une à deux minutes la première fois."):
        results = analyse(content, filename)
except (ValueError, KeyError) as error:
    st.error("Ce fichier n'a pas pu être analysé. Vérifie qu'il respecte le Data Passport (28 colonnes, dates AAAA-MM-JJ, saisies avant et pendant 2026). "
             "Si tu as ouvert puis réenregistré le CSV dans Excel, repars du fichier d'origine : Excel modifie les séparateurs et les dates.")
    st.caption(f"Détail technique : {error}")
    st.stop()

scope, proba, classes = results["scope"], results["proba"], results["classes"]
mondays = available_mondays(scope)
with st.sidebar:
    st.header("Date de la vue")
    demo_date = pd.Timestamp(tresorier.DEMO_DATE)
    default = mondays.index(demo_date) if demo_date in mondays else len(mondays) - 1
    as_of = st.selectbox("Lundi", mondays, index=default, format_func=lambda d: d.strftime("%d/%m/%Y"), label_visibility="collapsed")
    st.caption("La vue ne regarde que ce qui est connu ce jour-là.")

monday_tab, compare_tab, quality_tab, about_tab = st.tabs(["Lundi matin", "Excel contre modèle", "Qualité des données", "À propos"])

with monday_tab:
    is_open = tresorier.open_mask(scope, as_of)
    forecast = tresorier.weekly_forecast(scope[is_open], proba[is_open], classes, as_of, tresorier.HORIZON_WEEKS)
    alerts, n_open = tresorier.build_alerts(scope, proba, classes, as_of)
    this_week = forecast.iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("Factures ouvertes", f"{n_open:,}".replace(",", " "))
    b.metric("Alertes", f"{len(alerts):,}".replace(",", " "))
    c.metric("Sorties prévues", short_euros(this_week.prevision_modele), help="Pour la semaine qui commence, selon le modèle.")
    d.metric("Selon Excel", short_euros(this_week.prevision_excel), help="Pour la semaine qui commence, en supposant chaque facture payée à son échéance.")

    st.subheader(f"Sorties d'argent prévues à partir du {as_of:%d/%m/%Y}")
    show_real = st.checkbox("Afficher ce qui a été payé en réalité (vérification, inconnu d'un vrai trésorier)")
    st.altair_chart(chart_forecast(forecast, show_real), width="stretch")
    st.caption("Seulement les factures déjà saisies ce lundi : les factures à venir s'y ajouteront.")

    st.subheader("Factures à regarder en priorité")
    types = list(tresorier.ALERTS.values())
    chosen = st.multiselect("Type d'alerte", types, default=types)
    view = alerts[alerts.alerte.isin(chosen)].drop(columns=["payment_date"])
    labels = {"alerte": "Alerte", "invoice_id": "Facture", "supplier_name": "Fournisseur", "supplier_category": "Famille", "amount_ttc": "Montant TTC (€)",
              "due_date": "Échéance", "jours_avant_echeance": "Jours avant échéance", "validee": "Validée", "anomaly_type": "Anomalie",
              "proba_retard": "Risque de retard", "approver_id": "Valideur"}
    view = view.rename(columns=labels)
    st.dataframe(view, hide_index=True, width="stretch", height=380, column_config={
        "Montant TTC (€)": st.column_config.NumberColumn(format="localized"),
        "Échéance": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "Risque de retard": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })
    st.caption("Classées par urgence, puis par montant. Les seuils d'alerte sont des propositions, à valider avec un trésorier.")
    st.download_button("Télécharger la vue en Excel", to_excel({"prevision_8_semaines": forecast, "alertes": view}),
                       file_name=f"vue_tresorier_{as_of:%Y-%m-%d}.xlsx")

with compare_tab:
    st.subheader("Le modèle réduit-il l'erreur de prévision par rapport à Excel ?")
    for line in results["conclusion"]:
        st.write(line)
    st.caption("Sur 21 semaines de factures saisies en 2026, que le modèle n'avait jamais vues.")

    methods = results["methods"].copy()
    methods["methode"] = methods.methode.map(SHORT)
    st.dataframe(methods.rename(columns={"methode": "Méthode", "erreur_moyenne_eur_semaine": "Erreur moyenne par semaine (€)", "erreur_mediane_eur": "Erreur médiane (€)",
                                         "pire_semaine_eur": "Pire semaine (€)", "erreur_en_pct_du_reel": "Erreur en % du réel", "total_prevu_sur_reel": "Total prévu / réel"}),
                 hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="localized") for k in ["Erreur moyenne par semaine (€)", "Erreur médiane (€)", "Pire semaine (€)"]})

    selected = st.multiselect("Courbes à afficher", list(COLORS), default=["Réel", "Excel", "Modèle"])
    if selected:
        st.altair_chart(chart_weeks(results["table"], selected), width="stretch")

    st.subheader("Le gain peut-il venir du hasard ?")
    st.dataframe(results["tests"].rename(columns={"comparaison": "Comparaison", "gain_moyen_eur_semaine": "Gain moyen par semaine (€)", "gain_en_pct": "Gain en %", "semaines_gagnees": "Semaines gagnées",
                                                  "semaines_total": "Semaines", "ic95_bas_eur": "Intervalle 95 % : bas (€)", "ic95_haut_eur": "Intervalle 95 % : haut (€)", "verdict": "Verdict"}),
                 hide_index=True, width="stretch")

    st.subheader("Si la vue avait été utilisée chaque lundi de 2026")
    st.dataframe(results["monday_tests"].rename(columns={"horizon": "Horizon", "reel_moyen": "Réel moyen (€)", "erreur_modele": "Erreur modèle (€)", "erreur_excel": "Erreur Excel (€)",
                                                         "erreur_regle_simple": "Erreur règle simple (€)", "lundis": "Lundis", "modele_bat_excel": "Modèle bat Excel", "modele_bat_regle": "Modèle bat la règle"}),
                 hide_index=True, width="stretch")
    st.caption("Le modèle y garde un avantage en partie parce qu'il répartit les factures déjà en retard sur les semaines restantes, alors qu'Excel et la règle les placent toutes sur la semaine en cours.")

with quality_tab:
    st.subheader("Le bilan du pipeline")
    st.dataframe(results["quality"].rename(columns={"indicateur": "Indicateur", "valeur": "Valeur"}), hide_index=True, width="stretch")
    st.subheader("Les règles non respectées, avant nettoyage")
    validation = results["validation"]
    st.dataframe(validation[validation.failures > 0][["rule_id", "rule", "failures"]].rename(columns={"rule_id": "Règle", "rule": "Description", "failures": "Lignes"}),
                 hide_index=True, width="stretch")
    st.subheader("Les corrections faites")
    st.dataframe(results["log"].rename(columns={"action": "Action", "rows_affected": "Lignes", "detail": "Détail"}), hide_index=True, width="stretch")
    quarantine = results["quarantine"]
    st.subheader(f"La quarantaine : {len(quarantine)} pièces à vérifier par le comptable")
    paid = quarantine[quarantine.payment_status == "payee"].amount_ttc.sum()
    st.write(f"Aucune correction automatique sur ces pièces. Montant déjà payé : {comparaison.euros(paid)}.")
    st.dataframe(quarantine[["invoice_id", "supplier_name", "invoice_date", "amount_ttc", "payment_date", "quarantine_reason"]].rename(
        columns={"invoice_id": "Facture", "supplier_name": "Fournisseur", "invoice_date": "Date de facture", "amount_ttc": "Montant TTC (€)", "payment_date": "Paiement", "quarantine_reason": "Motif"}),
        hide_index=True, width="stretch", column_config={"Date de facture": st.column_config.DateColumn(format="DD/MM/YYYY"), "Paiement": st.column_config.DateColumn(format="DD/MM/YYYY")})

with about_tab:
    st.subheader("Ce que fait cette application")
    st.markdown("""
1. **Contrôle et nettoie** les factures selon les règles du Data Passport, et met de côté ce qu'un humain doit vérifier.
2. **Estime**, pour chaque facture, la probabilité d'un paiement chaque semaine, avec ce qu'on sait à la saisie.
3. **Prévoit** les sorties d'argent de chaque semaine et **compare** le résultat à Excel.
4. **Signale** au trésorier les factures à regarder en priorité.
""")
    st.subheader("Les limites à connaître")
    st.markdown("""
- **Les données sont simulées** : le modèle retrouve surtout les règles utilisées pour les générer. Les gains ne sont pas garantis sur de vraies factures.
- **Une seule période de test**, de 21 semaines : l'avance du modèle sur la règle simple n'est pas démontrée à l'étape de comparaison.
- **Les seuils d'alerte sont des propositions**, absentes du Data Passport.
- **Les dates d'entraînement et de test sont fixes** : un fichier couvrant une autre période demanderait d'adapter le code.
- **L'application n'a pas de comptes utilisateurs** : à ne pas mettre en ligne avec de vraies données d'entreprise.
""")
