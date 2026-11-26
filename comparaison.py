"""
Étape 8 du workflow : COMPARAISON
Répond à la question du projet : le modèle réduit-il l'erreur de prévision hebdomadaire par rapport à Excel ?

Entrée : transformed_dataset.csv, ou resultats_pipeline.xlsx (onglet transformed_dataset)
Sorties : comparaison_methodes.csv, tests_comparaison.csv, graphique_semaines.png, graphique_erreurs.png,
          resultats_comparaison.xlsx

Usage :
    results = run_comparaison("transformed_dataset.csv")
"""
import os
import sys
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter
from modele import PREVIEW_WEEKS
from prevision import EXCEL, MODEL, RULE, run_prevision

SEED = 42
N_BOOTSTRAP = 5000
METHODS = [EXCEL, RULE, MODEL]
SHORT = {EXCEL: "Excel", RULE: "Règle simple", MODEL: "Modèle"}
COLORS = {EXCEL: "#2a78d6", RULE: "#eb6834", MODEL: "#1baf7a"}   # palette validée, couleur fixe par méthode
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def euros(x):
    return f"{x:,.0f}".replace(",", " ") + " €"


def compare_methods(table):
    rows = []
    for m in METHODS:
        error = (table["Réel"] - table[m]).abs()
        rows.append(dict(methode=m, erreur_moyenne_eur_semaine=round(error.mean()), erreur_mediane_eur=round(error.median()),
                         pire_semaine_eur=round(error.max()), erreur_en_pct_du_reel=round(error.sum() / table["Réel"].sum() * 100, 1),
                         total_prevu_sur_reel=round(table[m].sum() / table["Réel"].sum(), 3)))
    return pd.DataFrame(rows)


def paired_test(table, challenger, reference):
    """Le gain est-il stable ? On retire au hasard 5 000 fois les semaines, avec remise."""
    gain = ((table["Réel"] - table[reference]).abs() - (table["Réel"] - table[challenger]).abs()).to_numpy()
    rng = np.random.default_rng(SEED)
    means = gain[rng.integers(0, len(gain), (N_BOOTSTRAP, len(gain)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    verdict = "Gain solide" if low > 0 else ("Perte solide" if high < 0 else "Non concluant, l'écart peut venir du hasard")
    return dict(comparaison=f"{SHORT[challenger]} contre {SHORT[reference]}", gain_moyen_eur_semaine=round(gain.mean()),
                gain_en_pct=round(gain.sum() / (table["Réel"] - table[reference]).abs().sum() * 100, 1),
                semaines_gagnees=int((gain > 0).sum()), semaines_total=len(gain),
                ic95_bas_eur=round(low), ic95_haut_eur=round(high), verdict=verdict)


def conclusion(methods, tests):
    m = methods.set_index("methode")
    versus_excel, versus_rule = tests.iloc[0], tests.iloc[1]
    answer = "Oui" if versus_excel.ic95_bas_eur > 0 else "Pas de façon démontrée"
    lines = [
        f"Question : le modèle réduit-il l'erreur de prévision hebdomadaire par rapport à Excel ? {answer}.",
        f"Erreur moyenne par semaine : {euros(m.loc[MODEL, 'erreur_moyenne_eur_semaine'])} pour le modèle, contre {euros(m.loc[EXCEL, 'erreur_moyenne_eur_semaine'])} pour Excel, soit {str(versus_excel.gain_en_pct).replace('.', ',')} % de moins.",
        f"Contre Excel : {versus_excel.verdict.lower()} ({versus_excel.semaines_gagnees} semaines gagnées sur {versus_excel.semaines_total}).",
        f"Contre la règle simple : {versus_rule.verdict.lower()} ({versus_rule.semaines_gagnees} semaines gagnées sur {versus_rule.semaines_total}, gain moyen de {euros(versus_rule.gain_moyen_eur_semaine)}).",
    ]
    return lines


def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_weeks(table, methods, path):
    """Mêmes axes pour les trois méthodes : l'écart au réel se compare d'un coup d'œil."""
    errors = methods.set_index("methode").erreur_moyenne_eur_semaine
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True, dpi=160)
    for ax, m in zip(axes, METHODS):
        ax.plot(table.index, table["Réel"] / 1000, color=INK, linewidth=2, label="Réel", solid_capstyle="round")
        ax.plot(table.index, table[m] / 1000, color=COLORS[m], linewidth=2, label=SHORT[m], solid_capstyle="round")
        ax.set_title(f"{SHORT[m]} : {euros(errors[m])} d'erreur moyenne", loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: MONTHS[mdates.num2date(v).month - 1]))
        style_axis(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2, labelcolor=MUTED, handlelength=1.5)
    axes[0].set_ylabel("Sorties d'argent par semaine (milliers d'euros)", color=MUTED, fontsize=10)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.12)   # place pour la légende
    fig.suptitle("Le modèle suit les sorties d'argent réelles de plus près qu'Excel", x=0.012, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.005, f"Factures saisies en 2026, {len(table)} semaines, du 02/02 au 28/06/2026. Données synthétiques.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return fig


def chart_errors(methods, path):
    data = methods.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=160)
    bars = ax.barh([SHORT[m] for m in data.methode], data.erreur_moyenne_eur_semaine / 1000,
                   color=[COLORS[m] for m in data.methode], height=0.42)
    for bar, value in zip(bars, data.erreur_moyenne_eur_semaine):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2, euros(value), va="center", fontsize=11, color=INK)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(axis="y", length=0, colors=INK, labelsize=11)
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, data.erreur_moyenne_eur_semaine.max() / 1000 * 1.22)
    ax.set_title("Erreur moyenne de prévision par semaine : plus c'est court, mieux c'est", loc="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return fig


def export_excel(path, tabs, images):
    with pd.ExcelWriter(path) as writer:
        for name, frame in tabs.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for i, col in enumerate(frame.columns, start=1):
                longest = frame[col].astype(str).str.len().max()
                sheet.column_dimensions[get_column_letter(i)].width = min(max(len(col), int(longest)) + 2, 90)
                is_date = pd.api.types.is_datetime64_any_dtype(frame[col])
                for row in range(2, len(frame) + 2):
                    if is_date:
                        sheet.cell(row, i).number_format = "DD/MM/YYYY"
                    elif pd.api.types.is_numeric_dtype(frame[col]) and frame[col].abs().max() > 1000:
                        sheet.cell(row, i).number_format = "#,##0"
        sheet = writer.book.create_sheet("graphiques")
        row = 1
        for image_path, height_rows in images:
            picture = Image(image_path)
            ratio = 1000 / picture.width
            picture.width, picture.height = 1000, int(picture.height * ratio)
            sheet.add_image(picture, f"A{row}")
            row += height_rows


def run_comparaison(path, out_folder=None):
    out_folder = out_folder or os.path.dirname(os.path.abspath(path))
    forecast = run_prevision(path, out_folder=out_folder)
    all_known = forecast["all_known"]
    table = all_known[(all_known.index >= PREVIEW_WEEKS[0]) & (all_known.index <= PREVIEW_WEEKS[1])]

    methods = compare_methods(table)
    tests = pd.DataFrame([paired_test(table, MODEL, EXCEL), paired_test(table, MODEL, RULE), paired_test(table, RULE, EXCEL)])
    lines = conclusion(methods, tests)

    weeks_png, errors_png = os.path.join(out_folder, "graphique_semaines.png"), os.path.join(out_folder, "graphique_erreurs.png")
    fig_weeks = chart_weeks(table, methods, weeks_png)
    fig_errors = chart_errors(methods, errors_png)

    methods.to_csv(os.path.join(out_folder, "comparaison_methodes.csv"), index=False)
    tests.to_csv(os.path.join(out_folder, "tests_comparaison.csv"), index=False)
    export_excel(os.path.join(out_folder, "resultats_comparaison.xlsx"),
                 {"conclusion": pd.DataFrame({"conclusion": lines}), "comparaison_methodes": methods, "tests_comparaison": tests,
                  "semaine_par_semaine": forecast["weekly"]},
                 [(errors_png, 18), (weeks_png, 22)])

    print("\n===== CONCLUSION =====\n" + "\n".join(lines))
    return dict(methods=methods, tests=tests, conclusion=lines, table=table, weekly=forecast["weekly"],
                fig_weeks=fig_weeks, fig_errors=fig_errors)


if __name__ == "__main__":
    matplotlib.use("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if a.lower().endswith((".csv", ".xlsx", ".xls"))]
    run_comparaison(args[0] if args else "transformed_dataset.csv")
