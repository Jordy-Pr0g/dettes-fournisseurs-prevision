# Prévision des décaissements fournisseurs

Projet du cours IA & Automatisation. L'application prévoit les sorties d'argent liées aux factures fournisseurs
et signale au trésorier les factures à regarder en priorité. Elle prévoit et alerte : aucun paiement n'est déclenché.

## La question du projet

Un workflow automatisé qui estime la semaine probable de paiement de chaque facture fournisseur peut-il réduire
l'erreur de prévision des décaissements hebdomadaires, par rapport à un suivi Excel fondé sur les dates d'échéance ?

## Lancer l'application

Sous Windows : double-clique sur `lancer_application.bat`. Python doit être installé.

Sinon :

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Le workflow

| Fichier | Rôle |
|---|---|
| `pipeline.py` | SOURCE, VALIDATE, CLEAN, TRANSFORM, OUTPUT |
| `modele.py` | Probabilité de paiement par semaine pour chaque facture |
| `prevision.py` | Sorties d'argent prévues chaque semaine |
| `comparaison.py` | Comparaison avec Excel et une règle simple, tests et graphiques |
| `tresorier.py` | Vue du lundi matin et alertes |
| `app.py` | L'application web |

## Résultats sur les données de démonstration

Erreur moyenne par semaine, sur 21 semaines de factures saisies en 2026 : 322 847 € pour Excel, 125 483 € pour la règle simple,
103 536 € pour le modèle. Le gain sur Excel est solide. L'avance sur la règle simple n'est pas concluante.

## Limites

- Les données sont simulées et l'entreprise est fictive. Les gains ne sont pas garantis sur de vraies factures.
- Une seule période de test. Les dates d'entraînement et de test sont fixes dans le code.
- Les seuils d'alerte sont des propositions, à valider avec un trésorier.
- L'application n'a pas de comptes utilisateurs : ne pas la mettre en ligne avec de vraies données d'entreprise.
