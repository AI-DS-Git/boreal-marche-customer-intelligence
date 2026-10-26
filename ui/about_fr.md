## Fiche du modèle — Tête A (rétention)
**Objectif.** Estimer la probabilité qu'un client passe une nouvelle commande dans les 90 jours, afin que le personnel décide s'il révise une offre de rétention. Le modèle appuie une décision humaine; il n'agit pas seul.

**Modèle.** Classificateur Extra Trees optimisé avec calibration isotonique, livré dans un seul artefact (prétraitement + modèle + calibrateur). Version `v0.3-d3`, scikit-learn 1.8.0.

**Données d'entraînement.** 5 249 clients (entraînement 3 674 · calibration 787 · test retenu 788), 28 caractéristiques comportementales construites à partir des transactions (récence, fréquence, dépenses, variété de produits, rythme, indicateur de pays).

**Performance (test retenu).** PR-AUC 0,875 · ROC-AUC 0,730. Règle de décision : offre si P(rétention) < 0,82, choisie pour maximiser le profit attendu (environ 1 066 $ contre 201 $ pour un coupon généralisé sur le jeu de test, selon les hypothèses de coûts énoncées).

**Limites.** Pouvoir de classement modéré (ROC-AUC 0,73). Entraîné sur l'historique 2009–2011 d'un seul détaillant; les comportements ont pu changer. Un audit par sous-groupes montre que les clients à faibles ou moyennes dépenses et les nouveaux clients n'obtiennent jamais plus d'environ 0,52 et passent tous sous le seuil de 0,82; c'est pourquoi chaque offre est révisée par une personne. Ne convient pas à des décisions automatisées visant directement le client.

## Avis de confidentialité
Nous utilisons votre historique d'achats pour estimer si vous risquez de cesser d'acheter et pour vous suggérer des produits. Cette évaluation est faite par un outil automatisé. Un membre du personnel examine chaque offre de fidélisation avant son envoi, et l'outil ne prend jamais seul une décision à votre sujet. Vous pouvez nous demander quels renseignements ont été utilisés et les principaux motifs d'un résultat, faire corriger ces renseignements et faire réviser par une personne toute décision vous concernant. Pour exercer vos droits, communiquez avec notre responsable de la protection des renseignements personnels.

Ce prototype repose sur un jeu de données de vente au détail public et dépersonnalisé. Il ne recueille aucun renseignement personnel auprès des visiteurs. Les requêtes envoyées sont journalisées (valeurs des caractéristiques, prédiction, durée, identifiant de requête) pour surveiller la qualité du modèle; aucun nom, courriel ni adresse n'est conservé. L'assistant refuse de communiquer ou de consulter les données d'une autre personne. En vertu de la Loi 25 du Québec, les clients doivent être informés lorsqu'un outil automatisé éclaire une décision les concernant et peuvent demander une révision humaine; un déploiement réel ajouterait la saisie du consentement, des durées de conservation et un responsable de la protection des renseignements personnels.

## Provenance des données
Dépôt UCI Machine Learning, *Online Retail II* (détaillant en ligne britannique, de décembre 2009 à décembre 2011, licence publique). Les transactions ont été nettoyées (annulations et lignes hors produits retirées), puis agrégées en une ligne par client. Les règles de recommandation ont été extraites des paniers (351 règles après déduplication et filtre de lift). Les textes de politiques de l'assistant sont des exemples de prototype, non de vraies politiques d'entreprise.

## Architecture
```
Interface Streamlit ──HTTP──> Service FastAPI ──> Modèle Tête A (joblib)
                                │   ├──────────> Moteur de règles Tête B (351 règles)
                                │   └──────────> Assistant Tête C (fournisseur par règles; LLM à brancher)
                                └──> PostgreSQL : prédictions, requêtes, alertes, résultats différés
Surveillance : dérive PSI/KS, latence, erreurs, alertes (+ webhook), rattrapage de la vérité terrain à 90 jours
CI/CD : GitHub Actions -> image Docker (étiquette = SHA du commit) -> AWS
```
