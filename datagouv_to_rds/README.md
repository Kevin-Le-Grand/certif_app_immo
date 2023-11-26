# Description de la branch datagouv_to_rds

## Première étape :
- Scraper les données disponibles sur:  
https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/
- Selectionner les variables 
- Nettoyer les données
- Stocker les données dans une table relationnelle sur Aws RDS

## Deuxième étape :
- Créer une Lambda sur Aws qui vvérifie chaque semaine si une mise à jour a été effectuée
- Si une mise à jour a été effectuée alors on récupère les données de la dernière année
- On effectue le même traitement de données qu'à la première étape
- On ajoute les nouvelles données dans la base de données relationnelle

## Connections :
- Les connections à RDS et s3 s'effectue à l'aide du fichier connection.py
- Les variables d'environnements sont stockées dans l'environnement virtuel de la machine 
en local et non publiées sur Github pour des raisons de sécurités

