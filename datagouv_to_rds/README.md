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
