<div style="text-align:center; background-color:blue; padding:10px;">
  <h1 style="color:white;">Description de la branch datagouv_to_rds</h1>
</div>
<br>

<div style="text-align:left; background-color:gray; padding:0px;">
  <h1 style="color:white;">Mode opératoire :</h1>
</div>

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

<br>
<div style="text-align:left; background-color:gray; padding:0px;">
  <h1 style="color:white;">Liste des fichiers :</h1>
</div>
<br>

- Notebook :
    - Test_scarping_rds_s3 : Test de scrapping et api pour récupérer les données disponible sur data.gouv 
        et les connections avec aws
    - EDA : Etude pour le nettoyage des données avant insertion en base de données et test slqalchemy
    loading_first_data_in_rds : Insertion de toutes les données dans RDS
- Fichier python :
    - connection :  Permet de créer les différentes connexions (aws, smtp)
    - data_processing : contient les fonctions pour le nettoyage de données
    - maj : Permet d'ajouter les dernières ventes si une mise à jour a été effectuée sur data.gouv
    - lambda : Permet de vérifier si une mise à jour est disponible (si oui envoie de mail) 
- Schéma sql :
    - create_tables_sql : Schéma de la base de données finale
    - test : Schéma d'une base de données pour test avec sqlalchemy

## Méthode pour récupérer les données et les stocker dans aws
- Préparation AWS :
    - Créer un utilisateur IAM
    - Créer une table RDS mysql
    - Créer un bucket s3
    - Définir les régles de sécurité avec EC2
- En local:
    - placer vous dans un dossier puis : cloner la branche datagouv_to_rds
    - Créer un environnement virtuel venv
    - Installer les dépendances avec le requirements.txt
    - Créer dans .venv/ un fichier .local pour stocker vos identifiants et mots de passe des services
    - Dans connection.py remplacer le chemin vers le fichier .local
- Execution :
    - Tester les connections et le scraping en executant le notebook Test_scraping_rds_s3
    - Executer les cellules du notebook loading_first_data_in_rds pour insérer les données dans rds
- Optionnel :
    - Si vous voulez que les données se mettent à jour automatiquement. 
    Sur Aws créer une lambda function avec le fichier lambda.py et 
    créer un événement EventBridge pour vérifier s'il y a une mise à jour
    - Lors d'une mise à jour executer le fichier maj.py pour ajouter les nouvelles données

    
