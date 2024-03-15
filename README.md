
![lien vers le site de données data.gouv](https://raw.githubusercontent.com/rastakoer/certif_app_immo/datagouv_to_rds/datagouv_to_rds/Capture_pour_readme.PNG)

## Descriptifs :
#### Cette branche permet de récupérer les données depuis https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/ et depuis l'api https://geo.api.gouv.fr/ . Les données sont ensuite sélectionnées puis traitées avant d'être insérer dans une base de données hébergée sur AWS RDS.

## Liste des fichiers :
- EDA.ipynb : Notebook qui a permis de comprendre comment les données étaient agencées dans les fichiers récupérés sur data.gouv.
- datgouv_to_rds/loading_first_data_in_rds : Notebook permettant d'insérer les données dans une base de données Mysql hébergée sur AWS RDS.
- data_processing.py : Fichier permettant de sélectionner les données et de les formater avant l'insertion en base de données.
- connection.py : Permet de créer une connexion entre votre machine et AWS.
- create_tables.sql : Script permettant la création des tables pour la base de données relationnelle.
- requirements.txt : Permet d'installer les dépendances dans votre environnement

## Schéma de la base de données
![lien vers le site de données data.gouv](https://raw.githubusercontent.com/rastakoer/certif_app_immo/datagouv_to_rds/datagouv_to_rds/mld.PNG)

## Méthodologie 
- Créer une base de données MySQL sur AWS RDS 
- Créer un environnement sur cette branche
- Installer les dépendances à l'aide du requirements.txt
- Créer un fichier avec vos variables d'environnement dans le fichier .env/local
- Changer le chemin vers le fichier contenant vos variables d'environnement dans le fichier connection.py
- Executer les cellules du notebook loading_first_datas_to_rds

## Configuration requise
#### La récupération des données se faisant à l'aide de la mémoire RAM de votre machine et non par la sauvegarde des CSV en local, vous devez disposer d'un minimum de 16Go de RAM.
