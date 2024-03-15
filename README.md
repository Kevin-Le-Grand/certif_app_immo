## Aperçu
![Capture d'écran de l'application](https://raw.githubusercontent.com/rastakoer/certif_app_immo/dec0e5d2b748cb70c2f1fd0321557384cb44f708/app/Capture_app.PNG)

## Liste des fichiers :
- app/app.py : Squelette de l'application.
- app/functions.py : fonctions utiles à l'application (connection, formulaire, création d'une carte interactive, logs pour Grafana...).
- app/pages.py : fonctions contenants l'affichage des pages en fonctions du parcours utilisateur.
- app/config_bdd.py : Module python pour l'interaction avec une base de données.
- app/bdd_grafana.ipynb : Notebook pour créer des tables, ajouter des colonnes ou lignes...
- app/test.py : Fichier permettant les tests unitaires.
- app/Dockerfile : Permet de créer l'image docker qui sera déployée sur Heroku.
- logo.png : Logo de l'application.
- Heroku.yml : Fichier pour indiquer la marche à suivre par Heroku lors d'un nouveau push sur la branche application.
- .github/workflow/test.yml : Fichier contenant les informations à suivre par Github lors d'un nouveau push sur la branche application.

## Descriptifs :
- Cette application permet de réaliser une estimation de biens immobilier.
- L'application utilise la base de données créée avec la branche datagouv.
- L'application utilise une api pour la effectuer une prédiction, réalisation sur la branche api.
- L'activité de l'application est stockée en base de données(prédictions,anomalies).
- L'application nécessite un compte sur Grafana.
- L'application utilise l'authentification utilisateur.
- Cette application peut être déployée sur Heroku si les branches datagouv, model et api ont été exécutées à conditions d'avoir crée un compte sur AWS et Heroku et d'avoir configuré ces dernières.


## Méthodologie 
#### Si vous avez cloné le projet, vous devez :
- Commencer par réaliser les méthodologie des branches datagouv puis mlflow puis api.
- Configurer les variables d'environnement sur Github et Heroku.
- Changer l'adresse du RUN pour cloner le repository dans le dockerfile.
- Changer l'adresse de l'api dans la fonction api_predict du fichier functions.py
- Créer des bases de données à l'aide du fichier bdd_grafana.ipynb afin de permettre le stockage des logs de l'activité ainsi que les prédictions effectuées.
- Créer un compte sur Grafana puis changer le href ligne 43 du fichier app.py