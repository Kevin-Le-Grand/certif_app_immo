## Aperçu
![Carte des température des prix](https://raw.githubusercontent.com/rastakoer/certif_app_immo/model/model/map_france_folium.PNG)
![Capture d'écran de l'api](https://raw.githubusercontent.com/rastakoer/certif_app_immo/model/model/Capture_mlflow.PNG)

## Descriptif :
- Dans cette branche un analyse exploratoire des données a été effectuée, avec les données récupérées de la base de données relationnelle crée avec la branche datagouv_to_rds, afin de déterminer les tendances et d'orienter le choix de variables pour la création des modèles.
- Plusieurs technologie ont été réalisée pour la création de modèles, XGBoost, RandomForestRegressor de scikit-learn et un réseau de neurones avec Tensorflow.
- Le monitoring des modèles est effectués avec MLflow.
- MLflow est déployé sur Heroku et les différentes composantes des modèles créés sont stockés bucket AWS S3, une base de données Postgre hébergé sur AWS RDS permet d'accéder à tous ces éléments.

## Liste des fichiers :
- model/EDA.ipynb : Analyse exploratoire permettant de faire des premières conclusions et d'orienter le choix des variables et méthodes à utiliser pour la création de modèles.
- model/connection.py : Fichier permettant d'établir une connexion avec AWS.
- model/data_processing.py : Fichier permettant de sélectionner les données et de les formater avant l'insertion en base de données. Utilisé lors de nouvelles mise à jour disponibles sur le site data.gouv.
- model/functions2.py : fichier permettant
    - La construction de requêtes
    - Le charment des données
    - Le splittage des données
    - Le filtrage des données (%minimum de ventes par commune, nombre d'outliers par commune)
    - La normalisation des données
    - La construction des modèles
    - Le tracé de courbes de validation et d'apprentissage
    - Le log des éléments sur MLflow
- 3 notebooks pour les différents entraînements de modèles 
- model/requirements.txt : Fichier pour importer les dépendances.
- mise_a_jour_donnees.ipynb : Fichier à executer lors de mise à jour disponible sur le site data.gouv.

## Méthodologie 
#### Si vous avez cloné le projet, vous devez :
- Commencer par réaliser les méthodologie de la branche datagouv_to_rds
- Créer un environnement sur la branche puis installer les dépendances à l'aide du requirements.txt
- Configurer les variables d'environnement sur Github et Heroku.
- Changer l'adresse du RUN pour cloner le repository dans le dockerfile.
- Changer l'adresse de uri_tracking de mlflow dans les notebook de création de modèle
- Créez vos propre modèles
- Exécutez les cellules du notebook mise_à_jour_donnees.ipynb
- Ré-entraînez un modèle avec le modèle le plus performant et ses meilleurs hyperparamètres
