## Aperçu
![Capture d'écran de l'api](https://raw.githubusercontent.com/rastakoer/certif_app_immo/api/api/Capture_api.PNG)

## Descriptif :
- Cette API est réalisée à l'aide de FastAPI et hébergée sur Heroku.
- Elle permet, dans sa configuration actuelle, d'estimer le prix d'un bien immobilier en fonction de 3 critères, la surface du bâti, la surface du terrain ainsi que le prix moyen au m² de la commune où est situé le bien.
- Le model et le scaler utilisé pour réaliser la prédiction sont récupérer depuis le module MLflow.

## Liste des fichiers :
- api/app.py : Servant à afficher le descriptif de l'API et d'appeler les fonctions servants à la prédiction du prix du bien immobilier.
- api/functions.py : Fichier permettant de récupérer le model et le scaler nécessaire pour la prédiction sur MLflow. Il gère aussi le traitement des données à effectué avant d'effectuer la prédiction.
- api/requirements.txt : Contient les dépendances nécessaires.
- api/Dockerfile : Fichier pour la création de l'image lors du déploiement sur Heroku
- api/test.py : Fichier pour les tests unitaires
- heroku.yml : Fichier pour indiquer à Heroku ce qu'il doit faire lors que les tests unitaires sont réussis.

## Méthodologie 
#### Si vous avez cloné le projet, vous devez :
- Commencer par réaliser les méthodologie des branches datagouv puis mlflow.
- Configurer les variables d'environnement sur Github et Heroku.
- Changer l'adresse du RUN pour cloner le repository dans le dockerfile.
- Changer l'adresse du tracking de mlflow ligne 55 du fichier functions.py
- Changer les liens du modèle et scaler crées ligne 56 et 60 du fichier functions.py
