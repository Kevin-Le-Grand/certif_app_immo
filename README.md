# Vidéo de présentation
<iframe width="560" height="315" src="https://www.youtube.com/embed/gA9sctlv6GY?si=UvYs2yMet0q5ChQv" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

# Schéma d'architecture de l'application
![Schéma de l'architecture de l'application](https://raw.githubusercontent.com/rastakoer/certif_app_immo/main/schema_application.PNG)

# Description
### Ce projet a été réalisé pour le passage du titre RNCP 34757 Dévellopeur en intelligence artificielle. Le projet a été séparé en quatres branches :
- datagouv_to_rds : Permettant la récupération de données depuis le site data.gouv par webscraping et api puis de stocker les données nécessaires dans une base de données hébergée sur AWS RDS.
- model : Dans cette branche une analyse exploratoire de données a été effectué. Cette branche permet également le déploiement de MLflow sur Heroku ainsi que de créer différents types de modèles.
- api : Pour la création d'une API avec FastAPI et déployé sur Heroku en CI/CD
- application : Branche pour la création de l'application réalisé avec le framework streamlit de python.

## Informations : 
- Si vous voulez reproduire cette application chez vous, clonez le projet complet depuis le main. Ensuite suivez les étapes en se référant aux README.md dans l'ordre suivant des branches :
    - datagouv_to_rds
    - model
    - api
    - app
- La construction de cette application nécessite la création de comptes sur AWS et Heroku et donc engendre des coûts.
