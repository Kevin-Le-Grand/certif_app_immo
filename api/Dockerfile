# # Utilisez l'image continuumio/miniconda3 comme image de base
FROM continuumio/miniconda3

# # Définissez le répertoire de travail
WORKDIR /home/app

# # Recuperation du repo
RUN git clone  https://github.com/rastakoer/deploy_irirs --branch api .
 
# # Copiez les fichiers nécessaires
COPY requirements.txt .

# # Installez les dépendances
RUN pip install -r requirements.txt

# # Exécutez votre application
CMD uvicorn app:app --host=0.0.0.0 --port=$PORT