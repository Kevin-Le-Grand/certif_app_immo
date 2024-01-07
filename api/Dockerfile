# # Utilisez l'image continuumio/miniconda3 comme image de base
FROM continuumio/miniconda3

# # Définissez le répertoire de travail
WORKDIR /home/app

# Install git
RUN apt-get update && apt-get install -y git nano

# # Recuperation du repo
RUN git clone  https://github.com/rastakoer/certif_app_immo.git --branch api .
 
# # Copiez les fichiers nécessaires
COPY requirements.txt .

# # Copiez les fichiers nécessaires
COPY model ./api/

# # Installez les dépendances
RUN pip install -r requirements.txt

# # Exécutez votre application
CMD uvicorn api.app:app --host=0.0.0.0 --port=$PORT