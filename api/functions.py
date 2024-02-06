import joblib
import io
import numpy as np
from pydantic import BaseModel
import os
import mlflow
import boto3
import os

#---------------------------------------------------------------------
#           Configuration S3
#---------------------------------------------------------------------
# Remplacez ces valeurs par les vôtres
# aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
# aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
# bucket = os.environ.get("BUCKET_NAME")

# # Initialiser du client S3
# s3 = boto3.client('s3',
#                   region_name="eu-west-3",
#                   aws_access_key_id=aws_access_key_id,
#                   aws_secret_access_key=aws_secret_access_key)

#---------------------------------------------------------------
#       Récupération d'objet depuis S3
#---------------------------------------------------------------
# Télécharger l'objet depuis S3 dans un flux BytesIO
# def load_joblib_from_s3(bucket_name, key):
#     response = s3.get_object(Bucket=bucket_name, Key=key)
#     joblib_content = response['Body'].read()

#     # Charger l'objet depuis le flux BytesIO
#     loaded_object = joblib.load(io.BytesIO(joblib_content))
#     return loaded_object

#---------------------------------------------------------------
#       Récupération des scalers 
#---------------------------------------------------------------
# Récupération des label_encoders et scalers depuis s3
# scalers = load_joblib_from_s3(bucket,"app_immo/joblib/scalers") 
# encoders = load_joblib_from_s3(bucket,"app_immo/joblib/encoders") 

# Récupération en local
# scalers = joblib.load("api/scalers")
# encoders = joblib.load("api/encoders")

#---------------------------------------------------------------
#           Récupération du modèle 
#---------------------------------------------------------------
# Récupération du modèle en local (en attendant de mettre en place MLFlow)
# model = joblib.load("api/model")

# Récupération du modèle sur MlFlow

mlflow.set_tracking_uri('https://mlflowimmoappkevleg-737621d410d0.herokuapp.com/') 
logged_model = 'runs:/eb04a2a765c04cb6bb2af1582b78e532/ImmoApp'
model = mlflow.sklearn.load_model(logged_model)

encoders = joblib.load(mlflow.artifacts.download_artifacts("mlflow-artifacts:/2/eb04a2a765c04cb6bb2af1582b78e532/artifacts/encoders.joblib"))
scalers = joblib.load(mlflow.artifacts.download_artifacts("mlflow-artifacts:/2/eb04a2a765c04cb6bb2af1582b78e532/artifacts/scalers.joblib"))


class Config_donnees(BaseModel):
    SURFACE_BATI : int
    NB_PIECES : int
    NAME_TYPE_BIEN: str
    Name_region: str

class reponse_model(BaseModel):
    reponse: float

def encod_scal(n:dict) ->list:
    """
    Fonction servant à labelliser et standardiser les données

    Args:
    - n (json) : Json contenant les données à prédire
    
    Returns ::
    - transformed_data (list) : Données standardisées de type [0.12,0.55,0.56,0.2]
    """
    transformed_data=[]
    transformed_data.append(scalers['SURFACE_BATI'].transform(np.array([n.SURFACE_BATI]).reshape(-1, 1))[0][0])
    transformed_data.append(scalers['NB_PIECES'].transform(np.array([n.NB_PIECES]).reshape(-1, 1))[0][0])
    type_bien = encoders['NAME_TYPE_BIEN'].transform([n.NAME_TYPE_BIEN])[0] 
    transformed_data.append(scalers['NAME_TYPE_BIEN'].transform(np.array([type_bien]).reshape(-1, 1))[0][0])
    region = encoders['Name_region'].transform([n.Name_region])[0] 
    transformed_data.append(scalers['Name_region'].transform(np.array([region]).reshape(-1,1))[0][0])
    return transformed_data

def predictions(data:list) -> dict:
    """
    Fonction permettant la prédiction 
    Sortie de type : {'reponse': 250000]}
    """
    # Prédiction en utilisant le modèle
    data = np.array([data])
    pred = model.predict(data)
    # Renvoi du dictionnaire contenant la prédiction
    return {'reponse':pred[0]}