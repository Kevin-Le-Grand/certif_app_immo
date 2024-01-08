from fastapi import FastAPI
import uvicorn
from functions import *


# Config apparence API
descritpion ="""
    Obtenez une prédiction de la valeur d'un bien grâce aux éléments suivants :
    - La région dans lequel ce situe le bien
    - La surface
    - Le nombre de pièces 
    - En choisissant le type de bien (Maison ou appartement)
    """
app = FastAPI(
    title="API prédictions d'un bien immobilier",
    summary="Api développée par Kevin LE GRAND",
    description=descritpion
)


# Définir une route POST pour la commande
@app.post("/predict", response_model=reponse_model, summary="Prédictions")
def predict(n:Config_donnees):
    """
    Route permettant de réaliser une prédiction
    """
    # Appel de la fonction servant à labelliser puis standardiser
    transform = encod_scal(n)
    
    # Appel de la fonction servant à réaliser les prédictions
    prediction= predictions(transform)
    # Réponse au format json
    return prediction


if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=4000)
