import streamlit as st
from functions import sqlengine
from sqlalchemy import text
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

engine = sqlengine()

def accueil():
    st.subheader("Cette application web permet d'estimer le montant d'un bien immobilier en France métropolitaine.")
    # Affichage de la période d'entraînement du modèle
    query=f"""SELECT MAX(DATE_MUTATION) - INTERVAL 15 MONTH AS DATE_MUTATION FROM VENTES;"""
    df = pd.read_sql(con=engine.connect(), sql=text(query))
    date_plus_recente = str(df['DATE_MUTATION'].min())
    st.write(f"Le modèle a été entraîné sur une période de 12 à partir du {date_plus_recente}")
    st.write("Il est possible que votre commune ne figure pas dans la liste pour les raisons suivantes :")
    st.write("  - N'ont été retenues que les communes ayant eu plus de 10 ventes sur la période d'entraînement")
    st.write("  - N'ont été retenues que les communes ayant moins de 30% de valeurs nettement supérieures à la moyenne nationale")
        

def stat_region(region : str) ->None:
    """ 
    Fonction permettant l'affichage des statistiques sur la région

    Args:
    - region (str) : Nom de la région
    """
    st.title("Page sur les statistiques en cours de construction...")
    st.subheader(f"Statistiques sur la région {region} :")
    for i in ["Maison","Appartement"]:
        query=f"""SELECT AVG(MONTANT)/AVG(SURFACE_BATI) m2 FROM VENTES V
                    INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
                    INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
                    INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
                    INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
                    WHERE NAME_TYPE_BIEN='{i}' 
                        AND Name_region='{region}';"""
        df = pd.read_sql(con=engine.connect(), sql=text(query))
        if i =="Maison" :
            print(f"Le prix moyen au m² pour une maison est de : {int(df.iloc[0,0])} €")
        else :
            print(f"Le prix moyen au m² pour un appartement est de : {int(df.iloc[0,0])} €")

def departement():
    pass

def commune():
    pass

