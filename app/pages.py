import streamlit as st
from functions import sqlengine
from sqlalchemy import text
import pandas as pd

engine = sqlengine()

def accueil():
    st.subheader("Cette application web permet d'estimer le montant d'un bien immobilier en France métropolitaine.")
    # Affichage de la période d'entraînement du modèle
    query=f"""((SELECT MAX(DATE_MUTATION) FROM VENTES), INTERVAL 15 MONTH)"""
    df = pd.read_sql(con=engine.connect(), sql=text(query))
    date_plus_recente = str(df['DATE_MUTATION'].min()).split("")[0]
    st.write(f"Le modèle a été entraîné sur une période de 12 à partir du {date_plus_recente}")
    st.write("Il est possible que votre commune ne figure pas dans la liste pour les raisons suivantes :")
    st.write("  - N'ont été retenues que les communes ayant eu plus de 10 ventes sur la période d'entraînement")
    st.write("  - N'ont été retenues que les communes ayant moins de 30% de valeurs nettement supérieures à la moyenne nationale")
        

def region():
    pass

def departement():
    pass

def commune():
    pass

