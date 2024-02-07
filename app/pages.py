import streamlit as st
from functions import sqlengine
from sqlalchemy import text
import pandas as pd

engine = sqlengine()

def accueil():
    st.subheader("Prix moyen en France")
    for i in ["Appartement","Maison"]:
        query=f"""SELECT AVG(MONTANT) AS AVGMontant
        FROM VENTES V
        INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
        INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
        INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
        INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
        WHERE NAME_TYPE_BIEN='{i}'
        AND R.Name_region NOT IN("Martinique","Guyane","La Réunion","Mayotte");
        """
        datas = pd.read_sql(con=engine.connect(), sql=text(query))
        col1, col2 = st.columns([3,1])
        with col1:
            st.text(i)
        with col2:
            st.text(f"{int(datas.iloc[0,0])} €")
        

def region():
    pass

def departement():
    pass

def commune():
    pass

