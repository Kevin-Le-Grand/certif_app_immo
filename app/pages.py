import streamlit as st
from functions import sqlengine
from sqlalchemy import text
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functions import api_predict, affichage_ventes_proximite

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
            st.write(f"Le prix moyen au m² pour une maison est de : {int(df.iloc[0,0])} €")
        else :
            st.write(f"Le prix moyen au m² pour un appartement est de : {int(df.iloc[0,0])} €")

def stat_departement(departement : str) -> None:
    """ 
    Fonction permettant l'affichage des statistiques sur le département

    Args:
    - departement (str) : Nom du département
    """
    st.title("Page sur les statistiques du département en cours...")
    st.subheader(f"Statistiques sur le département  {departement} :")
    for i in ["Maison","Appartement"]:
        query=f"""SELECT AVG(MONTANT)/AVG(SURFACE_BATI) m2 FROM VENTES V
                    INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
                    INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
                    INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
                    WHERE NAME_TYPE_BIEN='{i}' 
                        AND Name_departement='{departement}';"""
        df = pd.read_sql(con=engine.connect(), sql=text(query))
        if i =="Maison" :
            st.write(f"Le prix moyen au m² pour une maison est de : {int(df.iloc[0,0])} €")
        else :
            st.write(f"Le prix moyen au m² pour un appartement est de : {int(df.iloc[0,0])} €")

def stat_commune(commune : str ) ->None:
    """ 
    Fonction permettant l'affichage des statistiques sur une commune

    Args:
    - commune (str) : Nom de la commune
    """
    st.title("Page sur les statistiques de la commune en cours...")
    st.subheader(f"Statistiques sur la commune de {commune} :")
    for i in ["Maison","Appartement"]:
        query=f"""SELECT AVG(MONTANT)/AVG(SURFACE_BATI) m2 FROM VENTES V
                    INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
                    INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
                    WHERE NAME_TYPE_BIEN='{i}' 
                        AND NAME_COMMUNE='{commune}';"""
        df = pd.read_sql(con=engine.connect(), sql=text(query))
        if i =="Maison" :
            st.write(f"Le prix moyen au m² pour une maison est de : {int(df.iloc[0,0])} €")
        else :
            st.write(f"Le prix moyen au m² pour un appartement est de : {int(df.iloc[0,0])} €")

#//////////////////////////////////////////////////////////////////////////////
# Affichage des pages en fonctions de l'avancement du remplissage du formulaire
#//////////////////////////////////////////////////////////////////////////////
def formulaire_valide(cursor):
    # Recherche du prix au m2 de la commune
    query=f"""SELECT prix_m2 FROM COMMUNES C
                    INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
                    INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
                    WHERE NAME_COMMUNE='{st.session_state.commune}' 
                        AND Name_region='{st.session_state.region}'
                        AND NAME_COMMUNE='{st.session_state.commune}';"""
    df = pd.read_sql(con=engine.connect(), sql=text(query))
    st.session_state.prix_m2 = int(df.iloc[0,0])

    # Appel de l'API
    prediction = api_predict({'SURFACE_BATI' :st.session_state.surface_bati,
                            'SURFACE_TERRAIN' :st.session_state.surface_terrain,
                            'prix_moyen_commune_m2': st.session_state.prix_m2})
    
    st.session_state.pred=int(prediction['reponse'])
    # Affichage de la prédiction
    st.title(f"Le bien est estimé à {st.session_state.pred} €")

    if st.session_state.type_de_bien == "Maison":
        st.subheader(f"Statistiques sur la commune de {st.session_state.commune}  pour une maison sont :")
    else :
        st.subheader(f"Statistiques sur la commune de {st.session_state.commune}  pour un appartement sont :")

    query=f"""SELECT 
                    AVG(prix_m2) m2avg,
                    MIN(prix_m2) m2min,
                    MAX(prix_m2) m2max,
                    AVG(MONTANT) avgM,
                    MIN(MONTANT) minM,
                    MAX(MONTANT) maxM
                FROM VENTES V
                INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
                INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
                INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
                INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
                WHERE NAME_TYPE_BIEN='{st.session_state.type_de_bien}' 
                    AND Name_region='{st.session_state.region}'
                    AND Name_departement='{st.session_state.departement}'
                    AND NAME_COMMUNE='{st.session_state.commune}';"""
    df = pd.read_sql(con=engine.connect(), sql=text(query))
    #Affichage du dataframe
    df = df.rename(columns={'m2avg': 'Prix moyen du m²',
                            'm2min': 'Prix minimum du m²',
                            'm2max': 'Prix maximum du m²',
                            'avgM': 'Montant moyen',
                            'minM': 'Montant minimum',
                            'maxM': 'Montant maximum'})
    
    # Formatage des colonnes du dataframe
    df = df.astype(int)
    df = df.astype(str)

    # Remplacement des virgules par un espace et ajout du symbole "€"
    df = df.applymap(lambda x: x.replace(',', ' ').strip() + ' €')

    st.dataframe(df,hide_index=True)

    st.subheader(f"Voici les {st.session_state.type_de_bien}s vendus dans la commune de {st.session_state.commune}")

    # Affichage des ventes réalisées dans la commune
    st.components.v1.html(affichage_ventes_proximite([st.session_state.region,
                                                    st.session_state.departement,
                                                    st.session_state.commune,
                                                    st.session_state.type_de_bien],cursor), height=500)
