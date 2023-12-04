import streamlit as st
import pymysql
import os
from dotenv import load_dotenv

#//////////////////////////////////////////////////////////////////////////////
#                      Chargement des variable en local
#//////////////////////////////////////////////////////////////////////////////
load_dotenv(dotenv_path="/home/rastakoer/Bureau/certif_app_immo/app/.venv/.local")
#//////////////////////////////////////////////////////////////////////////////
#                          Database RDS AWS
#//////////////////////////////////////////////////////////////////////////////
db = pymysql.connect(host=os.environ['DB_HOST'],
user=os.environ['DB_USER'],
password=os.environ['DB_PASSWORD'],
port=int(os.environ['DB_PORT']))
cursor= db.cursor()
cursor.execute("USE datagouv;")

# Définir le titre et le logo de l'application
st.set_page_config(
    page_title="Immo App",
    page_icon=":rocket:",
    layout="wide"
)


# Affichage du logo
st.image("./Logo.png", width=200, caption="Votre Logo")

# Ajout d'un menu déroulant avec les régions:
query="""
SELECT Name_region FROM REGIONS;
"""
cursor.execute(query)
resultats = cursor.fetchall()
liste_regions=[row[0] for row in resultats]

option_par_defaut = "Sélectionnez une région"
region = st.selectbox("Sélectionnez une région", 
                              [option_par_defaut]+liste_regions)

if region!="Sélectionnez une région":
    # Ajout d'un menu déroulant avec les départements:
    query=f"""
    SELECT * FROM DEPARTEMENTS AS A
    JOIN (
    SELECT ID_REGION, Name_region
    FROM REGIONS
    )AS B
    ON A.ID_REGION=B.ID_REGION
    WHERE B.Name_region='{region}';
    """
    cursor.execute(query)
    resultats = cursor.fetchall()
    liste_departements=[row[1] for row in resultats]
    option_par_defaut="Sélecionnez un département"
    departement = st.selectbox("Sélecionnez un département", 
                              [option_par_defaut]+liste_departements)
    

    if departement!="Sélecionnez un département":
        # Ajout d'un menu déroulant avec les communes
        query=f"""
        SELECT * FROM COMMUNES AS A
        JOIN (
        SELECT ID_DEPT, Name_departement
        FROM DEPARTEMENTS
        )AS B
        ON A.ID_DEPT=B.ID_DEPT
        WHERE B.Name_departement='{departement}';
        """
        cursor.execute(query)
        resultats = cursor.fetchall()
        liste_communes=[row[1] for row in resultats]
        option_par_defaut="Sélecionnez une commune"
        commune = st.selectbox("Sélecionnez une commune", 
                              [option_par_defaut]+liste_communes)
        if commune!="Sélecionnez une commune":
            type_de_bien = st.selectbox("Sélectionnez le type de logement :", 
                                    ["Faites votre choix"]+["Maison", "Appartement"])
            if type_de_bien!="Faites votre choix":
                nb_pieces=st.selectbox("De combien de pièces est composé le bien", 
                                [0]+[i for i in range(0,15)])
                if nb_pieces!=0:
                    surface_bati = st.text_input("Surface habitable", "")
                    if surface_bati:
                        if type_de_bien=="Maison":
                            surface_terrain=st.text_input("Surface du terrain", "")
                            if surface_terrain:
                                valid_formulaire = st.button("Cliquez ici")
                            else:
                                st.write("Veuillez entrer la surface de votre terrain")
                        else:
                            surface_terrain=0
                            valid_formulaire = st.button("Cliquez ici")
                    else:
                        st.write ("Veuillez indiquer une surface habitable")
                else:
                    st.write("Sélectionnez le nombre de pièces")
            else:
                st.write("Sélectionner un type de bien")
        else:
            st.write("Sélecionnez une commune")
    else:
        st.write("Veuillez choisir un département")
else:
    st.write("Veuillez choisir une région")

if valid_formulaire:
    st.title("Le formulaire fonctionne plus qu'a réaliser le modèle et créer une api !!!")
    cursor.close()
    db.close()
else:
    pass
