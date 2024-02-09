import streamlit as st
import pandas as pd
import pymysql
import os
from datetime import date
# from dotenv import load_dotenv
from sqlalchemy import create_engine, Date, String, Text
from config_bdd import DataBaseV2


# Librairies pour afficher les ventes sur une carte
import folium
import geopandas as gpd
from folium.plugins import MarkerCluster

# Librairie pour l'api
import requests

# Librairie pour le cryptage
import hashlib 

# #//////////////////////////////////////////////////////////////////////////////
# #                      Chargement des variable en local
# #//////////////////////////////////////////////////////////////////////////////
# load_dotenv(dotenv_path="/home/kevin/workspace/certif_app_immo/app/.venv/.local")

#//////////////////////////////////////////////////////////////////////////////
#                          Database RDS AWS
#//////////////////////////////////////////////////////////////////////////////
# Connection avec pymysql à la base de  données datagouv
def create_connection(host,user,password,port,database):
    conn = pymysql.connect(host=host,
    user=user,
    password=password,
    port=port,
    database=database)
    return conn

# Connection avec sqlalchemy à la base de données datagouv
def sqlengine():
    engine = create_engine(f"mysql+pymysql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/datagouv")
    return engine

#//////////////////////////////////////////////////////////////////////////////
#                          CREATION DES TABLES POUR GRAFANA
#//////////////////////////////////////////////////////////////////////////////
database = DataBaseV2(
    db_type='postgresql',
    db_url=f"{os.environ['URL_POSTGRE']}"
)

database.create_table('kpis',date_pred=Date, 
                      type_de_bien=String, 
                      region=String,
                      departement=String,
                      commune=String)

database.create_table('crash',date_crash=Date, 
                      Infos = Text)


#//////////////////////////////////////////////////////////////////////////////
#                          Page d'authentification
#//////////////////////////////////////////////////////////////////////////////
# Création de la table si elle n'existe pas
def create_user_table(conn,cursor):
    """ 
    fonction permettant de créer la table users si elle n'a pas encore été créée
    """
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
    conn.commit()
    return

# Fonction pour ajouter un utilisateur à la base de données avec mot de passe crypté
def add_user(conn, username, password):
    with conn.cursor() as cursor:
        # Vérifier si l'utilisateur existe déjà
        cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            st.warning("Cet utilisateur existe déjà. Veuillez choisir un autre nom d'utilisateur.")
        else:
            # Ajouter l'utilisateur si celui-ci n'existe pas encore
            hashed_password = hashlib.md5(password.encode('utf-8')).hexdigest()  # Utilisation de MD5 pour le hachage
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            st.success("Utilisateur créé avec succès. Vous pouvez maintenant vous connecter.")



# Fonction pour vérifier les informations de connexion
def check_login(conn, username, password):
    with conn.cursor() as cursor:
        cursor.execute('SELECT password FROM users WHERE username = %s', (username,))
        result = cursor.fetchone()
        if result:
            # Vérifier si le mot de passe correspond en utilisant MD5
            hashed_password_input = hashlib.md5(password.encode('utf-8')).hexdigest()
            return hashed_password_input == result[0]
        else:
            return False

def authentification_page(conn):
    st.title("Login Immoapp")
    
    col1, col2 ,col3= st.columns(3)
    with col2:
        st.subheader("Connexion")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
    
        if st.button("Se connecter"):
            if check_login(conn, username, password):
                st.session_state.username = username
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

    with col2:
        st.subheader("Créer un nouvel utilisateur")
        new_username = st.text_input("Nouveau nom d'utilisateur")
        new_password = st.text_input("Nouveau mot de passe", type="password")

        if st.button("Créer utilisateur"):
            add_user(conn, new_username, new_password)


#//////////////////////////////////////////////////////////////////////////////
#                          Formulaire
#//////////////////////////////////////////////////////////////////////////////
def formulaire(cursor):
    """ 
    Fonction permettant l'affichage et le remplissage du formulaire dans la 
    sidebar de l'application.

    Args :
    - cursor : Permet la connection avec la base de données

    Remarque :
    La fonction va chercher les informations dans la base de données en partant 
    de la géolocalisation la plus large à la plus étroite (région, département, commune)
    puis le type de bien, le nombre de pièces...
    """
    # Ajout d'un menu déroulant avec les régions:
    query="""
    SELECT Name_region FROM REGIONS
    WHERE Name_region NOT IN ("Martinique","Guyane","La Réunion","Mayotte","Guadeloupe");
    """
    cursor.execute(query)
    resultats = cursor.fetchall()
    liste_regions=[row[0] for row in resultats]

    option_par_defaut = "Sélectionnez une région"
    st.session_state.region = st.sidebar.selectbox("Sélectionnez une région", 
                                [option_par_defaut]+liste_regions)

    if st.session_state.region!="Sélectionnez une région":
        # Ajout d'un menu déroulant avec les départements:
        query=f"""
        SELECT * FROM DEPARTEMENTS AS A
        JOIN (
        SELECT ID_REGION, Name_region
        FROM REGIONS
        )AS B
        ON A.ID_REGION=B.ID_REGION
        WHERE B.Name_region="{st.session_state.region}";
        """
        cursor.execute(query)
        resultats = cursor.fetchall()
        liste_departements=[row[1] for row in resultats]
        option_par_defaut="Sélectionnez un département"
        st.session_state.departement = st.sidebar.selectbox("Sélectionnez un département", 
                                [option_par_defaut]+liste_departements)
        

        if st.session_state.departement!="Sélectionnez un département":
            # Ajout d'un menu déroulant avec les communes
            query=f"""
            SELECT * FROM COMMUNES AS A
            JOIN (
            SELECT ID_DEPT, Name_departement
            FROM DEPARTEMENTS
            )AS B
            ON A.ID_DEPT=B.ID_DEPT
            WHERE B.Name_departement='{st.session_state.departement}';
            """
            cursor.execute(query)
            resultats = cursor.fetchall()
            liste_communes=[row[1] for row in resultats]
            option_par_defaut="Sélectionnez une commune"
            st.session_state.commune = st.sidebar.selectbox("Sélectionnez une commune", 
                                [option_par_defaut]+liste_communes)
            if st.session_state.commune!="Sélectionnez une commune":
                st.session_state.type_de_bien = st.sidebar.selectbox("Sélectionnez le type de logement :", 
                                        ["Faites votre choix"]+["Maison", "Appartement"])
                if st.session_state.type_de_bien!="Faites votre choix":
                    st.session_state.nb_pieces=st.sidebar.selectbox("De combien de pièces est composé le bien", 
                                    [0]+[i for i in range(0,15)])
                    if st.session_state.nb_pieces!=0:
                        st.session_state.surface_bati = st.sidebar.text_input("Surface habitable", "")
                        if st.session_state.surface_bati:
                            if st.session_state.type_de_bien=="Maison":
                                st.session_state.surface_terrain=st.sidebar.text_input("Surface du terrain", "")
                                if st.session_state.surface_terrain:
                                    st.session_state.valid_formulaire = st.sidebar.button("Valider")
                                else:
                                    st.write("Veuillez entrer la surface de votre terrain")
                            else:
                                st.session_state.surface_terrain=0
                                st.session_state.valid_formulaire = st.sidebar.button("Valider")
                        else:
                            st.write ("Veuillez indiquer une surface habitable")
                    else:
                        st.write("Sélectionnez le nombre de pièces")
                else:
                    st.write("Sélectionner un type de bien")
            else:
                st.write("Sélectionnez une commune")
        else:
            st.write("Veuillez choisir un département")
    else:
        st.write("Veuillez choisir une région")
    return

#//////////////////////////////////////////////////////////////////////////////
# Affichage des pages en fonctions de l'avancement du remplissage du formulaire
#//////////////////////////////////////////////////////////////////////////////
def formulaire_valide(cursor):
    # Appel de l'API
    prediction = api_predict({'SURFACE_BATI' :st.session_state.surface_bati,
                            'NB_PIECES' : st.session_state.nb_pieces,
                            'NAME_TYPE_BIEN':st.session_state.type_de_bien,
                            'Name_region' : st.session_state.region})
    # Affichage de la prédiction
    st.title(f"Le bien est estimé à {int(prediction['reponse'])} €")

    # Affichage des ventes réalisées dans la commune
    st.components.v1.html(affichage_ventes_proximite([st.session_state.region,
                                                    st.session_state.departement,
                                                    st.session_state.commune],cursor), height=500)

#//////////////////////////////////////////////////////////////////////////////
#                          Création de la carte interactive
#//////////////////////////////////////////////////////////////////////////////
def affichage_ventes_proximite(commune : list, cursor) -> str:
    """
    Fonction permettant d'afficher sur une carte les ventes effectuées sur une commune  
    à l'aide de Folium

    Args :
    - commune (list) : Liste de type ['Normandie','Calvados','Caen']
    - cursor: Connexion à la base de données

    Returns:
        str (html): <html>.....</html>
    """
    # Requête pour sélectionner les ventes sur la communes
    query=f"""
    SELECT 
        v.*,
        t.NAME_TYPE_BIEN,
        c.NAME_COMMUNE
    FROM VENTES v
    JOIN COMMUNES c ON v.ID_COMMUNE = c.ID_COMMUNE
    JOIN DEPARTEMENTS d ON c.ID_DEPT = d.ID_DEPT
    JOIN REGIONS r ON d.ID_REGION = r.ID_REGION
    JOIN TYPES_BIENS t ON t.ID_TYPE_BIEN=v.ID_TYPE_BIEN
    WHERE c.NAME_COMMUNE = %s
        AND d.Name_departement = %s
        AND r.Name_region = %s;
    """
    cursor.execute(query,(commune[2],commune[1],commune[0]))
    result = cursor.fetchall()
    # Transformation de la requête en data frame
    df = pd.DataFrame(result)
    
    df.columns= ["id","MONTANT","NUMERO_RUE","RUE","CODE_POSTAL","longitude","latitude",
                 "DATE_MUTATION","SURFACE_BATI","NOMBRE_PIECES","SURFACE_TERRAIN",
                 "ID_TYPE_BIEN","xxxxx","CODE_COMMUNE","NAME_TYPE_BIEN","NAME_COMMUNE"]
    
    # Création des popup folium
    df['info'] = df.apply(lambda row: f"""
    <b>Type de bien :</b><br>{row['NAME_TYPE_BIEN']} <br><br>
    <b>Date de vente : </b><br>{row['DATE_MUTATION']} <br><br>
    <b>Adresse : </b><br>{row['NUMERO_RUE']} {row['RUE']} <br>
    {row['CODE_POSTAL']} {row['NAME_COMMUNE']} <br><br>
    <b>Surface habitable :</b> {row['SURFACE_BATI']} <br>
    <b>Surface terrain :</b> {row['SURFACE_TERRAIN']} <br>
    <b>Nombre de pièces :</b> {row['NOMBRE_PIECES']} <br><br>
    <b>Montant : </b><br>{row['MONTANT']}
    """, axis=1)                

    # Centrage de la carte au milieu des coordonnées
    folium_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)

    # Création d'un cluster de marqueurs
    marker_cluster = MarkerCluster()

    # Ajout des marqueurs avec les information pour chaque point au cluster
    for _,row in df.iterrows():
        location = row['latitude'], row['longitude']
        marker = folium.Marker(location=location)
        folium.Popup(f'<div style="font-size: 10px; width: 150px;">{row["info"]}</div>').add_to(marker)
        marker_cluster.add_child(marker)
    
    # Ajout des cluster à la map
    marker_cluster.add_to(folium_map)

    # Convertir la carte Folium en HTML
    folium_html = folium_map.get_root().render()
    return  folium_html


#//////////////////////////////////////////////////////////////////////////////
#                          échange avec l'API
#//////////////////////////////////////////////////////////////////////////////
def api_predict(data: dict) -> dict:
    """
    Fonction permettant de faire l'appel à l'api et de recevoir une prédiction

    Args :
    - data (dict) : Dictionnaire avec les valeurs saisies par l'utilisateur

    Returns
     - 
    """
    response = requests.post('https://apiimmoappkevleg-7337fa262339.herokuapp.com/predict',
                              json=data)
    
    return response.json()
    


#//////////////////////////////////////////////////////////////////////////////
#                  enregistrement de la recherche dans grafana
#//////////////////////////////////////////////////////////////////////////////
def log_grafana() -> None:
    """
    Fonction permettant d'enregistrer les paramètres de la recherche dans Grafana.

    Cette fonction ne prend aucun argument en entrée et utilise st.session_state 
    pour stocker les données dans postgre.

    Cette fonction ne produit aucune sortie dans l'application.
    """
    database.add_row('kpis',
                     date_pred=date.today(),
                    type_de_bien=st.session_state.type_de_bien,
                    region=st.session_state.region,
                    departement=st.session_state.departement,
                    commune=st.session_state.commune)
    return

#//////////////////////////////////////////////////////////////////////////////
#                  enregistrement des erreurs dans grafana
#//////////////////////////////////////////////////////////////////////////////
def log_crash_grafana(texte : str) -> None:
    """
    Fonction permettant d'enregistrer les plantages de l'application dans Postgre.

    Args:
    - texte (str) : Texte de l'erreur avec try: ... except Exception as e

    Returns:
    - None

    Remarque:  
    
    Cette fonction a pour but de visualiser les erreurs dans Grafana puis d'envoyer 
    un mail à l'aide de grafana aux développeurs.
    """
    database.add_row('crash',
                     date_crash=date.today(),
                     Infos = texte)
    return