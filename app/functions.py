import pandas as pd
import pymysql
import os
# from dotenv import load_dotenv

# Librairies pour afficher les ventes sur une carte
import folium
import geopandas as gpd
from folium.plugins import MarkerCluster

# #//////////////////////////////////////////////////////////////////////////////
# #                      Chargement des variable en local
# #//////////////////////////////////////////////////////////////////////////////
# load_dotenv(dotenv_path="/home/kevin/workspace/certif_app_immo/app/.venv/.local")

#//////////////////////////////////////////////////////////////////////////////
#                          Database RDS AWS
#//////////////////////////////////////////////////////////////////////////////
db = pymysql.connect(host=os.environ['DB_HOST'],
user=os.environ['DB_USER'],
password=os.environ['DB_PASSWORD'],
port=int(os.environ['DB_PORT']))
cursor= db.cursor()
cursor.execute("USE datagouv;")


#//////////////////////////////////////////////////////////////////////////////
#                          Création de la carte interactive
#//////////////////////////////////////////////////////////////////////////////
def affichage_ventes_proximite(commune : list) -> str:
    """
    Fonction permettant d'afficher sur une carte les ventes effectuées sur une commune  
    à l'aide de Folium

    Args :
        liste : ['Normandie','Calvados','Caen']
    Returns:
        str : <html>.....</html>
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
    WHERE c.NAME_COMMUNE = '{commune[2]}' 
        AND d.Name_departement = '{commune[1]}' 
        AND r.Name_region = '{commune[0]}';
    """
    cursor.execute(query)
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
    folium_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=4)

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