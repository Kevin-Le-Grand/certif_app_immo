"""
Nom du Programme: maj.py

Auteur: LE GRAND Kevin

Description:
Cette fonction permet d'ajouter les dernières données disponible sur 
le site https://files.data.gouv.fr/geo-dvf/latest/

Le programme va réaliser les mêmes opérations que sur le notebook "loading_data_in_rds"
- utilisation des fonctions du fichier datas_processing.py
- Insertion des nouvelles données si elles ne sont pas déjà présente dans RDS
- Récupération de la nouvelle date de mise à jour et stockage dans s3

Date de création: 04/12/2023

Version: 1.0
"""

# Librairies pour la décompression de fichiers
import gzip
from io import BytesIO

# Librairies pour le scrapping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Visualisation de données
import pandas as pd

# Connection à RDS
from connection import db,cursor,s3,bucket_name

# datas processing
import datas_processing


#//////////////////////////////////////////////////////////////////////////////////
#               Récupération des datas sur le site data.gouv
#//////////////////////////////////////////////////////////////////////////////////

# URL de la page web
url = 'https://files.data.gouv.fr/geo-dvf/latest/csv/'

response = requests.get(url)
html_content = response.content

# Utilisez BeautifulSoup pour analyser le HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Trouver toutes les balises <a>
a_tags = soup.find_all('a')
annees=[]
# Parcourir chaque balise <a> et extraire le texte ainsi que la date/heure
for a_tag in a_tags:
    name = a_tag.text.strip()
    if name!="../":
        annees.append(name)
print(annees)

url = f'https://files.data.gouv.fr/geo-dvf/latest/csv/{annees[-1]}'
response = requests.get(url)
html_content = response.content
soup = BeautifulSoup(html_content, 'html.parser')

# Trouver la balise 'a' qui contient les liens de téléchargement
csv_element = soup.find('pre').find('a', {'href': "full.csv.gz"})
csv_link = urljoin(url, csv_element['href'])

# Téléchargement des datas et convertion en dataframe
csv_response = requests.get(csv_link)
if csv_response.status_code == 200:
    # Utilisation du buffer pour décompresser le fichier .gz
    try :
        with BytesIO(csv_response.content) as file_buffer:
            with gzip.GzipFile(fileobj=file_buffer, mode='rb') as gz_file:
                df = pd.read_csv(gz_file, low_memory=False)
    except Exception as e:
        print("Erreur dans la transformation des données en dataframe :",e)

else :
    print("Problème dans le lien de récupération de données !!!")

#//////////////////////////////////////////////////////////////////////////////////
#                           Nettoyage des données
#//////////////////////////////////////////////////////////////////////////////////

# Selection des données
datas_processing.select_datas(df)
    
# Gestion des valeurs manquantes :
datas_processing.nan_management(df)

# Adaptation des données du dataframe au format de la base de données
datas_processing.format_data(df)

#Suppression des lignes dupliquées
df = df.drop_duplicates()


#//////////////////////////////////////////////////////////////////////////////////
#                           Insertion en base de données
#//////////////////////////////////////////////////////////////////////////////////

# Chargement en base de données
try :
    for index,row in df.iterrows():
        # Utilisation de la base de données datagouv
        query="""
        USE datagouv;
        """
        cursor.execute(query)

        query="""
        INSERT IGNORE INTO VENTES (DATE_MUTATION, MONTANT,
                        NUMERO_RUE, RUE , CODE_POSTAL,ID_COMMUNE,
                        ID_TYPE_BIEN, NB_PIECES ,
                        SURFACE_BATI,  SURFACE_TERRAIN,
                        LONGITUDE, LATITUDE )
        VALUES (%s,%s,
                %s,%s,%s,%s,
                (SELECT ID_TYPE_BIEN FROM TYPES_BIENS WHERE NAME_TYPE_BIEN=%s),%s,
                %s,%s,
                %s,%s)
        """
        cursor.execute(query,(row["date_mutation"],row['valeur_fonciere'],
                    row["adresse_numero"],row["adresse_nom_voie"],row["code_postal"],row["code_commune"],
                    row["type_local"],row["nombre_pieces_principales"],
                    row["surface_reelle_bati"],row["surface_terrain"],
                    row["longitude"],row["latitude"]))
    db.commit()
    print(f" Les ventes de l'année {annees[-1]} ont bien été insérées dans RDS")
except Exception as e:
    print(f"Erreur dans le chargement des données de l'année {annees[-1]} dans RDS :",e)
finally:
    cursor.close()
    db.close()
