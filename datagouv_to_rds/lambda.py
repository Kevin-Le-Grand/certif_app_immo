"""
Nom du Programme: maj_data.py

Auteur: LE GRAND Kevin

Description:
Cette fonction Lambda permet de voir si une mise à jour a été effectuée sur 
le site https://files.data.gouv.fr/geo-dvf/latest/

Date de création: 04/12/2023

Version: 1.0

Remarques:
Le programme est réalisé pour avoir une execution périodique automatique afin de 
vérifier si une mise à jour des données est disponible.
Il utilise les services :
    - AWS Lambda et EventBridge pour l'exectution automatique périodique
    - AWS s3 pour récupérer la dernière date de mise à jour récupérée

Il utilise également Gmail pour l'envoie de messages d'erreurs lors de 
l'execution du programme ou pour avertir qu'une mise ç jour est disponible.

"""

import os
import boto3
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib


# Configuration email
smtp_server = 'smtp.gmail.com'
smtp_port = 465                      

email_user = os.environ['email_user']
name_application=os.environ['name_application']
email_password=os.environ['email_password']
recipient = os.environ['recipient']

# Créer un objet MIMEMultipart pour l'e-mail
msg = MIMEMultipart()
msg['From'] = email_user
msg['To'] = recipient
msg['Subject'] = 'Alerte App Immo'


def send_message(texte : str ,detail : str) -> None:
    """
    Fonction permettant d'envoyer le message d'erreur par email.

    Args :
        texte : Message personalisé
        error : Message du try except
    Retursn:
        None
    Raises:
        None
    """
    message = f"""Information sur la lambda de recherche de mise à jour.

    Message : {texte} 
    
    Erreur : {detail}"""
    
    msg.attach(MIMEText(message, 'plain'))

    server = None

    try :
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(email_user, email_password)
            server.sendmail(email_user, recipient, msg.as_string())
    except Exception as smtp_error:
        print(f"Erreur SMTP : {str(smtp_error)}")
    finally:
        if server is not None:
            server.quit()
    return


def lambda_handler(event, context):
    # Configuration S3
    s3 = boto3.client('s3')
    bucket_name = os.environ['S3_BUCKET']


    try:
        # Téléchargement du fichier depuis S3 vers un nouveau buffer
        buffer_read = BytesIO()
        s3.download_fileobj(bucket_name, 'app_immo/last_maj.txt', buffer_read)

        # Lecture du contenu du buffer
        buffer_read.seek(0)
        last_maj_saved = buffer_read.read().decode('utf-8')  # Decodez les bytes en UTF-8
    except Exception as e:
        send_message("Erreur lors de la recupération du fichier texte dans s3",e)
        return {
            'statusCode': 500,
            'body': f"Erreur lors de la récupération de la date depuis S3 : {str(e)}"
        }

    # Web scraping pour récupérer la nouvelle date
    try:
        # URL de la page web
        url = 'https://files.data.gouv.fr/geo-dvf/latest/csv/'

        response = requests.get(url)
        html_content = response.content

        # Utilisez BeautifulSoup pour analyser le HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        date_element = soup.find('pre').contents[-1].strip()
        date_maj = date_element.replace(" ","")
    except Exception as e:
        send_message("Erreur lors de la recupération de mise à jour sur le site data.gouv",e)
        return {
            'statusCode': 500,
            'body': f"Erreur lors du web scraping : {str(e)}"
        }
    

    if last_maj_saved!=date_maj:
        send_message("Lambda effectuée avec succés :", 
                     "Mise à jour disponible. Veuillez executer le fichier maj.py")
        return {
            'statusCode': 200,
            'body': 'La date récupérée est différente que celle dans le fichier S3.'
        }
    else:
        return {
            'statusCode': 200,
            'body': 'La date récupérée est identique à celle dans le fichier S3.'
        }
