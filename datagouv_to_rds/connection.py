"""
Nom du Programme: connection.py
Auteur: LE GRAND Kevin

Description: Programme permettant les connections aux services d'aws et Gmail

version : 1

Remarques: Le fichier utilise des variables d'environnement pour des raison de sécurité
"""


import pymysql
import boto3
from sqlalchemy import create_engine

from email.mime.multipart import MIMEMultipart

import os
from dotenv import load_dotenv

#//////////////////////////////////////////////////////////////////////////////
#                      Chargement des variable en local
#//////////////////////////////////////////////////////////////////////////////
load_dotenv(dotenv_path="/home/kevin/workspace/certif_app_immo/datagouv_to_rds/.venv/.local")


#//////////////////////////////////////////////////////////////////////////////
#                               Mail
#//////////////////////////////////////////////////////////////////////////////

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


#//////////////////////////////////////////////////////////////////////////////
#                          Database RDS AWS
#//////////////////////////////////////////////////////////////////////////////
db = pymysql.connect(host=os.environ['DB_HOST'],
user=os.environ['DB_USER'],
password=os.environ['DB_PASSWORD'],
port=int(os.environ['DB_PORT']))
cursor= db.cursor()

engine = create_engine(f"mysql+pymysql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/datagouv")
#//////////////////////////////////////////////////////////////////////////////
#                          S3 AWS
#//////////////////////////////////////////////////////////////////////////////
bucket_name = os.environ['BUCKET_NAME']
os.environ['AWS_ACCESS_KEY_ID'] = os.environ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['AWS_SECRET_ACCESS_KEY']
s3 = boto3.client('s3')
