import pymysql
from dotenv import load_dotenv
import os

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine


#//////////////////////////////////////////////////////////////////////////////
#                      Chargement des variable en local
#//////////////////////////////////////////////////////////////////////////////
load_dotenv(dotenv_path="/home/kevin/workspace/certif_app_immo/model/.venv/.local")


os.environ['AWS_ACCESS_KEY_ID'] = os.environ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['AWS_SECRET_ACCESS_KEY']

#//////////////////////////////////////////////////////////////////////////////
#                          Database RDS AWS
#//////////////////////////////////////////////////////////////////////////////
def connection_with_sqlalchemy(name_db : str) -> Engine:
    """
    Établit une connexion avec une base de données MySQL en utilisant SQLAlchemy.

    Args :
    - name_db (str) : Le nom de la base de données à laquelle se connecter.

    Return :
    sqlalchemy.engine.base.Engine : Un objet Engine SQLAlchemy représentant la connexion à la base de données.
    """
    engine = create_engine(f"mysql+pymysql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{name_db}")
    return engine