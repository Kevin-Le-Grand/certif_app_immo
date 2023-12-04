import pandas as pd

def select_datas(df : pd.DataFrame) -> pd.DataFrame:
    """
    Fonction permettant de sélectionner les données parmis celle disponibles.  

    Récupére les lignes concernant une vente et dont le local est une maison ou un appartement

    Les lignes récupérées sont :
        - date_mutation
        - valeur_fonciere
        - adresse_numero
        - adresse_nom_voie
        - code_postal
        - code_commune
        - nom_commune
        - code_departement
        - type_local
        - nombre_pieces_principales
        - surface_reelle_bati
        - surface_terrain
        - longitude
        - latitude
    """
    df = df.loc[(df.nature_mutation=="Vente") & ((df.type_local=="Maison") | (df.type_local=="Appartement")),
                                          ["date_mutation","valeur_fonciere",
                                           "adresse_numero","adresse_nom_voie",
                                           "code_postal","code_commune"
                                           "type_local","nombre_pieces_principales",
                                           "surface_reelle_bati","surface_terrain",
                                           "longitude", "latitude"]]
    return df

def nan_management(df : pd.DataFrame) -> pd.DataFrame:
    """
    Fonction permettant de remplacer les valeurs non renseignées par une valeur
    ou de supprimer la lignes 

    Traitements :
        - date_mutation : suppression de lignes
        - valeur_fonciere : suppression de lignes
        - adresse_numero : remplacement par 0 
        - adresse_nom_voie : suppression de lignes
        - code_postal : Suppression de lignes
        - code_commune : suppression de lignes
        - nombre_pieces_principales : suppression de lignes
        - surface_reelle_bati : suppression de lignes
        - surface_terrain : remplacement par 0
        - longitude : suppression de lignes
        - latitude : suppression de lignes
    """
    df = df.dropna(subset=['date_mutation'])
    df = df.dropna(subset=['valeur_fonciere'])
    df['adresse_numero'] = df['adresse_numero'].fillna(0)
    df = df.dropna(subset=['adresse_nom_voie'])
    df = df.dropna(subset=['code_postal'])
    df = df.dropna(subset=['code_commune'])
    df = df.dropna(subset=['nombre_pieces_principales'])
    df = df.dropna(subset=['surface_reelle_bati'])
    df['surface_terrain'] = df['surface_terrain'].fillna(0) 
    df = df.dropna(subset=['longitude'])
    df = df.dropna(subset=['latitude'])

    return df


def format_data(insert_data : pd.DataFrame) -> pd.DataFrame :
    """
    Fonction permettant d'adapter les données du dataframe au type de données de 
    la bases de données RDS :

        - date_mutation : pd.to_datetime
        - valeur_fonciere : int
        - adresse_numero : str
        - adresse_nom_voie : str
        - code_postal : int
        - code_commune : str
        - type_local : str
        - nombre_pieces_principales : int
        - surface_reelle_bati : int
        - surface_terrain : int
        - longitude : float
        - latitude : float
    """
    insert_data.loc[:, 'date_mutation'] = pd.to_datetime(insert_data['date_mutation'], errors='coerce')
    insert_data.loc[:, 'valeur_fonciere'] = insert_data.valeur_fonciere.astype(int)
    insert_data.loc[:, 'adresse_numero'] = insert_data.adresse_numero.astype(int)
    insert_data.loc[:, 'adresse_numero'] = insert_data.adresse_numero.astype(str)
    insert_data.loc[:, 'adresse_nom_voie'] = insert_data.adresse_nom_voie.astype(str)
    insert_data.loc[:, 'code_postal'] = insert_data.code_postal.astype(int)
    insert_data.loc[:, 'code_commune'] = insert_data.code_commune.astype(str)
    insert_data.loc[:, 'type_local'] = insert_data.type_local.astype(str)
    insert_data.loc[:, 'nombre_pieces_principales'] = insert_data.nombre_pieces_principales.astype(int)
    insert_data.loc[:, 'surface_reelle_bati'] = insert_data.surface_reelle_bati.astype(int)
    insert_data.loc[:, 'surface_terrain'] = insert_data.surface_terrain.astype(int)
    insert_data.loc[:, 'longitude'] = insert_data.longitude.astype(float)
    insert_data.loc[:, 'latitude'] = insert_data.longitude.astype(float)
    return insert_data