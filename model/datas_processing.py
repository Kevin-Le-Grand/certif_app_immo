import pandas as pd
import numpy as np

def select_datas(df : pd.DataFrame) -> pd.DataFrame:
    """
    Fonction permettant de sélectionner les données parmi celles disponibles.  

    Récupère les lignes concernant une vente et dont le local est uniquement une maison ou un appartement et qui peuvent posséder une dépendances.

    Les lignes récupérées sont :
        - id_mutation
        - date_mutation
        - valeur_fonciere
        - adresse_numero
        - adresse_nom_voie
        - code_postal
        - code_commune
        - type_local
        - nombre_pieces_principales
        - surface_reelle_bati
        - surface_terrain
        - longitude
        - latitude
    """
    df = df.loc[df.nature_mutation=="Vente",
                                        ["id_mutation",
                                            "date_mutation","valeur_fonciere",
                                           "adresse_numero","adresse_nom_voie",
                                           "code_postal","code_commune",
                                           "type_local","nombre_pieces_principales",
                                           "surface_reelle_bati","surface_terrain",
                                           "longitude", "latitude"]]
    
    # Récupération de tous les id_mutation comportant Appartement ou Maison
    valid_ids = df[df['type_local'].isin(['Appartement', 'Maison'])]['id_mutation'].unique()
    # Filtrage pour récupérer toutes les lignes pour les ventes qui concerne un appartement ou une maison
    df = df[df['id_mutation'].isin(valid_ids)]

    # Récupération de tous les id_mutation comportant 'Local industriel. commercial ou assimilé'
    not_valid_ids = df[df['type_local'].isin(['Local industriel. commercial ou assimilé'])]['id_mutation'].unique()
    # Filtrage pour récupérer les ventes d'un appartement, d'une maison  avec une dépendances ou non 
    # Mais sans local commercial.
    df = df[~df['id_mutation'].isin(not_valid_ids)]

    # Recherche d'id_mutation comportant une dépendance
    id_with_dependance= df[df['type_local'].isin(['Dépendance'])]['id_mutation'].unique()
    # Ajout d'une colonne dépendance égale à 1 si dépendance
    df['dependance'] = df['id_mutation'].isin(id_with_dependance).astype(int)
    # Remplacement de Dépendance dans type_local par Nan
    df['type_local'] = df['type_local'].replace('Dépendance', np.nan)

    # Récupération des id_mutation avec un type_local "Maison"
    liste_id_maison = set(df.loc[df.type_local == "Maison", 'id_mutation'].unique())
    # Récupération des id_mutation avec un type_local "Appartement"
    liste_id_appartement = set(df.loc[df.type_local == "Appartement", 'id_mutation'].unique())
    # Récupération des id_mutation en communs
    liste_id_communs = list(liste_id_maison.intersection(liste_id_appartement))
    # Retrait des id_mutation comportant une maison ET un appartement
    df = df[~df['id_mutation'].isin(liste_id_communs)]

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
        - nombre_pieces_principales : remplacement par 0
        - surface_reelle_bati : remplacement par 0
        - surface_terrain : remplacement par 0
        - longitude : suppression de lignes
        - latitude : suppression de lignes
    """

    # Récupération des id_mutation avec des Nan dans date_mutation
    liste_id_with_nan = df.loc[pd.isna(df['date_mutation']), 'id_mutation'].unique()
    # Suppression des lignes avec une valeur Nan dans date_mutation
    df = df[~df['id_mutation'].isin(liste_id_with_nan)]

    # Récupération des id_mutation avec des Nan dans valeur_foncière
    liste_id_with_nan = df.loc[pd.isna(df['valeur_fonciere']), 'id_mutation'].unique()
    # Suppression des lignes avec une valeur Nan dans valeur_foncière
    df = df[~df['id_mutation'].isin(liste_id_with_nan)]
    
    # Les Nan dans adresse_numero peuvent correspondre à un lieu dit donc sans numéro, 
    # la valeur des Nan sont remplacées par 0.0
    df['adresse_numero'] = df['adresse_numero'].fillna("0.0")

    # Remplacement des Nan par un texte vide. à noter qu'une même vente peut avoir 
    # deux adresse différente Lors du groupement sur l'id on prendra le texte le plus long
    df['adresse_nom_voie'] = df['adresse_nom_voie'].fillna("")
    
    # On remplace le code postal manquant par 0.0 et on prendra le max lors du groupement
    df['code_postal'] = df['code_postal'].fillna("0.0")

    # Récupération des id_mutation avec des Nan dans code_commune
    liste_id_with_nan = df.loc[pd.isna(df['code_commune']), 'id_mutation'].unique()
    df = df[~df['id_mutation'].isin(liste_id_with_nan)]

    # On va remplacer la valeur des Nan dans nombre_pieces_principales, surface_reelle_bati et surface_terrain par "O.O" 
    # Lors du groupement on prendra la valeur max de nombre_pieces_principales et surface_reelle_bati
    # Mais on fera la somme de surface_terrain qui correspond à la taille de plusieurs parcelles de terrain
    df['nombre_pieces_principales'] = df['nombre_pieces_principales'].fillna("0.0")
    df['surface_reelle_bati'] = df['surface_reelle_bati'].fillna("0.0")
    df['surface_terrain'] = df['surface_terrain'].fillna("0.0")

    # Visualisation des id avec des nan dans longitude et suppression des id_mutation
    liste_id_with_nan = df.loc[pd.isna(df['longitude']), 'id_mutation'].unique()
    df = df[~df['id_mutation'].isin(liste_id_with_nan)]

    # Visualisation des id avec des nan dans longitude et suppression des id_mutation
    liste_id_with_nan = df.loc[pd.isna(df['latitude']), 'id_mutation'].unique()
    df = df[~df['id_mutation'].isin(liste_id_with_nan)]

    return df


def format_data(df : pd.DataFrame) -> pd.DataFrame :
    """
    Fonction permettant d'adapter les données du data frame au type de données de 
    la bases de données RDS :
        - id_mutation : ne sera pas inclus dans la base de données
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
    df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce')

    df['valeur_fonciere'] = pd.to_numeric(df['valeur_fonciere'], errors='coerce')
    df['valeur_fonciere'] = df.valeur_fonciere.astype(int)

    df['adresse_numero'] = df.adresse_numero.astype(str)
    df['adresse_numero'] = df.adresse_numero.apply(lambda x: x.split('.')[0])

    df['adresse_nom_voie'] = df.adresse_nom_voie.astype(str)

    df['code_postal'] = df.code_postal.astype(str)
    df['code_postal'] = df.code_postal.apply(lambda x: x.split('.')[0].zfill(5)) 

    df['code_commune'] = df.code_commune.astype(str)

    df['type_local'] = df.type_local.astype(str)

    df['nombre_pieces_principales'] = pd.to_numeric(df['nombre_pieces_principales'], errors='coerce')
    df['nombre_pieces_principales'] = df.nombre_pieces_principales.astype(int)

    df['surface_reelle_bati'] = pd.to_numeric(df['surface_reelle_bati'], errors='coerce')
    df['surface_reelle_bati'] = df.surface_reelle_bati.astype(int)

    df['surface_terrain'] = pd.to_numeric(df['surface_terrain'], errors='coerce')
    df['surface_terrain'] = df.surface_terrain.astype(int)
    return df


def grouped_datas(df : pd.DataFrame)-> pd.DataFrame:
    """
    La vente d'un bien est constitué de plusieurs lots (parcelles)

    Cette fonction permet de grouper un bien sur une seule ligne.

    Voici les fonctions d'aggregations

        - 'date_mutation': 'max',   
        - 'valeur_fonciere': 'max',
        - 'adresse_numero' : longest_text,
        - 'adresse_nom_voie' : longest_text,
        - 'code_postal' : longest_text,
        - 'code_commune' : longest_text,
        - 'type_local' : longest_text,
        - 'nombre_pieces_principales' : 'max',
        - 'surface_reelle_bati' : 'max',
        - 'surface_terrain' : 'sum',
        - 'longitude' : 'mean',
        - 'latitude' : 'mean',
        - 'dependance' : 'max'
    """
    # Fonction personnalisée pour obtenir le texte le plus long
    def longest_text(series):
        """
        Retourne l'élément texte le plus long dans la Series pandas fournie.

        Paramètres :
        - series (pd.Series) : Series pandas d'entrée contenant des éléments texte.

        Retour :
        - str : L'élément texte le plus long dans la Series fournie, déterminé par la longueur du texte.
        """
        return max(series, key=len)

    aggregation_functions = { 'date_mutation': 'max',   
                            'valeur_fonciere': 'max',
                            'adresse_numero' : longest_text,
                            'adresse_nom_voie' : longest_text,
                            'code_postal' : longest_text,
                            'code_commune' : longest_text,
                            'type_local' : longest_text,
                            'nombre_pieces_principales' : 'max',
                            'surface_reelle_bati' : 'max',
                            'surface_terrain' : 'sum',
                            'longitude' : 'mean',
                            'latitude' : 'mean',
                            'dependance' : 'max'}  


    df = df.groupby('id_mutation').agg(aggregation_functions).reset_index()

    return df

def features_and_foreign_keys(df : pd.DataFrame , 
                              liste_ids : list ) -> pd.DataFrame :
    """
    Fonction permettant de supprimer des lignes dans les données à insérer et d'adapter
    les variables du data frame à celles de la base de données :
    
    - Suppression de la colonne id_mutation
    - Changement des valeurs de la colonne type_local pour s'adapter à la table TYPES_BIENS
    - Changement des nom des colonnes du data frame
    - Suppression des lignes comportant un id_commune non présent dans la table COMMUNE

    Args:
        - df : Données des ventes
        - liste_ids : liste des codes communes issus de l'api

    Return:
        - df : le data frame de sortie avec toutes les données pouvant être insérées

    Remarque :
    La fonction permet d'afficher le nombre de ventes supprimée sur l'année.
    """
    # Suppression de la colonne id_mutation qui est inutile maintenant
    df = df.iloc[:,1:]

    # Remplacement de la valeur de type local pour pouvoir être jointé
    # sur la table TYPES_BIENS
    df['type_local'] = df['type_local'].replace({'Maison': 2, 'Appartement': 1})
    df['type_local'] = df.type_local.astype(int)

    # Correlation entre les colonnes 
    column_mapping={'date_mutation' : 'DATE_MUTATION',
                    'valeur_fonciere' : 'MONTANT',
                    'adresse_numero' : 'NUMERO_RUE',
                    'adresse_nom_voie': 'RUE',
                    'code_postal' :  'CODE_POSTAL',
                    'code_commune' : 'ID_COMMUNE',
                    'type_local' : 'ID_TYPE_BIEN',
                    'nombre_pieces_principales' : 'NB_PIECES',
                    'surface_reelle_bati' : 'SURFACE_BATI',
                    'surface_terrain' : 'SURFACE_TERRAIN',
                    'longitude' : 'LONGITUDE',
                    'latitude': 'LATITUDE',
                    'dependance' : 'DEPENDANCES'}
    df.columns = [column_mapping[col] for col in df.columns]
    
    # Comparaison entre la table COMMUNES et VENTES
    liste_id_commune_a_verifier = df.ID_COMMUNE.to_list()
    elements_not_in_b = set(liste_id_commune_a_verifier) - set(liste_ids)
    print("Nombre de lignes ne pouvant être insérées :",
        df.loc[df['ID_COMMUNE'].isin(elements_not_in_b),:].shape[0])
    
    # Suppression des ventes ayant un code_commune non présent dans la table COMMUNES
    df = df[~df['ID_COMMUNE'].isin(elements_not_in_b)]

    return df