import pandas as pd
from connection import connection_with_sqlalchemy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def loading_data() -> pd.DataFrame:
    """ 
    Fonction permettant de de récupérer les données dans la base de données à
    l'aide de SQLAlchemy
    
     Aucun argument n'est nécessaire.

    Returns:
    - datas (pd.DataFrame) : Data frame avec les données récupérées sur RDS
    """
    engine = connection_with_sqlalchemy("datagouv")

    query="""
    SELECT 
        V.*,
        T.NAME_TYPE_BIEN,
        C.NAME_COMMUNE,
        D.Name_departement,
        R.Name_region
    FROM VENTES V
    INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
    INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
    INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
    INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
    WHERE MONTANT>10000;
    """
    datas = pd.read_sql(query, engine)
    # Fermeture de la connection
    engine.dispose()  
    return datas


def filtrer_outliers(groupe: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour filtrer les outliers dans un groupe de données.

    Args:
    - groupe (pd.DataFrame) : Groupe de données sur lequel appliquer le filtrage.

    Returns:
    - pd.DataFrame : DataFrame avec les outliers filtrés.
    """
    Q1 = groupe['MONTANT'].quantile(0.25)
    Q3 = groupe['MONTANT'].quantile(0.75)
    IQR = Q3 - Q1  # Range interquartile
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    return groupe[(groupe['MONTANT'] >= borne_inf) & (groupe['MONTANT'] <= borne_sup)]
                   


def encod_scal(df : pd.DataFrame) -> (pd.DataFrame , dict, dict):
    """ 
    Fonction permettant de labelliser puis de standardiser un Data frame  

    Args :
    - df (pd.DataFrame) : Les données à labelliser puis standardiser

    Return :
    - df (pd.DataFrame) : Les données labellisées et standardisées
    - encoders (dict) : Dictionnaire stockant les encodeurs pour chaque variable catégorielle
    - scalers (dict) : Dictionnaire stockant les scalers pour chaque variable numérique
    - non_numerical (list) : Liste des variables catégorielle
    - features (list) : Liste des colonnes à standardiser
    """
    # Sélection des variables non numériques
    non_numerical = df.select_dtypes(exclude=['number']).columns.to_list()
    # Sélection des colonnes à traiter (toutes sauf la valeur à prédire)
    features = df.drop('MONTANT', axis=1).columns

    # Dictionnaire où seront stockés les LabelEncoder et Scaler afin
    # de pouvoir inverser la labellisation et la standardisation
    encoders = {}
    scalers = {}

    # Encodage des variables catégorielles
    for col in non_numerical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    # Normalisation des données
    for col in features:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler
    return df,encoders,scalers, non_numerical, features

def reverse_scal_encod(df : pd.DataFrame, encoders : dict, scalers : dict,
                       non_numerical: list, features: list) -> pd.DataFrame:
    """ 
    Fonction permettant d'inverser la standardisation puis la labellisation  

    Args:
    - df (pd.DataFrame) : Data frame qui a été labellisé et standardisé  
    - encoders (dict) : Dictionnaire contenant les encodeurs pour chaque variable catégorielle
    - scalers (dict) : Dictionnaire contenant les scalers pour chaque variable numérique
    - non_numerical (list) : Liste des variables catégorielle
    - features (list) : Liste des colonnes à standardiser
    
    Returns:
    - df (pd.DataFrame) : Data frame avec les valeurs d'origine
    """
    # Inversion de la standardisation des données
    for col in features:
        scaler = scalers[col]
        df[col] = scaler.inverse_transform(df[col].values.reshape(-1, 1))

    # Inversion de l'encodage des variables catégorielles
    for col in non_numerical:
        le = encoders[col]
        df[col] = le.inverse_transform(df[col].astype(int))
    return df

def train_model(df: pd.DataFrame) ->(pd.DataFrame , dict, dict,list, list):
    """ 
    Fonction permettant d'entraîner un modèle à partir d'un data frame

    Args:
    - df (pd.DataFrame) : Données 

    Returns:
    - model : Modèle entraîné
    - encoders (dict) : Dictionnaire stockant les encodeurs pour chaque variable catégorielle
    - scalers (dict) : Dictionnaire stockant les scalers pour chaque variable numérique
    - non_numerical (list) : Liste des variables catégorielle
    - features (list) : Liste des colonnes à standardiser

    Remarque : Cette fonction utilise d'autres fonctions pour la labellisation et
    la mise à l'échelle des données
    """
    # Sélection des données non nulles
    df = df.loc[(df.NB_PIECES>0) & (df.SURFACE_BATI>0),:]
    # Suppression des valeurs extremes en région Bretagne
    df = df[~((df.Name_region=="Bretagne")&(df.MONTANT>6.5e6))]
    # Sélection des données
    df = df.loc[:,["SURFACE_BATI","NB_PIECES","NAME_TYPE_BIEN","Name_region","MONTANT"]]
    # Suppression des lignes dupliquées
    df = df.drop_duplicates()
    # Suppression des outliers
    df = df.groupby('Name_region').apply(filtrer_outliers)
    # Reset de L'index
    df = df.reset_index(drop=True)
    df, encoders, scalers, non_numerical, features= encod_scal(df)
    # Split de données
    X_train, _, y_train, _ = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.8,random_state=42)
    # Entraînement des données
    model=RandomForestRegressor()
    model.fit(X_train,y_train)
    return model, encoders, scalers, X_train, y_train