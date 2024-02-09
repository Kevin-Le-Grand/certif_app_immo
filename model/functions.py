import pandas as pd
from connection import connection_with_sqlalchemy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from sqlalchemy import text


def loading_data(query : str) -> pd.DataFrame:
    """ 
    Fonction permettant de de récupérer les données dans la base de données à
    l'aide de SQLAlchemy
    
    Args:
    - query (str) : Requête permettant de récupérer les données

    Returns:
    - datas (pd.DataFrame) : Data frame avec les données récupérées sur RDS
    """
    print("Chargement des données en cours...")

    engine = connection_with_sqlalchemy("datagouv")
    print("Création engine sqlalchemy OK")

    # Utilisation de sqlalchemy pour transformer la requête en dataframe
    datas = pd.read_sql(con=engine.connect(), sql=text(query))
    # Fermeture de la connection
    engine.dispose()  
    print("Chargement des données ok")
    return datas
    


def encod_scal(X_train : pd.DataFrame, X_test : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame , dict, dict):
    """ 
    Fonction permettant de labelliser puis de standardiser X_train et X_test  

    Args :
    - X_train (pd.DataFrame) : Données d'entraînement
    - X_test (pd.Dataframe) : Données de test

    Return :
    - X_train (pd.DataFrame) : Les données d'entraînement labellisées et standardisées
    - X_test (pd.Dataframe) : Les données de test labellisées et standardisées
    - encoders (dict) : Dictionnaire stockant les encodeurs pour chaque variable catégorielle
    - scalers (dict) : Dictionnaire stockant les scalers pour chaque variable numérique
    """
    print("Normalisation des données en cours...")
    # Sélection des variables non numériques
    non_numerical = X_train.select_dtypes(exclude=['number']).columns.to_list()
    # Sélection des colonnes à traiter (toutes sauf la valeur à prédire)
    features = X_train.columns.tolist()

    # Dictionnaire où seront stockés les LabelEncoder et Scaler afin
    # de pouvoir inverser la labellisation et la standardisation
    encoders = {}
    scalers = {}

    # Encodage des variables catégorielles
    for col in non_numerical:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        encoders[col] = le
    # Normalisation des données
    for col in features:
        scaler = StandardScaler()
        X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
        scalers[col] = scaler

    # Utilisation des encoders et scaler pour transformer X_test
    for col, encoder in encoders.items():

        X_test[col] = encoder.transform(X_test[col])
    
    # Normalisation des données
    for col, scaler in scalers.items():
        X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    print("Normalisation des données OK")
    return X_train, X_test, encoders,scalers

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
    - X_test (pd.Dataframe) : Données de test pour effectué les mesures de performances
    - y_test (pd.Series) : Prix de ventes
    - best_params : Les paramètres les plus performant du modèle

    Remarque : Cette fonction utilise d'autres fonctions pour la labellisation et
    la mise à l'échelle des données
    """
    print("Entraînement en cours...")
    # Suppression des lignes dupliquées
    df = df.drop_duplicates()

    # Tri du dataframe par ordre croissant de date
    df['DATE_MUTATION'] = pd.to_datetime(df['DATE_MUTATION'])
    df = df.sort_values(by='DATE_MUTATION', ascending=True)

    # Suppression de la colonne date
    df = df.drop("DATE_MUTATION", axis=1)

    # Reset de L'index
    df = df.reset_index(drop=True)

    # Split de données
    nb_lines_train = int(df.shape[0]*0.8)
    df_train = df.iloc[:nb_lines_train,:]
    df_test = df.iloc[nb_lines_train:,:]
    # Suppression des lignes dans X_test dont les ID_COMMUNE ne sont pas présents dans df_train
    df_test = df_test[df_test['ID_COMMUNE'].isin(df_train['ID_COMMUNE'])]
    # Séparation des données 
    y_train = df_train.loc[:,"MONTANT"]
    X_train = df_train.drop("MONTANT",axis=1)
    y_test = df_test.loc[:,"MONTANT"]
    X_test = df_test.drop("MONTANT", axis=1)

    # Labellisation et standardisation de X_train
    X_train, X_test, encoders, scalers = encod_scal(X_train,X_test)

    # Utilisation de GridSearch CV pour l'entraînement du modèle
    param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
    }   

    # Type de métriques pour la recherche de meilleurs paramètres
    scorer = make_scorer(r2_score)

    # Entraînement avec les différentes paramètres
    grid_search = GridSearchCV(RandomForestRegressor(), 
                               param_grid, cv=5, 
                               scoring=scorer,
                               verbose=1)
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    # Entraînement des données avec les meilleurs paramètres
    model=RandomForestRegressor(**best_params)
    model.fit(X_train,y_train)
    print("Entraînement OK")
    return model, encoders, scalers, X_test, y_test,best_params