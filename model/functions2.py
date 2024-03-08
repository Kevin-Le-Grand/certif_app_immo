import os
import numpy as np
import seaborn as sns
# Ligne à décommenter dans colab :
# from drive.MyDrive.PCO.connection import connection_with_sqlalchemy
# Ligne à commenter dans Colab :
from connection import connection_with_sqlalchemy
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,mean_squared_log_error
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,Callback
from xgboost import XGBRegressor
from sklearn.model_selection import learning_curve, validation_curve
from sqlalchemy import text
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
from IPython.display import display

# L.41 : Construction de la requête de base 
# L.116 : Chargement des données
# L.145 : Split des données 
# L.202 : Option de filtrage des données
# L.301 : Labellisation et standardisation des données
# L.355 : RANDOM FOREST REGRESSOR
# L.405 : TENSORFLOW
# L.508 : XGBOOST
# L.560 : Graphiques pour visualiser les données d'entraînement
# L.676 : LOG MLFLOW

#//////////////////////////////////////////////////////////////////////////////
#                       Construction de la requête de base
#//////////////////////////////////////////////////////////////////////////////
def construcion_requete(region: str,
                        type_de_bien : str,
                        nb_ventes_mini : int,
                        surface_terrain : bool,
                        nb_mois : int) -> Any:
    """
    Fonction permettant de créer la structure de la requête en fonction des divers éléments passés.

    Args:
    - region (str) : Nom d'une région ou ''
    - type de bien (str) : '' , 'Appartement' ou 'Maison'
    - nb_ventes_mini (int) : Nombre de ventes minimum dans une commune
    - surface_terrain (bool) : Inclure la surface du terrain True ou False.
    - nb_mois (int) : Nombre de mois pour la récupération des données

    Returns:
    - where_clause (str) : Condition de filtrage sur les régions et le type de bien
    - query (str) : requête complète permettant de récupérer les données
    
    Remarque :
    where_clause sera utilisé par la suite dans la fonction de filtrage des outliers
    """
    # Les filtres de la requête permettent de filtrer les valeurs aberrantes vu dans l'EDA
    where_clause = "WHERE R.Name_region NOT IN('Martinique', 'Guyane', 'La Réunion', 'Mayotte', 'Guadeloupe') "
    if region!='' and type_de_bien=='':
        where_clause = f"WHERE R.Name_region = '{region}'"
    elif region!='' and  type_de_bien!='': 
        where_clause = f"WHERE R.Name_region = '{region}' AND T.NAME_TYPE_BIEN='{type_de_bien}'"
    elif region=='' and  type_de_bien!='': 
        where_clause+= f"AND T.NAME_TYPE_BIEN='{type_de_bien}'"


    if nb_mois is not None:
        where_clause +=f" AND V.DATE_MUTATION >= DATE_SUB((SELECT MAX(DATE_MUTATION) FROM VENTES), INTERVAL {nb_mois} MONTH)"
        
    query=f"""
    # Table pour compter le nombre de ventes par commune
    WITH nb_ventes_mini AS(
    SELECT
        ID_COMMUNE AS ID_COMMUNE,
        count(*) nb_ventes_par_commune
    FROM VENTES 
    WHERE DATE_MUTATION >= DATE_SUB((SELECT MAX(DATE_MUTATION) FROM VENTES), INTERVAL {nb_mois} MONTH)
    GROUP BY ID_COMMUNE
    )

    # Selection des variables voulues pour l'entraînement du modèle
    SELECT 
        V.SURFACE_BATI,
        V.ID_COMMUNE,
        V.DATE_MUTATION,
        # T.NAME_TYPE_BIEN,
        # R.Name_region,
        {'V.SURFACE_TERRAIN,' if surface_terrain==True else ''}
        V.MONTANT
    FROM VENTES V
    INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
    INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
    INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
    INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
    {where_clause}
    AND V.MONTANT>15000  AND V.MONTANT<6500000
    AND V.SURFACE_BATI>0
    AND V.NB_PIECES>0
    AND V.ID_COMMUNE IN (
        SELECT 
            ID_COMMUNE 
        FROM nb_ventes_mini 
        WHERE nb_ventes_par_commune>={nb_ventes_mini})
    """
    return where_clause , query

#//////////////////////////////////////////////////////////////////////////////
#                       Chargement des données
#//////////////////////////////////////////////////////////////////////////////
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


#//////////////////////////////////////////////////////////////////////////////
#                       Split des données 
#//////////////////////////////////////////////////////////////////////////////
def split_with_m2(df: pd.DataFrame):
    """ 
    Fonction permettant de séparer les données en données d'entraînement et de test,
    d'ajouter une colonne avec le prix moyen au m2 par commune et de supprimer 
    la colonne ID_COMMUNE rendue inutile

    Args :
    - df (pd.DataFrame) : Données à splitter

    Return :
    - X_train (pd.Dataframe) : Données d'entrée d'entraînement.
    - y_train (pd.Series) : Données de sortie d'entraînement.
    - X_test (pd.Dataframe) : Données d'entrée de test.
    - y_test (pd.Series) : Données de sortie de test.
    """
    print("Split des données en cours...")
    # Suppression des lignes dupliquées
    df = df.drop_duplicates()

    # Tri du dataframe par ordre croissant de date
    df.loc[:, 'DATE_MUTATION'] = pd.to_datetime(df['DATE_MUTATION'])
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

    # Calcul du prix au m² par commune dans df_train puis ajout dans df_test
    df_train['M2'] = df_train['MONTANT'] / df_train['SURFACE_BATI']
    df_avg_m2_commune = df_train.groupby('ID_COMMUNE')['M2'].mean().reset_index(name='prix_moyen_commune_m2')
    df_avg_m2_commune['prix_moyen_commune_m2']= df_avg_m2_commune['prix_moyen_commune_m2'].round(2)
    # df_train
    df_train = df_train.merge(df_avg_m2_commune, on='ID_COMMUNE')
    df_train = df_train.drop(["ID_COMMUNE","M2"], axis=1)
    # df_test
    df_test = df_test.merge(df_avg_m2_commune, on='ID_COMMUNE')
    df_test = df_test.drop(["ID_COMMUNE"], axis=1)

    # Séparation des données 
    y_train = df_train.loc[:,"MONTANT"]
    X_train = df_train.drop("MONTANT",axis=1)
    y_test = df_test.loc[:,"MONTANT"]
    X_test = df_test.drop("MONTANT", axis=1)
    print("Split OK")
    return X_train,y_train, X_test, y_test


#//////////////////////////////////////////////////////////////////////////////
#                       Option de filtrage des données
#//////////////////////////////////////////////////////////////////////////////
def filtrage(df : pd.DataFrame, tx_filtrage : List[int], where_clause :str ) -> pd.DataFrame:
    """
    Fonction permettant de supprimer les outliers et de filtrer les données

    Args :
    - df (pd.DataFrame) : Dataframe contenant les données pré-sélectionnées
    - tx_filtrage (List[int,int]) : Paramètres du filtrage
    - where_clause (str) : Condition de filtrage sur le type de bien et la région.

    Returns :
    - df_filtered (pd.DataFrame) : Dataframe avec les données filtrées.

    Remarques sur le taux de filtrage :

    - Le premier élément est le nombre de vente minimum par commune
    - Le deuxième élément est le pourcentage de outliers dans une commune

    Exemple pour tx_filtrage =[10,30] les lignes qui seront conservées seront :
    - Les communes dans lesquelles il restera minimum 10 ventes après suppression des outliers
    - ET  moins de 30% d'outliers
    - ET les lignes dont le montant est inférieur à la limite des outliers
    - tx_filtrage = [0,100] => Suppression des lignes dont le montant est supérieur à la limite des outliers.
    """
    # Affichage du nombre de communes et de vantes avant suppression
    print(f"Il y a {df.ID_COMMUNE.nunique()} communes avec plus de 10 ventes avant", end="")
    print(f" suppression des outliers, pour un total de {df.shape[0]} ventes")
    
    # Calcul de la limite haute pour enlever les outliers
    q1 = np.percentile(df["MONTANT"], 25)  # Premier quartile (25e percentile)
    q3 = np.percentile(df["MONTANT"], 75)  # Troisième quartile (75e percentile)
    iqr = q3 - q1  # Intervalle interquartile
    # Calcul de la limite supérieure des moustaches
    whisker_upper = q3 + 1.5 * iqr

    # Chargement des données pour réaliser les statistiques 
    df_supp = loading_data(f""" 
        WITH OUTLIERS AS(
        SELECT 
            V.ID_COMMUNE,
            count(*) AS nb_outliers  
        FROM VENTES V
        INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
        INNER JOIN COMMUNES C ON V.ID_COMMUNE=C.ID_COMMUNE
        INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
        INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION           
        {where_clause}
        AND MONTANT > {whisker_upper}
        GROUP BY ID_COMMUNE),

        ALL_VENTES AS(
        SELECT 
            V.ID_COMMUNE,
            count(*) AS total_ventes  
        FROM VENTES V
        INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
        INNER JOIN COMMUNES C ON V.ID_COMMUNE=C.ID_COMMUNE
        INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
        INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION           
        {where_clause}
        GROUP BY ID_COMMUNE
        )

        SELECT 
            A.*,
            B.total_ventes,
            B.total_ventes - A.nb_outliers AS ventes_restantes,
            C.NAME_COMMUNE,
            ROUND(A.nb_outliers*100/B.total_ventes,2) as pourcentage_ventes_retirees
        FROM OUTLIERS A

        INNER JOIN ALL_VENTES B ON A.ID_COMMUNE=B.ID_COMMUNE
        INNER JOIN COMMUNES C ON A.ID_COMMUNE=C.ID_COMMUNE

        ORDER BY nb_outliers DESC
        """)
    # Affichage d'un dataframe avec les statistiques, trié par ordre décroissant du nombre d'outliers
    display(df_supp.head(10))

    # Affichage de l'incidence du filtrage
    print(f"Nombre d'outliers : {df_supp.nb_outliers.sum()}")
    print(f"Nombre de commune avec des outliers : {df_supp.shape[0]}")
    print("Nombre de communes contenant des outliers et pour lesquels il", end="")
    print("reste plus de 10 ventes après suppression des outliers : ", end="")
    print(df_supp.loc[df_supp.ventes_restantes>=tx_filtrage[0],:].shape[0])
    print("Nombre de communes avec plus de 30% de ventes retirées : ",end="")
    print(df_supp.loc[df_supp.pourcentage_ventes_retirees>tx_filtrage[1],:].shape[0])
    filtre_de_supp = df_supp.loc[(df_supp.ventes_restantes<tx_filtrage[0]) | (df_supp.pourcentage_ventes_retirees>tx_filtrage[1]),"ID_COMMUNE"].tolist()
    print(f"Nombre de communes qui seront retirées : {len(filtre_de_supp)}")
    nb_lignes_apres_filtrage = df.loc[(~df['ID_COMMUNE'].isin(filtre_de_supp)) & (df.MONTANT<whisker_upper),:].shape[0]
    print(f"Nombre de ventes restantes après suppression des outlier et après filtrage : {nb_lignes_apres_filtrage}")
    print(f"Il y a donc eu {round((df.shape[0]-nb_lignes_apres_filtrage)*100/df.shape[0],2)}% de lignes supprimées après filtrage")
    
    df_filtered = df.loc[(~df['ID_COMMUNE'].isin(filtre_de_supp)) & (df.MONTANT<whisker_upper),:]
    return df_filtered


#//////////////////////////////////////////////////////////////////////////////
#                  Labellisation et standardisation des données
#//////////////////////////////////////////////////////////////////////////////
def encod_scal(X_train : pd.DataFrame, 
               X_test : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame , dict, dict]:
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


#//////////////////////////////////////////////////////////////////////////////
#                       RANDOM FOREST REGRESSOR
#//////////////////////////////////////////////////////////////////////////////
def train_model_randomforest(X_train : pd.DataFrame ,
                y_train : pd.Series,
                param_grid : dict, 
                cv : int) -> Tuple[BaseEstimator, Dict] : 
    """
    Fonction permettant d'entraîner un modèle et de sauvegarder un graphique 
    avec l'importance des variables.

    Args:
    - X_train (pd.Dataframe) : Données d'entrée d'entraînement.
    - y_train (pd.Series) : Données de sortie d'entraînement.
    - param_grid (dict) : Dictionnaire avec les différent hyperparamètres à tester.
    - cv (int) : Un entier pour choisir le nombre de pli pour la validation croisée.

    Return:
    - model (BaseEstimator) : model entraîné.
    - best_params (dict) : Dictionnaire avec les meilleurs hyperparamètres

    Remarque :
    - Le modèle est entraîné avec GridSearchCV en utilisant RandomForestRegressor() ,
    param_grid, le nombre de pli cv.
    - Les meilleurs paramètres du modèle sont établie en fonction de la métrique r2 score.
    - Le modèle retourné est celui avec les meilleurs paramètres.
    """
    print("Entraînement en cours ...")
    # Type de métriques pour la recherche de meilleurs paramètres
    scorer = {'mae': 'neg_mean_absolute_error' , 'r2' : 'r2'}

    # Entraînement avec les différentes paramètres
    grid_search = GridSearchCV(RandomForestRegressor(), 
                               param_grid, cv=cv, 
                               scoring=scorer,
                               refit='mae',
                               verbose=2)
    
    grid_search.fit(X_train, y_train)

    # Affichage des différents résultats
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by='rank_test_r2')
    display(results.head())
    best_params = grid_search.best_params_
    
    model = grid_search.best_estimator_
    print("Entraînement OK")
    return model, best_params



#//////////////////////////////////////////////////////////////////////////////
#                       TENSORFLOW SEQUENTIAL
#//////////////////////////////////////////////////////////////////////////////

#     Objet pour ajouter des métriques de performances au réseau de neurones
class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_r2 = []
        self.val_r2 = []
        self.train_rmse = []
        self.val_rmse = []

    def on_epoch_end(self, epoch, logs=None):
        train_pred = self.model.predict(self.X_train)
        val_pred = self.model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, train_pred)
        val_r2 = r2_score(self.y_test, val_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_test, val_pred))
        
        self.train_r2.append(train_r2)
        self.val_r2.append(val_r2)
        self.train_rmse.append(train_rmse)
        self.val_rmse.append(val_rmse)

def train_tensor_flow(X_train,y_train,X_test,y_test):
    model = Sequential([Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                        Dropout(0,2),
                        Dense(64, activation='relu'), 
                        Dense(1, activation='linear') 
                        ])
    # Imprimer le summary du modèle
    model.summary()

    # Compilation du modèle
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Définition de l'arrêt précoce (Early Stopping)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4,min_delta=0.005, restore_best_weights=True)
    # Réduction de la descente de gradient
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.1, patience=3,min_delta=0.01, min_lr=0.0001)
    # Enregistrement de métriques complémentaires
    metrics_callback = MetricsCallback(X_train, y_train, X_test, y_test)

    # Entraînement du modèle avec l'arrêt précoce
    history = model.fit(X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, metrics_callback],
    verbose=1)

    # Extraction des métriques
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    train_r2 = metrics_callback.train_r2
    val_r2 = metrics_callback.val_r2
    train_rmse = metrics_callback.train_rmse
    val_rmse = metrics_callback.val_rmse

    # Tracé des courbes
    epochs = range(1, len(train_mae) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_mae, 'b', label='Training MAE')
    plt.plot(epochs, val_mae, 'r', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_r2, 'b', label='Training R2 Score')
    plt.plot(epochs, val_r2, 'r', label='Validation R2 Score')
    plt.title('Training and Validation R2 Score')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_rmse, 'b', label='Training RMSE')
    plt.plot(epochs, val_rmse, 'r', label='Validation RMSE')
    plt.title('Training and Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()

    # Enregistrement des graphiques avec MLflow
    image_path="training_validation_metrics.png"
    plt.savefig(image_path)
    print("Entraînement OK")
    return model,image_path


#//////////////////////////////////////////////////////////////////////////////
#                               XGBoost
#//////////////////////////////////////////////////////////////////////////////
def train_model_xgboost(X_train: pd.DataFrame, 
                        y_train: pd.Series, 
                        param_grid: Dict, 
                        cv: int) -> Tuple[BaseEstimator, Dict]:
    """
    Fonction permettant d'entraîner un modèle XGBoost.

    Args:
    - X_train (pd.DataFrame): Données d'entrée d'entraînement.
    - y_train (pd.Series): Données de sortie d'entraînement.
    - param_grid (dict): Dictionnaire avec les différents hyperparamètres à tester.
    - cv (int): Un entier pour choisir le nombre de pli pour la validation croisée.

    Return:
    - model (BaseEstimator): Modèle entraîné.
    - best_params (dict): Dictionnaire avec les meilleurs hyperparamètres.

    Remarque :
    - Le modèle est entraîné avec GridSearchCV en utilisant XGBRegressor(),
    param_grid et le nombre de plis cv.
    - Les meilleurs paramètres du modèle sont établis en fonction de la métrique r2_score.
    - Le modèle est celui avec les meilleurs paramètres.
    """
    print("Entraînement en cours ...")
    
    # Type de métrique pour la recherche des meilleurs paramètres
    scorer = {'r2': 'r2'}

    # Entraînement avec les différents paramètres
    grid_search = GridSearchCV(XGBRegressor(), 
                               param_grid, 
                               cv=cv, 
                               scoring=scorer, 
                               refit='r2',
                               verbose=2)
    
    grid_search.fit(X_train, y_train)

    # Affichage des différents résultats
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by='rank_test_r2')
    display(results.head())
    best_params = grid_search.best_params_

    model = grid_search.best_estimator_
    return model, best_params

    
#//////////////////////////////////////////////////////////////////////////////
#             Graphiques pour visualiser les données d'entraînement
#//////////////////////////////////////////////////////////////////////////////
def plot_validation_learning_curve(model : BaseEstimator,
                        X_train : pd.DataFrame,
                        y_train : pd.Series) -> list:
    """
    Fonction permettant le tracé et l'enregistrement en local de la courbe d'apprentissage, 
    la learning curve ainsi que les futures importances avec la mae et le r2 score.

    Args:
    - model (BaseEstimator) : Modèle avec les meilleurs hyperparamètres.
    - X (pd.DataFrame) : Données d'entrée d'entraînement.
    - y (pd.Series) : Données de sortie d'entraînement.

    Return:
    - images (list) : Chemin vers l'image stockée en local.

    Remarque:
    - La fonction permet d'afficher la courbe d'apprentissage et d'enregistrer
    l'image en local.
    """
    print("Graphique en cours...")

    images =[]

    # Affichage de l'importance des variables et de la courbe d'apprentissage
    # et sauvegarde des graphiques
    importances = model.feature_importances_

    # Tri des indices des variables par importance
    indices = np.argsort(importances)[::-1]

    # Plot de l'importance des variables
    plt.figure(figsize=(10, 6))
    plt.title("Importance des variables")
    plt.bar(range(X_train.shape[1]), importances[indices], color="b", align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.xlabel("Variables")
    plt.ylabel("Importance")
    plt.tight_layout()
    image_path_feature = "./images/feature_importance.png"
    images.append(image_path_feature)
    plt.savefig(image_path_feature)
    plt.show()

    # Courbe d'apprentissage
    train_sizes=[i / 10.0 for i in range(1, 11)]
    train_sizes[-1]=0.99
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X_train, y_train, train_sizes=train_sizes, cv=3)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                    validation_scores_mean + validation_scores_std, alpha=0.1,
                    color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Score d'entraînement")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g",
            label="Score de validation croisée")
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score")
    plt.title("Courbe d'apprentissage")
    plt.legend(loc="best")
    image_path_learning = "./images/feature_learning.png"
    images.append(image_path_learning)
    plt.savefig(image_path_learning)
    plt.show()


    # Courbe de validation
    param_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    train_scores, test_scores = validation_curve(model, X_train, y_train,
                         param_name='n_estimators', 
                         param_range=param_range,
                         cv = 3,
                         scoring='r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.rcParams["font.size"] = 12
    plt.title("Validation Curve (R2)", fontsize = 20)
    plt.xlabel("n_estimators", fontsize =14)
    plt.ylabel("Score", fontsize = 14)
    plt.ylim(0.5, 0.9)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.rcParams["font.size"] = 10
    plt.legend(loc="best")
    image_path_validation = "./images/feature_validation.png"
    images.append(image_path_validation )
    plt.savefig(image_path_validation)
    plt.show()
    print("Entraînement OK")
    return images

#//////////////////////////////////////////////////////////////////////////////
#                                    MLFlow
#//////////////////////////////////////////////////////////////////////////////
class param_mlflow():
    def __init__(self, 
                 uri_tracking : str,
                  model : BaseEstimator,
                  X_test : pd.DataFrame,
                  y_test : pd.DataFrame,
                  experiment_name : str ="test", 
                  run_name : str ="test",
                  model_name : str = "test",
                  images : List[str] =[],
                  best_params : dict =None,
                  encoders : dict =None,
                  scalers : dict =None,                
                  ):
        """
        Classe permettant d'instancier un objet en vue d'enregistrer les 
        paramètres fournis dans mlflow

        Args :
        - uri_tracking (str): URI de suivi MLflow.
        - experiment_name (str): Nom de l'expérience MLflow.
        - run_name (str): Nom de l'exécution MLflow.
        - best_params (dict): Meilleurs paramètres du modèle.
        - model (BaseEstimator): Le modèle d'apprentissage automatique entraîné.
        - model_name (str): Nom du modèle d'apprentissage automatique.
        - X_test (pd.DataFrame): Caractéristiques de l'ensemble de données de test.
        - y_test (pd.Series): Variable cible de l'ensemble de données de test.
        - encoders (LabelEncoder): Encodeurs de prétraitement utilisés sur les données.
        - scalers (StandardScaler): Scalers de prétraitement utilisés sur les données.
        - image_path (str): Lien vers l'image 
        - image_path_feature (str) : Lien vers l'image avec les feature importances.

        Returns : None
        """
        self.uri_tracking = uri_tracking
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model_name = model_name
        self.best_params = best_params
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.encoders = encoders
        self.scalers = scalers
        self.images =images

    def log_mlflow(self) -> None:
        """ 
        Fonction permettant de loguer les paramètres fournis à l'objet param_mlflow
        """
        print("Log du modèle et des artifacts en cours...")
        mlflow.set_tracking_uri(self.uri_tracking)

        mlflow.sklearn.autolog()

        # Vérification que l'experience existe
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            # Si l'expérience n'existe pas, la créer
            mlflow.create_experiment(self.experiment_name)
            # On récupère à nouveau l'expérience après la création
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

        with mlflow.start_run(experiment_id = experiment.experiment_id, 
                              run_name=self.run_name):  
            try :
                if self.best_params is not None :      
                    # Enregistrement des meilleurs paramètres du modèle
                    mlflow.log_params(self.best_params)

                # Enregistrement des graphiques
                if len(self.images)!=0:
                    for image in self.images :
                        mlflow.log_artifact(image)

                # Calcul des métriques
                y_pred = self.model.predict(self.X_test)
                r2 = r2_score(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                mape = mean_absolute_percentage_error(self.y_test, y_pred)
                msle = mean_squared_log_error(self.y_test, y_pred)

                # Enregistrement des métriques
                mlflow.log_metric("train_r2", r2)
                mlflow.log_metric("train_mse", mse)
                mlflow.log_metric("train_rmse", rmse)
                mlflow.log_metric("train_mae", mae)
                mlflow.log_metric("train_mape", mape)
                mlflow.log_metric("train_msle", msle)

                # Enregistrement du modèle
                input_example = self.X_test.head(1)
                signature = infer_signature(input_example,self.model.predict(input_example))
                mlflow.sklearn.log_model(self.model,
                                        "ImmoApp",
                                        input_example = input_example,
                                        signature=signature,
                                        registered_model_name = self.model_name)
                
                # Sauvegarde des encoders et scalers
                if len(self.encoders)!=0:
                    joblib.dump(self.encoders, "./encoders.joblib")
                    mlflow.log_artifact("./encoders.joblib")
                if len(self.scalers)!=0:
                    joblib.dump(self.scalers, "./scalers.joblib")
                    mlflow.log_artifact("./scalers.joblib")
            except Exception as e:
                print(f"Erreur dans le chargement de données dans mlflow : {e}")

            finally:
                # Fermeture du run
                mlflow.end_run()    
        return