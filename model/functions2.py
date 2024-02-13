import os
import numpy as np
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,mean_squared_log_error
import joblib
import pandas as pd
from connection import connection_with_sqlalchemy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,Callback
from sqlalchemy import text
import matplotlib.pyplot as plt
from typing import Tuple, Dict



#//////////////////////////////////////////////////////////////////////////////
#     Objet pour ajouter des métriques de performances au réseau de neurones
#//////////////////////////////////////////////////////////////////////////////
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
#                       Split des données (2 méthodes)
#//////////////////////////////////////////////////////////////////////////////
def split(df: pd.DataFrame):
    """ 
    Fonction permettant de séparer les données en données d'entraînement et de test

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
    # Séparation des données 
    y_train = df_train.loc[:,"MONTANT"]
    X_train = df_train.drop("MONTANT",axis=1)
    y_test = df_test.loc[:,"MONTANT"]
    X_test = df_test.drop("MONTANT", axis=1)
    print("Split OK")
    return X_train,y_train, X_test, y_test


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
#                  Labellisation et standardisation des données
#//////////////////////////////////////////////////////////////////////////////
def encod_scal(X_train : pd.DataFrame, X_test : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame , dict, dict]:
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
#             Fonctions pour les différents modèles
#//////////////////////////////////////////////////////////////////////////////
def train_model_randomforest(X_train : pd.DataFrame ,
                y_train : pd.Series,
                param_grid : dict, 
                cv : int) -> Tuple[BaseEstimator, Dict] : 
    """
    Fonction permettant d'entraîner un modèle.

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
    - Le modèle est ré-entraîné avec les meilleurs paramètres
    """
    print("Entraînement en cours ...")
    # Type de métriques pour la recherche de meilleurs paramètres
    scorer = make_scorer(r2_score)

    # Entraînement avec les différentes paramètres
    grid_search = GridSearchCV(RandomForestRegressor(), 
                               param_grid, cv=cv, 
                               scoring=scorer,
                               verbose=2)
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    # Entraînement des données avec les meilleurs paramètres
    print("Ré-entraînement avec les meilleurs hyperparamètres en cours...")
    model=RandomForestRegressor(**best_params)
    model.fit(X_train,y_train)
    print("Entraînement OK")
    return model, best_params

def train_tensor_flow(X_train,y_train,X_test,y_test):
    model = Sequential([Dense(64, activation='relu',
    input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'), Dense(1) # Couche de sortie avec une seule sortie pour la régression
    ])
    # Imprimer le summary du modèle
    model.summary()

    # Compilation du modèle
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Définition de l'arrêt précoce (Early Stopping)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Réduction de la descente de gradient
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)
    # Enregistrement de metriques complémentaires
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
#             Graphiques pour visualiser les données d'entraînement
#//////////////////////////////////////////////////////////////////////////////
def plot_learning_curve(estimator : BaseEstimator,
                        type_de_bien : str ,
                        X : pd.DataFrame,
                        y : pd.Series) -> str:
    """
    Fonction permettant le tracé et l'enregistrement en local de la courbe d'apprentissage.

    Args:
    - estimator (BaseEstimator) : Modèle avec les meilleurs hyperparamètres.
    - type_de_bien (str) : Informations sur les données entraînées.
    - X (pd.DataFrame) : Données d'entrée d'entraînement.
    - y (pd.Series) : Données de sortie d'entraînement.

    Return:
    - image_path (str) : Chemin vers l'image stockée en local.

    Remarque:
    - La fonction permet d'afficher la courbe d'apprentissage et d'enregistrer
    l'image en local.
    """
    print("Tracé du graphique en cours...")
    train_sizes=[i / 10.0 for i in range(1, 11)]
    train_sizes[-1]=0.99
    train_scores = []
    validation_scores = []

    iteration=1
    for size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(X, y, train_size=size, random_state=42)
        estimator.fit(X_train_subset, y_train_subset)
        train_scores.append(estimator.score(X_train_subset, y_train_subset))
        validation_scores.append(estimator.score(X, y))
        print(f"Itération N°{iteration} : train score = {train_scores[-1]} -- validation score = {validation_scores[-1]}")
        iteration+=1

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Score d'entraînement")
    plt.plot(train_sizes, validation_scores, 'o-', color="g", label="Score de validation")

    plt.title(f"Courbe d'apprentissage pour les {type_de_bien}s")
    plt.xlabel("Taille de l'échantillon d'entraînement")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    # Enregistrer l'image localement
    image_path = f"./Learning_curve_{type_de_bien}.png"
    plt.savefig(image_path)
    plt.show()
    print("Tracé OK")
    return image_path


#//////////////////////////////////////////////////////////////////////////////
#                                    MLFlow
#//////////////////////////////////////////////////////////////////////////////
def log_mlflow(uri_tracking : str,
               experiment_name : str,
               run_name : str,
               best_params : dict,
               model : BaseEstimator, model_name : str,
               X_test : pd.DataFrame, y_test : pd.Series,
               encoders : LabelEncoder ,scalers : StandardScaler,
               image_path : str) -> None:
    """
    Fonction permettant d'enregistrer le modèle et les artifacts

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

    Returns : None
    """
    print("Log du modèle et des artifacts en cours...")
    mlflow.set_tracking_uri(uri_tracking)

    mlflow.sklearn.autolog()

    # Vérification que l'experience existe
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Si l'expérience n'existe pas, la créer
        mlflow.create_experiment(experiment_name)
        # On récupère à nouveau l'expérience après la création
        experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=run_name):        
        # Enregistrement des meilleurs paramètres du modèle
        mlflow.log_params(best_params)

        # Enregistrement de la figure learning curve
        mlflow.log_artifact(image_path)

        # Calcul des métriques
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)

        # Enregistrement des métriques
        mlflow.log_metric("train_r2", r2)
        mlflow.log_metric("train_mse", mse)
        mlflow.log_metric("train_rmse", rmse)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_mape", mape)
        mlflow.log_metric("train_msle", msle)

        # Enregistrement du modèle
        input_example = X_test.head(1)
        signature = infer_signature(input_example,model.predict(input_example))
        mlflow.sklearn.log_model(model,
                                "ImmoApp",
                                input_example = input_example,
                                signature=signature,
                                registered_model_name = model_name)
        
        # Sauvegarde des encoders et scalers
        joblib.dump(encoders, "./encoders.joblib")
        joblib.dump(scalers, "./scalers.joblib")
        mlflow.log_artifact("./encoders.joblib")
        mlflow.log_artifact("./scalers.joblib")

        # Fermeture du run
        mlflow.end_run()    
    return