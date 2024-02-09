import os
import numpy as np
import mlflow
from mlflow.models import infer_signature
from functions import train_model,loading_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,mean_squared_log_error
import joblib


def main():
    for i in ("Appartement","Maison"):
        # Requêtes permettant de récupérer les données
        if i=="Maison":
            query=f"""
            SELECT 
                V.SURFACE_BATI,
                V.DEPENDANCES,
                V.ID_COMMUNE,
                R.Name_region,
                V.SURFACE_TERRAIN,
                V.DATE_MUTATION,
                V.MONTANT
            FROM VENTES V
            INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
            INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
            INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
            INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
            WHERE V.MONTANT>15000
            AND T.NAME_TYPE_BIEN='{i}'
            AND V.SURFACE_BATI>0
            AND V.NB_PIECES>0
            AND R.Name_region NOT IN("Martinique",
                                    "Guyane",
                                    "La Réunion",
                                    "Mayotte",
                                    "Guadeloupe");
            """
        else:
            query=f"""
            SELECT 
                V.SURFACE_BATI,
                V.DEPENDANCES,
                V.ID_COMMUNE,
                R.Name_region,
                V.DATE_MUTATION,
                V.MONTANT
            FROM VENTES V
            INNER JOIN TYPES_BIENS as T ON V.ID_TYPE_BIEN = T.ID_TYPE_BIEN
            INNER JOIN COMMUNES AS C ON V.ID_COMMUNE = C.ID_COMMUNE
            INNER JOIN DEPARTEMENTS AS D ON C.ID_DEPT = D.ID_DEPT
            INNER JOIN REGIONS R ON D.ID_REGION = R.ID_REGION
            WHERE V.MONTANT>15000 AND V.MONTANT<6500000
            AND T.NAME_TYPE_BIEN='{i}'
            AND V.SURFACE_BATI>0
            AND V.NB_PIECES>0
            AND R.Name_region NOT IN("Martinique","Guyane","La Réunion","Mayotte","Guadeloupe");
            """
        
        # Récupération des données
        df = loading_data(query)

        # Préparation des données
        model, encoders, scalers, X_test, y_test, best_params = train_model(df)

        print("Chargement des artifacts avec mlflow en cours")
        # Connexion à MLflow
        mlflow.set_tracking_uri("https://mlflowimmoappkevleg-737621d410d0.herokuapp.com/")

        # Configuration de l'autolog
        mlflow.sklearn.autolog()

        # Connexion à une expérience
        experiment_name = "RandomForestRegressor_all_datas"
        run_name = i
        model_name = f"RFR_all_datas_{i}"

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

            # Calcul des métriques
            r2 = model.score(X_test, y_test)
            mse = mean_squared_error(y_test, model.predict(X_test))
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, model.predict(X_test))
            mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
            msle = mean_squared_log_error(y_test, model.predict(X_test))

            # Enregistrement des métriques
            mlflow.log_metric("train_r2", r2)
            mlflow.log_metric("train_mse", mse)
            mlflow.log_metric("train_rmse", rmse)
            mlflow.log_metric("train_mae", mae)
            mlflow.log_metric("train_mape", mape)
            mlflow.log_metric("train_msle", msle)

            # Enregistrement du modèle
            mlflow.sklearn.log_model(model,
                                    "ImmoApp",
                                    input_example = df.head(5).drop(columns=['DATE_MUTATION']),
                                    registered_model_name = model_name)
            
            # Sauvegarde des encoders et scalers
            joblib.dump(encoders, "./encoders.joblib")
            joblib.dump(scalers, "./scalers.joblib")
            mlflow.log_artifact("./encoders.joblib")
            mlflow.log_artifact("./scalers.joblib")

            # Fermeture du run
            mlflow.end_run()    

if __name__ == '__main__':
    main()