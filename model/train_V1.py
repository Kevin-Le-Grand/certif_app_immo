import os
import numpy as np
import mlflow
from mlflow.models import infer_signature
from functions import train_model,loading_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,mean_squared_log_error
import joblib


def main():
    # Récupération des données
    df = loading_data()

    # Préparation des données
    model, encoders, scalers, X_train, y_train = train_model(df)

    print("Chargement des artifacts avec mlflow en cours")
    # Connexion à MLflow
    mlflow.set_tracking_uri("https://mlflowimmoappkevleg-737621d410d0.herokuapp.com/")

    # Configuration de l'autolog
    mlflow.sklearn.autolog()

    # Connexion à une expérience
    experiment_name = "test"
    run_name = "Training_V1"
    model_name = "RFR"

    # Vérification que l'experience existe
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Si l'expérience n'existe pas, la créer
        mlflow.create_experiment(experiment_name)
        # On récupère à nouveau l'expérience après la création
        experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id = experiment.experiment_id, run_name=run_name):

        # Calcul des métriques
        r2 = model.score(X_train, y_train)
        mse = mean_squared_error(y_train, model.predict(X_train))
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_train, model.predict(X_train))
        mape = mean_absolute_percentage_error(y_train, model.predict(X_train))
        msle = mean_squared_log_error(y_train, model.predict(X_train))

        # Enregistrement des métriques
        mlflow.log_metric("train_r2", r2)
        mlflow.log_metric("train_mse", mse)
        mlflow.log_metric("train_rmse", rmse)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_mape", mape)
        mlflow.log_metric("train_msle", msle)

        mlflow.sklearn.log_model(model,
                                "ImmoApp",
                                input_example = X_train.head(1),
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