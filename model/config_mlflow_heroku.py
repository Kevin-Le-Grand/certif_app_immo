import mlflow
import os

if __name__ == "__main__":
    variable = os.environ.get("BACKEND_STORE_URI")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",variable)
    port = int(os.environ.get("PORT", 5000))
    mlflow.set_tracking_uri(os.environ.get("BACKEND_STORE_URI"))
    mlflow.sklearn.autolog()
    mlflow.start()

