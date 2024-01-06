import mlflow
import os

variable = os.environ.get("BACKEND_STORE_URI")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",variable)
mlflow.sklearn.autolog()