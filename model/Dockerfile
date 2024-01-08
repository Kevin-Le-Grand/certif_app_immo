FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update

RUN pip install psycopg2-binary boto3 mlflow

CMD mlflow server \
    --port $PORT \
    --host 0.0.0.0 \
    --artifacts-destination $ARTIFACT_STORE_URI \
    --backend-store-uri $BACKEND_STORE_URI 