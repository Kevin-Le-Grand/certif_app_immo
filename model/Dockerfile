FROM continuumio/miniconda3

WORKDIR /home/app/

RUN apt-get update && apt-get install -y git nano

RUN git clone https://github.com/rastakoer/certif_app_immo.git --branch model --single-branch .

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "config_mlflow_heroku.py"]