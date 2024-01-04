FROM continuumio/miniconda3

WORKDIR /home/app/

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/rastakoer/certif_app_immo.git --branch model --single-branch .

RUN pip install -r requirements.txt

CMD python3 train.py