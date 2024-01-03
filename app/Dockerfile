FROM continuumio/miniconda3

WORKDIR /home/app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the repository
RUN git clone https://github.com/rastakoer/certif_app_immo.git --branch application .

# Copiez les fichiers nécessaires
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . /home/app

# Set the command to run on container start
CMD streamlit run app/app.py --server.port $PORT
