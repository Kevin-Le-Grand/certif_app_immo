# Connect to Heroku
heroku login

# Heroku container login
heroku container:login

# Create Heroku app
heroku create immo_app

# Build Image MAC ARM
# docker buildx build --platform linux/amd64 -t streamlit-isen-g1  .

# Build Image
docker build . -t immo_app

# Tag Image to Heroku app
docker tag immo_app registry.heroku.com/application_immo/web

# Push Image to Heroku
docker push registry.heroku.com/application_immo/web

# Release Image to Heroku
heroku container:release web -a application_immo

# Open Heroku app
heroku open -a application_immo

# Logs
heroku logs --tail -a application_immo