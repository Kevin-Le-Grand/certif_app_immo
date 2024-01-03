# Connect to Heroku
heroku login

# Heroku container login
heroku container:login

# Create Heroku app
heroku create immoappkevleg2 --region eu

# Build Image MAC ARM
# docker buildx build --platform linux/amd64 -t streamlit-isen-g1  .

# Build Image
docker build -t immoappkevleg2 .

# Tag Image to Heroku app
docker tag immoappkevleg2 registry.heroku.com/immoappkevleg2/web

# Push Image to Heroku
docker push registry.heroku.com/immoappkevleg2/web

# Release Image to Heroku
heroku container:release web -a immoappkevleg2

# Environ keys
heroku config:set AWS_ACCESS_KEY_ID=AKIA5JTHFGQOCZZAOYY2 AWS_SECRET_ACCESS_KEY=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 DB_HOST=datagouv.c4yi8wliu37x.eu-west-3.rds.amazonaws.com DB_USER=kevappimmo DB_PASSWORD=KevAppImmo50! DB_PORT=3306 -a immoappkevleg2

# Open Heroku app
heroku open -a immoappkevleg2

# Logs
heroku logs --tail -a immoappkevleg2