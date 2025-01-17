# local run

# 1. install poetry in python

# 2. create virtual enviroment and activate

# 3. use poetry install --no-root 
it installs all dependencies

# 4. run either the notebook for local run or run trainer script and app.py

# docker run USING DOCKER RUN
# 1. make sure docker is up and running 
# 2. build image `docker build -t flask-ml-app .`
# 3. run docker to start simple predictive site `docker run -p 5000:5000 flask-ml-app`

# 4. open http://localhost:5000/
# 5. upload cleaned data for prediction

# docker run USING DOCKER COMPOSE
# 1. make sure docker is up and running 
# 2. build image `docker-compose up --build -d`
# 3. open http://localhost:5000/
# 4. upload cleaned data for prediction