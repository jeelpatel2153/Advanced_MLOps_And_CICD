# Advanced MLOps CICD Project

## Features
- CI/CD using GitHub Actions
- Model training & testing
- Flask API
- Docker container
- Drift monitoring

## Run Project
pip install -r requirements.txt
python src/train.py
pytest
python app.py
docker run -p 5000:5000 mlops-project