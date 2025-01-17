FROM python:3.11

COPY . .

RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root 
# RUN gunzip model.pkl.gz
CMD ["python", "app.py"]

EXPOSE 5000
