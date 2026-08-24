FROM python:3.10-slim

WORKDIR /code
ENV HF_HOME=/data
RUN mkdir -p /data/snapshots && chmod 777 -R /data

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./scripts /code/scripts
COPY ./models /code/models
COPY ./bot /code/bot
COPY ./data /code/data

RUN ls -lR /code

# Expõe a porta que a API vai usar
EXPOSE 8000

# Comando para iniciar a API quando o contêiner rodar
CMD ["uvicorn", "scripts.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]