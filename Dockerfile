FROM python:3.10-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /code

# Copia o arquivo de dependências para o contêiner
COPY ./requirements.txt /code/requirements.txt

# Instala as dependências
# O --no-cache-dir economiza espaço na imagem final
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia as pastas com seus scripts e o modelo treinado
COPY ./scripts /code/scripts
COPY ./models /code/models

# Expõe a porta em que a API vai rodar
EXPOSE 8000

# Comando para iniciar a API quando o contêiner for executado
# O --host 0.0.0.0 é essencial para que a API seja acessível de fora do contêiner
CMD ["uvicorn", "scripts.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]