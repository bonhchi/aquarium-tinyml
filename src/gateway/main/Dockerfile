FROM python:3.11-slim

WORKDIR /app

# Copy the shared gateway dependency list from the repo root.
COPY src/gateway/requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools --no-cache-dir
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Ship only the Flask API sources.
COPY src/gateway/flask_api/ /app

EXPOSE 5000

CMD ["python", "app.py"]
