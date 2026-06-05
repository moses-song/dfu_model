FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

COPY mvp1_classification/requirements.txt ./mvp1_classification/requirements.txt
RUN pip install --upgrade pip && pip install -r ./mvp1_classification/requirements.txt

COPY mvp1_classification ./mvp1_classification
COPY Model_training ./Model_training

WORKDIR /srv/mvp1_classification

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
