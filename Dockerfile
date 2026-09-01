# Container image for api/main.py (the churn-prediction API). Same overall
# shape as the emotisense project's Dockerfile (Lambda Web Adapter as an
# inert-outside-Lambda extension, so this one Dockerfile also works for a
# plain `docker run` locally) - but meaningfully simpler: no build-essential
# (scikit-learn ships prebuilt wheels, nothing to compile) and no
# Hugging-Face-Hub-style runtime model download. The model is small enough
# (~140KB) to just bake into the image directly.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Only what api/main.py actually imports at runtime: itself, the trained
# model + its metadata, features.py (FEATURE_COLUMNS), and prediction_log.py
# (S3 logging).
COPY api/ api/
COPY features.py features.py
COPY prediction_log.py prediction_log.py
COPY models/ models/

# Lambda Web Adapter - lets Lambda invoke a plain uvicorn HTTP server (no
# handler rewrite needed). No-op outside Lambda. See
# https://github.com/awslabs/aws-lambda-web-adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_INVOKE_MODE=buffered

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
