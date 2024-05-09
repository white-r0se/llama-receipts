FROM python:3.11 as dependencies

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=10 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    VENV_PATH="/opt/pysetup/.venv" \
    PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH" \
    POETRY_VERSION=1.6.1

WORKDIR /opt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libcurl4 \
    libcurl4-openssl-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates --fresh

COPY pyproject.toml poetry.lock ./

RUN pip install --no-compile --no-cache-dir -U pip poetry==${POETRY_VERSION} \
    && ln -sf /etc/ssl/certs/ca-certificates.crt $(python -m certifi 2>&1 | grep cacert.pem) \
    && poetry install --only main \
    && ln -sf /etc/ssl/certs/ca-certificates.crt $(poetry run python -m certifi 2>&1 | grep cacert.pem) \
    && rm -rf ~/.cache/pypoetry \
    && find /opt -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete

FROM dependencies as development

RUN poetry install \
    && rm -rf ~/.cache/pypoetry \
    && find /opt -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete
