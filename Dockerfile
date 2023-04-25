FROM python:latest

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.0.0

WORKDIR /src/UsovikTask

COPY . .

RUN pip install poetry && \
    poetry install

RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["poetry", "run", "python", "main.py"]