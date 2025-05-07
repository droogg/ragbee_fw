# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Отключаем cuda-репы, чтобы apt-update не жаловался
RUN sed -i 's/^deb.*cuda/#&/' /etc/apt/sources.list.d/*.list

# Устанавливаем системные пакеты и чистим кеш
RUN apt-get update \
 && apt-get install -y python3 python3-pip git tmux wget curl locales \
 && rm -rf /var/lib/apt/lists/*

# Локаль
RUN locale-gen ru_RU.UTF-8
ENV LANG=ru_RU.UTF-8 LANGUAGE=ru_RU:ru LC_ALL=ru_RU.UTF-8

WORKDIR /workspace

# Poetry (без создания venv)
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV PATH="/usr/local/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Апгрейд pip для Poetry и прочих инструментов
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Установка зависимостей проекта через Poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-ansi

# Nvidia-runtime ENV
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["bash"]