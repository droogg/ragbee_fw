FROM nvidia/cuda:12.4.0-devel-ubuntu22.04


RUN sed -i 's/^deb.*cuda/#&/' /etc/apt/sources.list.d/* 
RUN apt-get update && apt-get install -y \
    python3 python3-pip git tmux wget curl locales && apt-get clean

RUN locale-gen ru_RU.UTF-8
ENV LANG ru_RU.UTF-8
ENV LANGUAGE ru_RU:ru
ENV LC_ALL ru_RU.UTF-8

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip &&\
    pip install --no-cache-dir poetry

WORKDIR /workspace

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITES compute,utility

CMD ["bash"]