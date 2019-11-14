# WORK IN PROGRESS - NOT READY
FROM ocrd/core
VOLUME ["/data"]
MAINTAINER OCR-D
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8

WORKDIR /build-ocrd
COPY setup.py .
COPY README.md .
COPY requirements.txt .
#COPY requirements_test.txt .
COPY ocrd_pc_segmentation ./ocrd_pc_segmentation
COPY Makefile .
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        build-essential \
        python-opencv \
    && make deps install \
    && apt-get -y remove --auto-remove build-essential
