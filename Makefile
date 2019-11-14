SHELL = /bin/bash
PYTHON = python
PIP = pip

DOCKER_TAG = 'ls6uniwue/ocrd_pixelclassifier_segmentation'

ifndef TENSORFLOW_GPU
	TENSORFLOW_VARIANT = tf_cpu
else
	TENSORFLOW_VARIANT = tf_gpu
endif

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps         install dependencies"
	@echo "    install      install package"
	@echo ""
	@echo "  Environment variables"
	@echo ""
	@echo "    TENSORFLOW_GPU   if set, uses tensorflow-gpu"
	@echo "                     requires working cuDNN setup"


# Install python deps via pip
deps:
	$(PIP) install -r requirements.txt
	$(PIP) install ocr4all-pixel-classifier[$(TENSORFLOW_VARIANT)]

# Install
install: deps
	$(PIP) install .

# Build docker image
docker:
	docker build -t $(DOCKER_TAG) .

#TODO tests

.PHONY: install deps
