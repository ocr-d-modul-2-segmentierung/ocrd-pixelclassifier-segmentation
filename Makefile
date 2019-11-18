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

# Install testing python deps via pip
deps-test:
	$(PIP) install -r requirements_test.txt

# Install
install: deps
	$(PIP) install .

# Build docker image
docker:
	docker build -t $(DOCKER_TAG) .

#TODO tests

.PHONY: install deps


test: test/assets

# Test the command line tools
test-cli: test/assets install
	rm -rfv test-workspace
	cp -rv test/assets/kant_aufklaerung_1784-binarized test-workspace
	cd test-workspace/data && \
		ocrd-pc-segmentation -l DEBUG -m mets.xml -I OCR-D-IMG-BIN -O OCR-D-SEG-BLOCK

#
# Assets
#

# Clone OCR-D/assets to ./repo/assets
repo/assets:
	mkdir -p $(dir $@)
	git clone https://github.com/OCR-D/assets "$@"


# Setup test assets
test/assets: repo/assets
	mkdir -p $@
	cp -r -t $@ repo/assets/data/*

.PHONY: assets-clean
# Remove symlinks in test/assets
assets-clean:
	rm -rf test/assets
