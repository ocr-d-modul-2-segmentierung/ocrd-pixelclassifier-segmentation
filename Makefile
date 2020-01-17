SHELL = /bin/bash
PYTHON = python
PIP = pip

DOCKER_TAG = 'ls6uniwue/ocrd_pixelclassifier_segmentation'

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps         install dependencies"
	@echo "    install      install package"
	@echo "    test-cli     run cli tests"
	@echo "    docker       build docker container"


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
test-cli: clean-workspace install test-workspace
	cd test-workspace/data && \
		ocrd-pc-segmentation -l DEBUG -m mets.xml -I OCR-D-IMG-BIN -O OCR-D-SEG-BLOCK

test-workspace: test/assets
	cp -rv test/assets/kant_aufklaerung_1784-binarized test-workspace

.PHONY: clean-workspace
clean-workspace:
	rm -rfv test-workspace

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
