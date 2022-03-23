SHELL = bash
PYTHON = python
PIP = pip

DOCKER_TAG = 'ls6uniwue/ocrd_pixelclassifier_segmentation'

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps          Install python deps via pip"
	@echo "    deps-test     Install testing python deps via pip"
	@echo "    install       Install"
	@echo "    docker        Build docker image"
	@echo "    test-cli      Test the command line tools"
	@echo "    repo/assets   Clone OCR-D/assets to ./repo/assets"
	@echo "    test/assets   Setup test assets"
	@echo "    assets-clean  Remove symlinks in test/assets"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    TENSORFLOW_GPU  If set to 1, uses tensorflow-gpu. Requires working cuDNN setup. Default: $(TENSORFLOW_GPU)"

# END-EVAL

# Install python deps via pip
deps: requirements.txt
	$(PIP) install -r requirements.txt
	$(PIP) install 'ocr4all-pixel-classifier'

# Install testing python deps via pip
deps-test:
	$(PIP) install -r requirements_test.txt

# Install
install: deps
	$(PIP) install .

# Build docker image
docker:
	docker build -t $(DOCKER_TAG) .

.PHONY: install deps


# TODO tests
# test: test/assets

# Test the command line tools
test-cli: test/assets # install
	rm -rfv test-workspace
	cp -rv test/assets/kant_aufklaerung_1784-binarized test-workspace
	ocrd-pc-segmentation -m test-workspace/data/mets.xml -I OCR-D-IMG-BIN -O OCR-D-SEG-BLOCK
	fgrep -c -e TextRegion -e ImageRegion test-workspace/data/OCR-D-SEG-BLOCK/*.xml

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

.PHONY: requirements.txt
requirements.txt: pip-tools
	pip-compile --upgrade

.PHONY: pip-tools
pip-tools:
	$(PIP) install -U pip-tools
