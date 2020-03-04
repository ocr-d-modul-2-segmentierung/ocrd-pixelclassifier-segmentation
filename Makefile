SHELL = bash
PYTHON = python
PIP = pip

DOCKER_TAG = 'ls6uniwue/ocrd_pixelclassifier_segmentation'

# If set to 1, uses tensorflow-gpu. Requires working cuDNN setup. Default: $(TENSORFLOW_GPU)
TENSORFLOW_GPU ?= 0

ifeq ($(TENSORFLOW_GPU),1)
TENSORFLOW_VARIANT = tf_gpu
else
TENSORFLOW_VARIANT = tf_cpu
endif

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
deps:
	$(PIP) install -r requirements.txt
	$(PIP) install 'ocr4all-pixel-classifier[$(TENSORFLOW_VARIANT)]'

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
