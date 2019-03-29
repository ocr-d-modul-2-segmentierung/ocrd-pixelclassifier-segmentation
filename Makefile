export

SHELL = /bin/bash
PYTHON = python
PIP = pip
LOG_LEVEL = INFO
PYTHONIOENCODING=utf8

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps         Install python deps via pip (use cpu tensorflow)"
	@echo "    deps-gpu     Install python deps via pip (use gpu tensorflow)"
	@echo "    install      Install"

# END-EVAL

# Install python deps via pip
deps:
	# cpu and gpu tensorflow may conflict, remove other
	$(PIP) uninstall -y -q tensorflow_gpu
	$(PIP) install -r requirements.txt
	$(PIP) install ./page-segmentation[tf_cpu]

deps-gpu:
	# cpu and gpu tensorflow may conflict, remove other
	$(PIP) uninstall -y -q tensorflow
	$(PIP) install -r requirements.txt
	$(PIP) install ./page-segmentation[tf_gpu]

# Install
install:
	@echo "Notice: Either \"make deps\" or \"make deps-gpu\" must be run"
	@echo "        to select tensorflow version"
	$(PIP) install -e .
