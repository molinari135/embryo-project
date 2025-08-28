#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = embryo-project
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

TRAIN_DIR ?= data/train
NUM_AUG ?= 2

MODEL_NAME ?= ResNet18LSTM
BATCH_SIZE ?= 64
NUM_EPOCHS ?= 100
PATIENCE ?= 5
LR ?= 1e-4
WEIGHT_DECAY ?= 1e-4
MAX_SEQ_LEN ?= 20


#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) embryo_project/dataset.py

## Run data augmentation
.PHONY: augment
augment: requirements
	$(PYTHON_INTERPRETER) embryo_project/augment.py augmentation --train-dir data/train --num-aug 5

## Run data augmentation only for class-1 folders
.PHONY: augment-class1
augment-class1: requirements
	$(PYTHON_INTERPRETER) embryo_project/augment.py augmentation --train-dir data/train --num-aug 5 --only-class1

## Remove augmented folders
.PHONY: augment-clean
augment-clean: requirements
	$(PYTHON_INTERPRETER) embryo_project/augment.py clean --train-dir data/train

## Train the model
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) embryo_project/modeling/train.py training \
		--model-name $(MODEL_NAME) \
		--batch-size $(BATCH_SIZE) \
		--num-epochs $(NUM_EPOCHS) \
		--patience $(PATIENCE) \
		--lr $(LR) \
		--weight-decay $(WEIGHT_DECAY)

## Evaluate a trained model
.PHONY: eval
eval: requirements
	$(PYTHON_INTERPRETER) embryo_project/modeling/predict.py main \
		--model-name $(MODEL_NAME) \
		--batch-size $(BATCH_SIZE) \
		--max-seq-len $(MAX_SEQ_LEN)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
