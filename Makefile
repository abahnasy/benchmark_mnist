# Makefile
SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."


# Styling
.PHONY: style  # execute whether there is a file named style or not
style:
	black .
	flake8
	python3 -m isort .

# Environment, TODO: replace with conda
.PHONY: venv  # execute whether there is a file named venv or not
#.ONESHELL: use this or chain commands with &&
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage

# Set variable
MESSAGE := "hello world"

# Use variable
greeting:
	@echo ${MESSAGE}
