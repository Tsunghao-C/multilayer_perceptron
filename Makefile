# Makefile for building the project

SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)
PYTHON_VERSION := 3.12

.PHONY: all clean install test setup help

all: help

# Help target
help:
	@echo "Makefile for building the project"
	@echo "Available targets:"
	@echo "  setup   - Set up the development environment"
	@echo "  install - Install the project dependencies"
	@echo "  test    - Run the tests"
	@echo "  clean   - Clean up build artifacts"

# Setup development environment
setup:
	@echo "Setting up the development environment..."
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv package..."; \
		brew install uv; \
	fi
	@if ! command -v pyenv &> /dev/null; then \
		echo "Installing pyenv..."; \
		brew install pyenv; \
	fi
	@if ! pyenv versions --bare | grep -q $(PYTHON_VERSION); then \
		echo "Installing Python $(PYTHON_VERSION)..."; \
		pyenv install $(PYTHON_VERSION); \
	fi
	@if ! command -v pre-commit &> /dev/null; then \
		echo "Installing pre-commit..."; \
		brew install pre-commit; \
	fi
	@echo "Initiating pre-commit hooks..."
	@pre-commit install
	@if ! command -v ruff &> /dev/null; then \
		echo "Installing ruff..."; \
		brew install ruff; \
	fi
	@echo "Installing git hooks..."
	@chmod +x scripts/setup-git-hooks.sh
	@./scripts/setup-git-hooks.sh
	@echo "Development environment set up successfully."

# Install project dependencies
install:
	@echo "Installing dependencies for all projects with pyproject.toml..."
	@find . -type f -name pyproject.toml | while read file; do \
		dir=$$(dirname $$file); \
		echo "🔧 Processing: $$dir"; \
		cd $$dir && \
		pyenv local $(PYTHON_VERSION) && \
		uv sync; \
		cd - > /dev/null; \
	done
	@echo "Dependencies installed successfully."

# Run tests
test:
	@set -e; \
	for pkg in multilayer_perceptron; do \
		echo "Running tests for package: $$pkg"; \
		cd $$pkg && \
		pyenv local $(PYTHON_VERSION) && \
		uv venv --clear && \
		uv run pytest --json-report --json-report-file=report.json || true; \
		cd - > /dev/null; \
	done
	@echo "Merging JSON reports..."
	@python scripts/merge_json_reports.py
	@echo "Merged JSON reports saved to merged_report.json"

%:
	@echo "No target specified. Use 'make help' to see available targets."