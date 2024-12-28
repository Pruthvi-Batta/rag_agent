# Variables
CONDA_PREFIX = $(HOME)/miniconda3
CONDA_BIN = $(CONDA_PREFIX)/bin/conda
ENV_NAME = rag_agent_env
ENV_FILE = env.yaml

# Default target
.PHONY: all
all: setup-env

# Check if Miniconda is installed, and install it if necessary
.PHONY: install-miniconda
install-miniconda:
	@echo "Checking for Miniconda installation..."
	@if [ ! -d "$(CONDA_PREFIX)" ]; then \
		echo "Miniconda not found. Installing..."; \
		curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3.sh; \
		bash Miniconda3.sh -b -p $(CONDA_PREFIX); \
		rm Miniconda3.sh; \
		echo "Miniconda installed successfully."; \
	else \
		echo "Miniconda is already installed."; \
	fi

# Create the Conda environment if it doesn't exist
.PHONY: create-env
create-env: install-miniconda
	@echo "Checking for Conda environment '$(ENV_NAME)'..."
	@if ! $(CONDA_BIN) env list | grep -q "^$(ENV_NAME) "; then \
		echo "Environment '$(ENV_NAME)' not found. Creating it..."; \
		$(CONDA_BIN) env create -n $(ENV_NAME) -f $(ENV_FILE) -y; \
		echo "Environment '$(ENV_NAME)' created successfully."; \
	else \
		echo "Environment '$(ENV_NAME)' already exists."; \
	fi

# Activate the Conda environment
.PHONY: activate-env
activate-env:
	@echo "To activate the environment, run: source $(CONDA_PREFIX)/bin/activate $(ENV_NAME)"

# Full setup process
.PHONY: setup-env
setup-env: create-env activate-env
	@echo "Setup completed. Use the command above to activate your environment."

# Activate the Conda environment
.PHONY: run
run:
	cd src && streamlit run UI_screen.py