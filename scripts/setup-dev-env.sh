#!/bin/bash

# Check the operating system.
if [[ "$OSTYPE" != "linux-gnu"* && "$OSTYPE" != "darwin"* ]]; then
    echo "This script is only compatible with Linux and macOS systems."
    exit 1
fi

# Check if virtualenv is installed, if not, install it.
if ! command -v virtualenv &>/dev/null; then
    echo "Installing virtualenv!"
    pip install virtualenv
fi

# Create a virtual environment named 'venv' if it doesn't exist.
if [ ! -d "venv" ]; then
    echo "Creating virtual environment!"
    python -m venv venv
fi

# Activate the virtual environment.
echo "Activating virtual environment!"
source venv/bin/activate

# Install the required dependencies, every file that starts with 'requirements'.
echo "Installing python package!"
pip install --upgrade pip
pip install -r <(cat requirements*.txt)

# Deactivate the virtual environment.
echo "Deactivating virtual environment!"
deactivate
