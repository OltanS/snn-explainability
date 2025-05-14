#!/bin/bash

# Create the virtual environment
python3 -m venv .ea

# Activate the virtual environment
source .ea/bin/activate

# Install requirements
pip install --no-cache-dir -r requirements.txt