#!/bin/bash
module load python
module load openssl
# Create the virtual environment
python3 -m venv .ea

# Activate the virtual environment
source .ea/bin/activate

# Install requirements
pip3 install --no-cache-dir -r requirements.txt