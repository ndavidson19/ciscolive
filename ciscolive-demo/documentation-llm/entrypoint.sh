#!/bin/bash
pip install -r /app/requirements.txt

# Navigate to training directory and run scripts
cd /app/training
python pdf.py
python db-embeddings.py

# Navigate to the inference API directory and start the API
cd /app/backend/inference
python main.py

