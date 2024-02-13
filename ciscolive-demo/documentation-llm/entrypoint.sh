#!/bin/bash
pip install -r /app/requirements.txt

# Get modelfile
wget https://huggingface.co/TheBloke/rocket-3B-GGUF/resolve/main/rocket-3b.Q4_K_M.gguf -O /app/backend/llm/model.gguf

# Navigate to training directory and run scripts
cd /app/training
python pdf.py
python db-embeddings.py

# Navigate to the inference API directory and start the API
cd /app/backend/inference
python main.py

