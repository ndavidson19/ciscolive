# Cisco Documentation RAG LLM

# LLM Documentation Demo for Nexus Dashboard

A demo application showcasing how to run a local LLM on your own hardware. Includes samples that leverage open-source libraries (llama.cpp) and models (llama), as well as documentation from Nexus Dashboard.

## Installation

First go into project directory
```
cd documentation-llm
```

We need to select a model to use. For this demo we will use the base LLaMa-2 model. You are encouraged to use Huggingface Leaderboards to find the best model for your usecase. 

Model link: https://huggingface.co/TheBloke/Llama-2-7B-GGML/blob/main/llama-2-7b.ggmlv3.q4_0.bin 

Once downloaded, make a llm directory in /documentation-llm/backend and place in the downloaded modelfile.

```
cd backend
mkdir llm
```

### Install requirements
It is recommended to create a virtual-env before installing dependencies. Or use a dependency manager such as anaconda.
Ex.

```
python3 -m venv venv_name
source venv_name/bin/activate
```

```
pip install -r requirements.txt
```

### Pull docker image for the postgres vector database
This image has the pgvector extension for postgres that allows for fast vector embeddings and lookups.
Make sure you have docker installed. https://docs.docker.com/engine/install/

```
docker pull ankane/pgvector
docker run -p 5432:5432 -e POSTGRES_PASSWORD=secret -e POSTGRES_USER=postgres ankane/pgvector
```

## Usage (required)

### Training Pipeline 
This module contains four scripts that parse pdfs to text, clean the text, create vectorized embeddings, and insert the embeddings into the postgres database. 
---
```
cd training
python pdf.py
python clean.py
python embeddings.py
python db-embeddings.py
```
Once inserted make sure your docker image and daemon are running in order for the retrieval process to work.

### Start the inference API's
This module contains the API's neccessary in order to combine the user prompt with the retrieved information from the vector DB.

---
```
cd backend/inference
python main.py
```

### Load UI (html)
To load the UI you just need to open the index.html file that lives in the /docuementation/ui directory. 
You should be all set to start asking questions!

## Licensing info

A license is required for others to be able to use your code. An open source license is more than just a usage license, it is license to contribute and collaborate on code. Open sourcing code and contributing it to [Code Exchange](https://developer.cisco.com/codeexchange/) requires a commitment to maintain the code and help the community use and contribute to the code. 
[More about open-source licenses](https://github.com/CiscoDevNet/code-exchange-repo-template/blob/main/manual-sample-repo/open-source_license_guide.md)

----


