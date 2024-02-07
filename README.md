# Cisco Documentation RAG LLM

# LLM Documentation Demo for Nexus Dashboard

A demo application showcasing how to run a local LLM on your own hardware. Includes samples that leverage open-source libraries (llama.cpp) and models (llama), as well as documentation from Nexus Dashboard.

## Installation

First clone the project and navigate into project directory
```
git clone https://github.com/ndavidson19/ciscolive.git
cd ciscolive/ciscolive-demo/documentation-llm
```

Next you must download the modelfile. Huggingface has so many models to choose from and all have very elaborate names. We will be choosing a DPO finetuned version of StableLM. This small 3B model punches above its weight when it comes to RAG applications. 

[Rocket 3B](https://huggingface.co/TheBloke/rocket-3B-GGUF/blob/main/rocket-3b.Q4_K_M.gguf)

Details about the model training can be found at: https://huggingface.co/pansophic/rocket-3B

Next create a directory called llm in the backend folder
```
cd /cisco-live/documentation-llm/backend
mkdir llm
```

Then move the modelfile to the correct directory ciscolive/ciscolive-demo/documentation-llm/backend/llm/dolphin-2.6-mistral-7b-dpo-laser.Q4_K_M.gguf
```
cd /cisco-live/documentation-llm/backend
mkdir llm
```

## Usage 

This entire application has been dockerized and can be run with just
```
docker-compose up --build
```
This starts three different services.
1. The Vector Datastore (pgvector)
   - This pulls a postgres image from ankane/pgvector that installs the correct extensions for allow vectors within postgres.
2. The Flask serving APIs and VectorDB insertion
   - This service starts a flask API endpoint route (/get_message) on port :5000 that allows for a user to send queries to the LLM being served using LlamaCPP (https://github.com/abetlen/llama-cpp-python) using script at /backend/main.py
   - This service also parses the pdf living in /training/pdfs/ using /training/pdf.py and then inserts it into the database using /training/db-embeddings.py
3. The UI service
   - Uses nginx to start a basic webserver for the basic index.html file

Note: This is a very simplistic scaled down version of our full architecture we are running in production and should be treated as a starting point. Look into the llama-cpp-python OpenAI compatible webserver if you are going to be creating your own application.


## Manual Usage

It is recommended to create a virtual-env before installing dependencies. Or use a dependency manager such as anaconda.
Ex.

```
python3 -m venv venv_name
source venv_name/bin/activate
```

```
pip install -r requirements.txt
```

Next you must download the modelfile. We are using LlaMa-2 chat quantized to 4-bit by TheBloke.
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf

Next move the modelfile to the correct directory /cisco-live/documentation-llm/backend/llm/llama-2-7b-chat.Q4_K_M.gguf
```
cd /cisco-live/documentation-llm/backend
mkdir llm
```

### Training Pipeline 
This module contains two scripts that parse pdfs to text, clean the text, create vectorized embeddings, and insert the embeddings into the postgres database. 
---
```
cd training
python pdf.py
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
Run the below command in the root directory of the project.
```
python -m http.server
```
Navigate to http://localhost:8000/ in your browser.
To load the UI you just need to open the index.html file that lives in the cisco-live/documentation-llm/ui directory. 

You should be all set to start asking questions!


## Licensing info

A license is required for others to be able to use your code. An open source license is more than just a usage license, it is license to contribute and collaborate on code. Open sourcing code and contributing it to [Code Exchange](https://developer.cisco.com/codeexchange/) requires a commitment to maintain the code and help the community use and contribute to the code. 
[More about open-source licenses](https://github.com/CiscoDevNet/code-exchange-repo-template/blob/main/manual-sample-repo/open-source_license_guide.md)

----


