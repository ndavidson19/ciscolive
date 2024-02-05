# Cisco Documentation RAG LLM

# LLM Documentation Demo for Nexus Dashboard

A demo application showcasing how to run a local LLM on your own hardware. Includes samples that leverage open-source libraries (llama.cpp) and models (llama), as well as documentation from Nexus Dashboard.

## Installation

First clone the project and navigate into project directory
```
git clone https://github.com/ndavidson19/ciscolive.git
cd ciscolive/ciscolive-demo/documentation-llm
```

Next you must download the modelfile. Huggingface has so many models to choose from and all have very elaborate names. We will be choosing a finetuned version of Mistral v2 7B with a very funny name.  
[https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF/blob/main/dolphin-2.6-mistral-7b-dpo-laser.Q4_K_M.gguf)

Details about the model training can be found at: https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser

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


## Licensing info

A license is required for others to be able to use your code. An open source license is more than just a usage license, it is license to contribute and collaborate on code. Open sourcing code and contributing it to [Code Exchange](https://developer.cisco.com/codeexchange/) requires a commitment to maintain the code and help the community use and contribute to the code. 
[More about open-source licenses](https://github.com/CiscoDevNet/code-exchange-repo-template/blob/main/manual-sample-repo/open-source_license_guide.md)

----


