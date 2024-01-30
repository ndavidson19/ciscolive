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

Choosing a license can be difficult and depend on your goals for your code, other licensed code on which your code depends, your business objectives, etc.   This template does not intend to provide legal advise. You should seek legal counsel for that. However, in general, less restrictive licenses make your code easier for others to use.

> Cisco employees can find licensing options and guidance [here](https://wwwin-github.cisco.com/DevNet/DevNet-Code-Exchange/blob/master/GitHubUsage.md#licensing-guidance).

Once you have determined which license is appropriate, GitHub provides functionality that makes it easy to add a LICENSE file to a GitHub repo, either when creating a new repo or by adding to an existing repo.

When creating a repo through the GitHub UI, you can click on *Add a license* and select from a set of [OSI approved open source licenses](https://opensource.org/licenses). See [detailed instructions](https://help.github.com/articles/licensing-a-repository/#applying-a-license-to-a-repository-with-an-existing-license).

Once a repo has been created, you can easily add a LICENSE file through the GitHub UI at any time. Simply select *Create New File*, type *LICENSE* into the filename box, and you will be given the option to select from a set of common open source licenses. See [detailed instructions](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository).

Once you have created the LICENSE file, be sure to update/replace any templated fields with appropriate information, including the Copyright. For example, the [3-Clause BSD license template](https://opensource.org/licenses/BSD-3-Clause) has the following copyright notice:

`Copyright (c) <YEAR>, <COPYRIGHT HOLDER>`

See the [LICENSE](./LICENSE) for this template repo as an example.

Once your LICENSE file exists, you can delete this section of the README, or replace the instructions in this section with a statement of which license you selected and a link to your license file, e.g.

This code is licensed under the BSD 3-Clause License. See [LICENSE](./LICENSE) for details.

Some licenses, such as Apache 2.0 and GPL v3, do not include a copyright notice in the [LICENSE](./LICENSE) itself. In such cases, a NOTICE file is a common place to include a copyright notice. For a very simple example, see [NOTICE](./NOTICE). 

In the event you make use of 3rd party code, it is required by some licenses, and a good practice in all cases, to provide attribution for all such 3rd party code in your NOTICE file. For a great example, see [https://github.com/cisco/ChezScheme/blob/main/NOTICE](https://github.com/cisco/ChezScheme/blob/main/NOTICE).  

[More about open-source licenses](https://github.com/CiscoDevNet/code-exchange-repo-template/blob/main/manual-sample-repo/open-source_license_guide.md)

----


