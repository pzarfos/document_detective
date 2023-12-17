# Document Detective

This script answers questions about files in a local directory.
It can be used to create a GPT chat engine for your private data.
Written in Python, this script uses Ollama and the LangChain packages to generate responses.

## Features

The following features are supported:

* Apple Silicon GPU support
* Scan multiple documents in a directory
* Ability to answer questions about the contents of the documents
* Specify which Ollama model to use
* Specify the temperature of the model

## Installation

### Step 1. Install Ollama and a model

* Download [Ollama](https://ollama.ai/) from the Ollama site
* Download a model or two, like "llama2:13b"
    * Note: The "llama2:13b" requires 16 GB RAM

### Step 2. Clone this repository

```sh
git clone https://github.com/pzarfos/document_detective
cd document_detective
```

### Step 3. Install the required Python packages

* The dependencies are for Apple Silicon, so the tensorflow library will run on the GPU
* If you are running on Linux, adjust the 'tensorflow' packages accordingly

```sh
# example: install in a Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

To run the script, run it with the following command.  It will parse your documents and present you with a prompt:

```sh
python document_detective.py -d <directory>

# Type `exit` to end the prompt
```

You can ask a single question on the command line, for example:

```sh
python document_detective.py -d <directory> -q "How do I create a new plugin for our system?"
```

You can specify different Ollama models and temperatures if you like, for example:

```sh
python document_detective.py -d <directory> -m mistral -t 0.5
```

## Attribution

* This script was copied from the Ollama examples directory, and modified to support text files, command line arguments, and Apple Silicon GPUs
* Source: <https://github.com/jmorganca/ollama/tree/main/examples/langchain-python-rag-document>

## License

This project is licensed under the MIT License - see the <LICENSE> file for details.
