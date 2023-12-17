"""
Document Detective

Use Ollama and LangChain to answer questions about files in a specified directory.

Based on this script:
https://github.com/jmorganca/ollama/tree/main/examples/langchain-python-rag-document
"""

import argparse
import os
import sys
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class SuppressStdout:
    """
    Suppress stdout and stderr
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# Prompt template
def get_template():
    return """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use three sentences maximum and keep the answer as concise as possible. 
        {context}
        Question: {question}
        Helpful Answer:
        """


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        default="docs",
        help="Directory containing documents",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="llama2:13b",
        help="Name of Ollama model (list available models using 'ollama list')",
        required=False,
    )
    parser.add_argument(
        "-q",
        "--query",
        help="The question you want to ask",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--temperature",
        default=0.2,
        type=float,
        help="Model temperature (0.0-1.0, a higher number has more hallucinations)",
        required=False,
    )

    args = parser.parse_args()
    return [args.directory, args.model, args.query, args.temperature]


def main():
    [directory, model, initial_query, temperature] = get_args()

    # Load the files in the specified directory and split it into chunks
    loader = DirectoryLoader(directory, glob="**/*")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    with SuppressStdout():
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=GPT4AllEmbeddings()
        )

    # Loop while answering questions
    while True:
        if initial_query:
            # Question was passed in as an argument
            query = initial_query
        else:
            # Prompt the user for a question
            query = input("\nQuery: ").strip()
            if query.lower() == "exit":
                break
            if query == "":
                continue

        qa_chain_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=get_template(),
        )

        llm = Ollama(
            model=model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=temperature,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )

        _ = qa_chain({"query": query})
        print("\n")

        if initial_query:
            # Question was passed in as an argument, so exit the loop
            break


if __name__ == "__main__":
    main()
