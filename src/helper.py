from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf" ,loader_cls=PyPDFLoader)       # Load all PDF files from a directory
    document=loader.load()                                                     # here .load() is a method which will return a list of dictionaries, each dictionary contains the text of a pdf file
    return document


def chunk_splitter(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500 , chunk_overlap=50)    # Split the text into chunks of 500 characters with an overlap of 50 characters
    text_chunks = splitter.split_documents(extracted_data)                          # here .split() is a method which will return a list of dictionaries, each dictionary contains the text of a chunk
    return text_chunks

def load_embedding_model():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding

