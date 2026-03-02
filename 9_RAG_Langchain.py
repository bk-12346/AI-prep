##### RAG #####

### STANDARD RAG WORKFLOW ###
# embedding user query -> to retrieve relevant docs -> incorporate them into the model's prompt
# RAG provides extra context for more informed LLM responses

### Preparing Data for Retrieval ###
# 1. load the docs to build the knowledge base
# 2. split it into chunks to be processed
# 3. create numerical representations from text called embeddings
# 4. embeddings/vectors are stored in a vector database for future retrieval

## DOCUMENT LOADERS ##
# loaders handle different file types including standard formats like CSV and PDFs + specialized formats supported by 3rd-party providers like Amazon S3 files etc

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredHTMLLoader

csv_loader = CSVLoader(file_path='path/to/file.csv')

# to load these doc to memory we use .load() method
docs = csv_loader.load()
print(docs)

# each doc has .page_content and .metadata attributes that can be used to access the respective data
print(docs[0].page_content)
print(docs[0].metadata)

# PDF Loader
pdf_loader = PyPDFLoader(file_path='path/to/file.pdf')
docs = pdf_loader.load()
print(docs[0].page_content)
print(docs[0].metadata)

# HTML Loader
html_loader = UnstructuredHTMLLoader(file_path='path/to/file.html')
docs = html_loader.load()
print(docs[0].page_content)
print(docs[0].metadata)

## SPLITTING/CHUNKING ##
# chunks contain sufficient context that the LLM can use to generate a response
# too big: retrieval will be too slow; LLM may not be able to extract relevant context from chunk to respond
# chunk_size parameter used to control the size
# chunk_overlap parameter used to control the loss of info between the boundaries of the chunks

# CHARACTER_TEXT_SPLITTER #

from langchainn_text_splitters import CharacterTextSplitter

text = """Hello. This is Machine Learning going on and on. Thankyou"""

# 1. instantiate a splitter using the class
# 2. specify separator to split on
# 3. chunk_size to control the size of the chunks
# 4. chunk_overlap to control the loss of info between the boundaries of the chunks
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size=100,
    chunk_overlap=10
)

chunks = text_splitter.split_text(text)
print(chunks)
print([len(chunk) for chunk in chunks])

# RECURSIVE_CHARACTER_TEXT_SPLITTER #

# takes a list of separators to split on and works through the list from left to right, splitting the document using each separator
# often preserves more context

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

print(chunks)
print([len(chunk) for chunk in chunks])

# SPLITTING DOCUMENTS #

# just use .split_documents() instead of .split_text()

from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(file_path='path/to/file.pdf')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)
print(chunks)

# each doc has .page_content and .metadata attributes for extracting the respective information
print(chunks[0].page_content)
print(chunks[0].metadata)

# calculating the number of characters in each chunk
print([len(chunk.page_content) for chunk in chunks])

## EMBEDDINGS ##
# aim to catch meaning of the text
# these number's catch the text's position in a high-dimensional or vector space

# vector stores are databases specifically designed to store and retrieve this high-dimensional vector data
# similar documents are located closer together in the vector space

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# initialize the model
embedding_model = OpenAIEmbeddings(
    api_key = openai_api_key,
    model="text-embedding-3-small"
)

# to embed and store  the chunks in one operation, we call the .from_documents() method on the Chroma class, passing the chunks and model
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)
