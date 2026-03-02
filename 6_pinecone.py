##### PINECONE #####
# a fully managed vector database solution for building and scaling generative AI apps
# -> uses embeddings to store and retrieve data based on their semantic similarity

### INDEXES ###
# -> one of the core componenets of the pinecone infrastructure
# -> used to store vectors + serve queries and other vector manipulationsn over the vectors it contains
# -> each vector is contained in a record in an index
# -> each record stores additional metadata that can be used in querying

## TYPES
# 1. Pod-Based Indexes
# -> choose one or more pre-configured units of hardware called pods to create the index
# -> each pod type has its own amounts of storage, query latency and query throughput

# 2. Serverless
# -> don't require managing resources
# -> scale automatically according to usage
# -> run on cloud like AWS and vectors are store in blob storage on that platform
# -> easier to use + lower cost

## CREATING A SERVERLESS INDEX
import itertools
from pinecone import Pinecone, ServerlessSpec

# client is configured with the env to work with the API
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) #OR pc = Pinecone(api_key="your_api_key")

# ServerlessSpec is used for creating a serverless index
# to create the index, call .creat_index() method on the client
pc.create_index(
    name = 'data-index',    # name of the index
    dimension = 1536,       # dimension of the vectors it will store
    spec = ServerlesSpec(
        cloud = 'aws',
        region = 'us-east-1'
    )
)

# to verify that the index was created, use .list_indexes() method on client
pc.list_indexes()
############################################################################################################

### MANAGING INDEXES ###

## CONNECTING TO THE INDEX
# use the .Index() method on the client to connect to the index just created
# will 404 error if try to connect to an index that was not created   
index = pc.Index('data-index')

## INDEX STATISTICS
# use .describe_index_stats() method to get the:
# -> no. of vectors in the index
# -> proportion of the index that is full - fullness
# -> dimensionality of the index
# -> also an empty dict to hold namespaces
index.describe_index_stats()

# NAMESPACES - containers for partitioning indexes into smaller units
# -> to keep distinct datasets or data versions separate or to partition groups in the parent dataset

## ORGANIZATIONS
# -> provide control over access and billing to different Pinecone projects
# hierarchy = organization -> project -> index -> namespace -> individual records
# -> ROLES: 1. Owner    2. User
# ->> Owner: - permissions across entire org - manage billing, users, all projects
# ->> User: - restricted org-level permissions - invited to specific projects by the owner

## DELETING INDEXES
# use .delete_index() method
# -> will also delete all of the records it contains
pc.delete_index('data-index')
#####################################################################################################

### INGESTING VECTORS #####
# vectors can be a list of dictionaries containing a unique ID and vector values

## BEFORE INGESTION
# -> 1. Pinecone requires vectors in the following format
vectors =[
    {
        "id":"0",
        "values":"[0.0255255475892155, ..., 0.01489631177]",
        "metadata": {"genre":"productivity", "year":"2020"}
    },
    ...,
    {
        "id":"9"
        "valuses":"[0.020712569484512, ..., 0.00654864125]"
    }
]

# -> 2. make sure they have same dimensionality as index
# check dimensionality by creating a list comprehension that checks if the length of the list located under 'values' key is 1536 for EACH vector
vector_dims = [len(vector['values']) == 1536 for vector in vectors]
# to check if TRUE for all vectors call all method on the list
all(vector_dims)

## UPSERT METHOD
# if vector id is already present in index, it gets updated otherwise it is inserted as a new entry
index.upssert(
    vectors=vectors
)

# to check if vectors were successfully inserted 
index.describe_index_stats()

## INGESTING VECTORS WITH METADATA
# some vectors also have metadata
# useful because then we can just filter by metadata to only search over the most relevant records
# same syntax
index.upsert(vectors = vectors)
##################################################################################################################
##################################################################################################################

### RETRIEVING VECTORS ###
## ACCESSING VECTORS
# two methods: 1. Fetching 2. Querying
# 1. Fetching
# ->> retrieve vectors based on their IDs
# ->> normally done to explore and verify particular records from the index
# ->> like searching for vectors based on their IDs
# ->> call the fetch method by passing a list of record ids to return

index.fetch(
    ids=["0", '1'],
    namespace='namespace1'
)
# results show the vectors listed under vectors + read_units under usage
# -> READ_UNITS: measure of resources consumed during read operations like fetching, querying and listing

# Initialize the Pinecone client with your API key
pc = Pinecone(api_key="<PINECONE_TOKEN>")

index = pc.Index('datacamp-index')
ids = ['2', '5', '8']

# Fetch vectors from the connected Pinecone index
fetched_vectors = index.fetch(ids=ids)

# Extract the metadata from each result in fetched_vectors
metadatas = [fetched_vectors['vectors'][id]['metadata'] for id in ids]
print(metadatas)

# 2. Querying
# ->> retrieve similar vectors to an input vector
# ->> like searching for vectors similar to the ones we have
# use .query() method, pass it a vector to query with + top_k argument

index.query(
    vector=[-0.236987411, ...],
    top_k=3,
    include_values=True         # to see the vector values in the output
)

# results in the top_k matches + a score for every id which is the measure of similarity
# -> READ_UNITS - dependent on 1. no. of records, 2. size of records which is the vector dimensionality + amount of metadata

## DISTANCE METRICS
# Cosine, Euclidean, Dot
# can require a bit of experimentation

pc.create_index(
    name="datacamp-index",
    dimension=1536,
    metric='dotproduct',
    specs=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)
##################################################################################

### METADATA FILTERING ###
# use filter method in query
# takes a dictionary where each key is the metadata to filter and their values specify how to filter them

index.query(
    vector=[-0.025697, ...],
    filter={
        "genre":{"$eq":"documentary"},      # $eq is the equality operator
        "year":2019
    },
    top_k=1,
    include_metadatas=True      # to include the metadat in the result
)

index = pc.Index('datacamp-index')

# Retrieve the MOST similar vector with genre and year filters
query_result = index.query(
    vector=vector,
    top_k=1,
    filter={
        "genre":{"$eq":"thriller"},
        "year":{"$lt":2018}
    }
)
print(query_result)
######################################################################################

### UPDATING AND DELETING VECTORS
## UPDATE

index.update(
    id="1",     # the ID of the vector we want to update
    values=[0.3644646, ...],    # new values we want to update 
    set_metadata={"genre":"comedy", "rating": 5}    # update the metadata
)

## DELETE

index.delete(
    id=ids,
    filter={
        "genre":{"$eq":"action"}
    }
)

# to delete all records from a namespace
index.delete(delete_all=True, namespace='namespace1')
###########################################################################################
###########################################################################################

### PERFORMANCE TUNING ###

## 1. BATCHING UPSERTS
# -> limitations: 1. rate of requests   2. Size of requests

# -> 1. create a chunk function that breaks the vectors into small chunks containing batch_size vectors
# ->>> convert the list, which is an iterable into an iterator using the iter() function
# ->>> iterables are strings, lists, dictionaries where elements can be extracted one at a time
# ->>> iterators only produce elements on demand and return data in streams

# -> 2. create the first chunk by calling itertools.islice() on the iterator + specify batch_size parameter

# -> 3. use a while loop to iterate until no chunks left
# ->>> yield the current chunk which can then be upserted
# ->>> update the chunk with the next batch of vectors using the same code that we used to define the first chunk 

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

## SEQUENTIAL BATCHING
# splitting requests into chunks and sending them one-by-one to the index
# -> 1. initialize the client and connect to the index
pc.Pinecone(api_key="API_KEY")
index=pc.Index('data-index')

# -> 2. define the chunking function
# -> 3. upsert all the chunks sequentially

for chunk in chunks(vectors):
    index.upsert(vectors=chunk)

# -> solves issue of size and rate.
# -> tradeoff with speed -very slow!
# -> use parallel batching

## PARALLEL BATCHING
# split requests and send them in parallel
# -> 1. initialize pinecone client with the pool_threads parameter -> sets max no. of pool requests
# -> upsertion is done async so that requests can be sent independently
# -> call the .get() method on each async result to wait for and retrieve the responses

pc = Pinecone(api_key="API_KEY", pool_threads=30)
with pc.Index('data-index', pool_threads=30) as index:      # connect to index
    async_results = [index.upsert(vectors=chunk, async_req=True)
        for chunk in chunks(vectors,  batch_size=100)]
    
    [async_result.get() for async_result in async_results]
###################################################################################

### MULTITENANCY AND NAMESPACES ###

## 2. MULTITENANCY
# a software archi where a system can serve multiple groups of users called tenants in isolation
# like storing records in different namespaces in the same index
# can help reduce query latency

# -> STRATEGIES:
# 1. Namespaces - enables targeted queries by minimizing the number of scanned records
# 2. Metadata Filtering - attach metadata to records and use queries to search over records with specific metadata
# ->>> enables querying across tenants
# 3. Separate Indexes for Each Segment - provides detailed resources to each tenant
# ->>> provides max isolation

#########################################################################################

### SEMANTIC SEARCH ENGINES ###
# 1. embed and ingest docs into Pinecone index
# 2. embed a user query
# 3. query the index with the embedded user query

## Setting up OpenAI and Pinecone for Semantic Search
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 1. Initialize clients
client = OpenAI(api_key='KEY')
pc = Pinecone(api_key='KEY')

# 2. Create Index
pc.create_index(
    name="semantic-search",
    dimension =1536,            # same as the dimensions for the outputs of the OpenAI model
    spec = ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# 3. Connect to the Index
pc.Index("semantic-search")

# 4. Prepare docs for ingesting
import pandas as pd
import numpy as np
from uuid import uuid4      # to generate unique record IDs before upserting

df = pd.read_csv('squad_dataset.csv')   # using SQuAD dataset comprising of wikipedia articles to semantically search through
# dataset contains columns for article id, text and title
# | id  |   text    |   title   |

# 5. Ingesting Documents
# setting a batch_limit of 100 docs per request
batch_limit=100

# using numpy's array split func we iterate through the docs in batches
# with each batch we extract the metadata into a dictionary using a list comprehension
# assigns row values to dictionary keys
for batch in np.array_split(df, len(df)/batch_limit):
    metadatas = [{
        "text_id":row['id'],
        "text": row['text'],
        "title":row['title']
    }
    for _, row in batch.iterrows()]

    texts = batch['text'].tolist()

    # also extract the article text tp embed and create unique IDs for each row using the uuid4() func
    ids = [str(uuid4()) for _ in range(len(texts))]

    # create a request to OpenAI's API to embed these texts, extracting the raw embeddings from the response
    response = client.embeddings.create(
        input=texts,
        model = "text-embedding-3-small"
    )

    embeds = [np.array(x.embedding) for x in response.data]

    # zip together the ids, vectors, and metadatas and upsert them to a new namespace
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace="squad_dataset")

# 6. Semantic Search with Pinecone
query = "To whom did the Virgin Mary appear in 1858 in Lourdes France?"

# encode the query using the same embedding model again
query_response = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)

query_emb = query_response.data[0].embedding

retrieved_docs = index.query(
    vector=query_emb,
    top_k=3, 
    namespace=namespace,
    include_metadata=True
)

# loop through the records and get the similarity scores and texts
for result in retrieved_docs['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")
#####################################################################################################

### RAG CHATBOT ###
## RAG
# -> sys archi designed to improve question answering models by providing them with additional info
# 1. user query embedded
# 2. retrieve similar docs from the database
# 3. docs are added to the model prompt so that the model has extra context to inform its response

# 1. Set up Libraries
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from uuid import uuid4

# 2. Initialize clients
client = OpenAI(api_key="KEY")
pc = Pinecone(api_key="KEY")

# 3. Create Index
pc.create_index(
    name="semantic-search",
    dimension =1536,            # same as the dimensions for the outputs of the OpenAI model
    spec = ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# 4. Connect to the Index
pc.Index("semantic-search")

# 5. Prepare Dataset
youtube_df = pd.read_csv('youtube_rag_data.csv')

#   |   id  |   blob    |   channel_id  |   end |   published   |   start   |   text    |   title   |   url |
#   |-------|-----------|---------------|-------|---------------|-----------|-----------|-----------|-------|
#   |   int |   dict    |   str         |   int |   datetime    |   int     |   str     |   str     |   str |

# -> can save all this info as metadata

# 6. Ingest Docs
# with each batch extract metadata using a list comprehension
# creates a metadata dictionary for each row in the batch and assigns the row values to dict keys

batch_limit=100

for batch in np.array_split(youtube_df, len(youtube_df)/batch_limit):
    metadatas = [
        {
            "text_id":row['id'],
            "text":row['title'],
            "url":row['url'],
            "published":row['published']
        }
        for _, row in batch.iterrows()
    ]

    # extract article texts to embed + create unique ids
    texts = batch['text'].tolist()
    ids = [str(uuid4()) for _ in range(len(texts))]

    # request to OpenAI to embed these texts, extracting raw embeddings from the response
    response = client.embeddings.create(
        inpts=texts,
        model="text-embedding-3-small"
    )
    embeds = [np.array(x.embedding) for x in response.data]

    # zip together the ids, vectors, and metadatas and upsert them to a new namespace
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace="youtube_rag_dataset")

# 7. retrieval Function to Retrieve Docs from Pinecone Index Based on a Query
def retrieve(query, top_k, namespace, emb_model):
    query_response = client.embeddings.create(
        input=query,
        model=emb_model
    )

    query_emb = query_response.data[0].embedding        # extract embeddings from API response

    retrieved_docs = []
    sources = []
    docs = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace='youtube_rag_dataset',
        include_metadata=True
    )

    # for each retrieved doc, extract the text and source
    for doc in docs['matches']:
        retrieved_docs.append(doc['metadata']['text'])
        sources.appens((doc['metadata']['title'], doc['metadata']['url']))

    return retrieved_docs, sources

# 8. Retrieval Output
query="How to build next-level Q&A with OpenAI"
documents, sources = retrieve(query, top_k=3, namespace='youtube_rag_dataset', emb_model="text-embedding-3-small")

# 9. Create a Contextual Prompt for the model
def prompt_with_context_builder(query, docs):
    delim = '\n\n---\n\n'
    prompt_start = 'Answer the question based in the context below.\n\nContext:\n'
    prompt_end = f'\n\nQuestion:{query}\nAnswer:'

    prompt = prompt_start + delim.join(docs) + prompt_end

    return prompt

# 10. Use it
query="How to build next-level Q&A with OpenAI"
context_prompts=prompt_with_context_builder(query, documents)

# 11. Question-Answering Function
# user prompt is the output from prompt_with_context_builder() function

def question_answering(prompt, sources, chat_model):
    sys_prompt = "You are a helpful assistant that always anwsers questions."
    res = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role":"system", "content":sys_prompt},
            {"role":"user", "content":prompt}
        ],
        temperature =0
    )

    answer = res.choices[0].message.content.strip()
    answer += "\n\n\Sources:"

    for source in sources:
        answer += "\n" + source[0] + ": " + source[1]
    
    return answer

# 11. Use it
query="How to build next-level Q&A with OpenAI"
answer = question_answering(context_prompts, sources, chat_model='gpt-4o-mini')
print(answer)