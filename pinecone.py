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