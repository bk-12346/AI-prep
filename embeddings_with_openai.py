##### EMBEDDINGS #####
# fundamental concept in NLP
# numerical representation of text
# embeddings model map text onto a multi-dimensional space, or vector space
# numbers outputted by the model are the text's location in space
# similar words can appear together

## Why Useful?
# can capture semantic meaning of text
# full context and intent of the word is captured

## Enable 
# 1. Creation of semantic search engines
# use embeddings to understand the intent and context of the input before providing an output
# search query -> passed to an embedding model -> generate numbers -> mapped to vector space -> embedded results closest to it are returned
# 2. Recommendation Systems
# job descriptions can be displayed according to similar searches
# 3. Classification tasks

### OpenAI API has an Embeddings endpoint
# specify model argument
# text to embed is passed to the input argument
# use model.dump() to convert the response to a dictionary

from ssl import ALERT_DESCRIPTION_PROTOCOL_VERSION
from openai import OpenAI

client = OpenAI(api_key="<YOUR_KEY>")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="embeddings ...."
)

response_dict = response.model_dump()
print(response_dict)    # prints the whole response

print(response_dict['data'][0]['embedding'])    # print the full list of all the numbers representing the text
print(response_dict['usage']['total_tokens'])   # to get total tokens
###################################################################################################################################################

### Vetor Space ###
# uaing example of a dataset of articles stored in a list of dictionaries
# each article has a headline stored under the headline key and a topic stored under the topic key

articles = [
    {"headline":"economic growth continues ...", "topic":"Business"},
    {}
]

# we will embed each headline's text and add them back to the headlines dictionary, stored under the embeddings key
# 1. Extract each headline using a list comprehension, accessing the headline key from each dictionary

headline_text = [article['headline'] for article in articles]
headline_text

# 2. To compute the embeddings:
# -> pass the entire list as an input to the create method

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=headline_text
)
response_dict = response.model_dump()

# batching embeddings in this way is much more effecient than making API calls for each input
# the output of this is a dictionary for each input

# 3. Extract these embeddings from the response and store them in the articles list of dictionaries
# -> loop over the indexes and articles using enumerate
# -> for each article, we assign the embedding at the same index in the response to the article's embedding key

for i, article in enumerate(articles):
    article['embedding'] = response_dict['data'][i]['embedding']

print(articles[:2])

# the total number/dimensions that represents the semantic meaning of its headline
# in other words, its position, vector, in the vector space
# key property of OpenAI embeddings is that they always return the number 1536
len(articles[0]['embedding'])

### Dimensionality Reduction snd t-SNE
# to reduce the dimensions from 1536 to 2 so that we can visualize
# use scikit-learn
# extract the embeddings from the articles list of dictionaries
# n_components: no. of dimensions we want to reduce to
# perplexity: used by the algorithm, must be less than the number of data points
# returns the transformed embeddings in a numpy array with n_components dimensions
# -> will result in some loss of information

from sklearn.manifold import TSNE
import numpy as np

embeddings = [article['embedding'] for article in articles]

tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))

# visualize the 2d
import matplotlip.pyplot as plt
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

topics = [article['topic'] for aticle in articles]
for i, topic in enumerate(topics):
    plt.annotate(topic, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()
######################################################################################################################

### Text Similarity ###
# we can measure how semantically similar two texts are by computing the distance between the vectors in the vector space
# using cosine distance -> ranges from 0 to 2 -> smaller numbers = greater similarity

# custom function
# send a request to the API and extract and return embeddings from the response
# function can be called on a single string or a list of strings
# always returns a list of lists

def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    response_dict = response.model_dump()

    return [data['embedding'] for data in response_dict['data']]

print(create_embeddings(["python is the best", "r is the best"]))
# to return just a single list of embeddings for the single string case, use zero-indec
print(create_embeddings("data is awesome")[0])

import scipy.spatial import distance
import numpy as np  # to access its argmin function, which treturns the index of the smallest value in the list

search_text = "computer"    # we want to compare this text to the embedded headlines
# embed this text using the create_embeddings func
search_embedding = create_embeddings(search_text)[0]

# to find the most similar headline to this text, loop over each article
# calculating the cosine distance between each embedded headline and the embedded query
# -> create an empty list to store the distances
# -> loop over each article 
# most similar headline will have the smallest cosine distance
# -> numpy's argmin function to return the index of the smallest value in the distances list
# -> use it to subset the article at this index and return its headline

distances = []
for article in articles:
    dist = distance.consine(search_embedding, article["embedding"])
    distances.append(dist)

min_dist_ind = np.argmin(distances)
print(articles[min_dist_ind]['headline'])








