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

######################################################################################################
######################################################################################################

### SEMANTIC SEARCH AND ENRICHED EMBEDDINGS
## Semantic Search: uses embeddings to return the most similar results to a search query
# STEP 1: Embed the search query and text to compare against
# STEP 2: compute the consine distances between the embedded search query  and other embedded texts
# STEP 3: Extract the texts with the smallest consine distance

## Enriched Embeddings
# STEP 1:
# -> embed the headline text + the topic and keywords
# -> to do this, combine the information from each article into a single string
# -> this reflects the information stored in the dictionary, and the keywords are delimited with a comma and space

articles = [
    {
        "headline":"Economic Growth Continues Amid Global Uncertainity",
        "topic":"Business",
        "keywords":["economy", "business", "finance"]
    },
    ...
    {
        "headline":"1.5 Billion Tune-in to the World Cup Final",
        "topic":"Sport",
        "keywords":["soccer", "world cup", "tv"]
    }
]

# to combine these features for each article, we define a function called create_article_text
# -> it uses a f-string, to return the desired string structure
# -> define multi-line string using 3 """"""
# -> the values are extracted using their keys and inserted into the string at the desired locations
# -> for 'keywords' we use the join list method, which joins the contents of the list together into a single string

def create_article_text (article):
    return f"""Headline:{article['headline']}
    Topic:{article['topic']}
    Keywords:{', '.join(article['keywords'])}"""

print(create_article_text(articles[-1]))

# to apply the function and combine the features for each article
# -> use a list comprehension
# -> calling our function on each article in articles

article_texts = [create_article_text(article) for article in articles]

# to embed these strings, call the create_embeddings function on the result
# this creates a list of embeddings for eacch input using the OpenAI API
article_embeddings = create_embeddings(article_texts)
print(article_embeddings)

# STEP 2:
# define a function that takes a query_vector + the embedded search query + embeddings to compare against our embedded article 
# -> returns n most similar results based on their cosine distances
# -> for each embedding calculate the cosine distance to the query_vectore
# -> store it in a dictionary along with the embedding's index
# -> append to a list called distances
# -> use the sorted function and its key argument to sort the distances list by the distance key in each dictionary
# -> key argument takes a function to evaluate each dictionary in diatances and sort by 
# -> here we use the lambda function that accesses the distance key from each dictionary

from scipy.spatial import distance

def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance":dist, "index":index})
    distances_sorted = sorted(distances, key=lambda x:x["distance"])
    return distances_sorted[0:n]

# STEP 3:
# -> embed search wuery using create_embeddings function
# -> extract its embeddings by zero-indexing
# -> next, use find_n_closest function to find the 3 closest hits based on our articles embeddings
# -> to extract the most similar headlines, we loop through each hit, using the hit's index to subset the corresponding headline and print

query_text = "AI"
query_vector = create_embeddings(query_text)[0]

hits = find_n_closest(query_vector, article_embeddings)

for hit in hits:
    article = article[hit['index']]
    print(article['headline'])
##########################################################################################################################################

### RECOMMENDATION SYSTEMS ###
# work similar to semantic search engines
# a list of items we want to recommend and at least one data point we want to recommend based on
# STEP 1: Embed the potential recommendations + the data point we have
# STEP 2: Calculate the cosine distances
# STEP 3: Recommend the item or items that are closest in the vector space
# same articles dataset example from above
# recommendation is based on headline, topic and keywords, stored in a dictionary called current_article

current_article = {"headline":"How NVIDIA GPUs Could Decide Who Wins the AI Race",
"topic":"Tech",
"keywords":["ai", "business", "computers"]}

# -> to prepare for embedding, combine the features into a single string for each article
# -> create function that extracts the headline, topic and keywords and uses an f string to combine them into a single string
def create_article_text(article):
    return f"""Headline:{article['headline']}
    Topic:{aticle['topic']}
    Keywords:{', '.join(article['keywords'])}"""

# to combine the features, call the function on each article in articles using a list comprehension
# do the same for current_article
article_texts = [create_article_text(article) for article in articles]
current_article_text = create_article_text(current_article)
print(current_article_text)

# -> next, embed both sets of article strings using the create_embeddings function

def create_embeddings(texts):
    response = openai.Embedding.create(
        model = "text-embedding-3-small",
        input = texts
    )
    response_dict = response.model_dump()

    return [data['embedding'] for data in response_dict['data']]

article_embeddings = create_embeddings(article_texts)
current_article_embeddings = create_embeddings(current_article_text)[0]

# find closest distances
def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = spatial.distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index":index})
    distances_sorted = sorted(distances, key=lambda x:x["distance"])
    return distances_sorted[0:n]

# call this function on both sets of embedded articles, then loop through
hits = find_n_closest(current_article_embeddings, article_embeddings)

for hit in hits:
    article = articles[hit['index']]
    print(article['headline'])
#####################################################################################################################

## a more sophisticated system would base the recommendations on the current article + user history
### ADDING HISTORY ###
# let's consider that a user has visited two articles, stored in user_history

user_history =[
    {
        "headline":"How NVIDIA GPUs Could Decide Who Wins the AI Race",
        "topic":"Tech",
        "keywords":["ai", "business", "computers"]
    },
    {
        "headline":"Tech Giant Buys 49% Stake in AI Startup",
        "topic":"Tech",
        "keywords":["business", "AI"]
    }
]

# -> now to get the vector most similar to the 2 vectors, combine them by taking the mean
# -> combine cosine distances
# -> recommend the closest vector
# ->> if the recommended article has already been viewed, we'll make sure to return the nearest seen article
# first 2 steps are the same as before: combining the features for each article and embedding the resulting strings
# ->> difference is that now we take the mean to aggregate the 2 vectors into one that we can compare to other articles

def create_article_text(article):
    return f"""Headline:{article['headline']}
    Topic:{article['topic']}
    Keywords:{', '.join(article['keywords'])}"""

history_texts = [create_article_text(articles) for article in user_history]
history_embeddings = create_embeddings(history_texts)
mean_history_embeddings = np.mean(history_embeddings, axis=0)

# for the articles to recommend, we filter the list so that it contains only the articles not in user_history
articles_filtered = [article for article in articles if article not in user_history]

# now combine the features, then embed the text
article_texts = [create_article_text(article) for article in articles_filtered]
article_embeddings = create_embeddings(article_texts)

# now just compute the cosine distances
hits = find_n_closest(mean_history_embeddings, article_embeddings)

for hit in hits:
    article = articles_filtered[hit['index']]
    print(article['headline'])
##################################################################################################################################

### EMBEDDINGS FOR CLASSIFICATION TASKS ###
# assigning labels to items
# -> use zero-shot classification: classification won't use any laba=eled data
# Process:
# 1. Embed class descriptions: embed the labels and use them as reference points to base the classification on
# 2. Embed the item to classify
# 3. Calculate cosine distanced to each embedded label
# 4. Assign the most similar label

# let's say we have the following topic classes
topics = [
    {'label':'Tech'},
    {'label':'Science'},
    {'label':'Sport'},
    {'label':'Business'}
]

# STEP 1: Extract the labels as a single list and use these as the class descriptions
class_description = [topic['label'] for topic in topics]

# STEP 2: Embed each topic label using the create_embeddings function
class_embeddings = create_embeddings(class_description)

# article we want to classify
article = {"headline":"How NVIDIA GPUs Could Decide Who Wins the AI Race",
"keywords":["ai", "business", "computers"]}

# STEP 3: combine the headline and the keyword information into a single string that we can embed
# -> define the create_article_text fubction then use it
def create_article_text(article):
    return f"""Headline:{article['headline']}
    Keywords:{', '.join(article['keywords'])}"""

article_text = create_article_text(article)

# STEP 4: Embed the text using create_embeddings() and using zero-index so that we have a single list of numbers
article_embeddings=create_embeddings(article_text)[0]

# STEP 5: Cosine Calculations
# we only want one result so we remove the n= part
def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index":index})

    return min(distances, key=lambda x: x["distance"])

# STEP 6: Call the function to return the distance and the index o the label
closest = find_closest(article_embeddings, class_embeddings)

# STEP 7: Use this index to subset the topics dictionary and extract the label
label = topics[closest['index']['label']]
print(label)

## ->> this can return a wrong label because we didn't define any meaning for the class labels
# -> a better approach is to use more detailed class descriptions

topics = [
    {'label':'Tech', 'description':'A news article about technology'},
    {'label':'Science', 'description':'A news article about science'},
    {'label':'Sport', 'description':'A news article about business'},
    {'label':'Business', 'description':'A news article about business'}
]

# now only difference here is that we extract the descriptions in a single list using the dexcriptions key and embe key
class_descriptions = [topic['description' for topic in topics]]
class_embeddings = create_embeddings(class_descriptions)
