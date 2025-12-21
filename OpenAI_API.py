##### Structuring an API call

### API call should be able to:
# 1. Handle errors:
# -> display a user-friendly error message
# -> alternatives for when the service is unavailable
# 2. Should include moderation and safety features like:
# -> controlling unwanted inputs
# -> minimizing the risk of data leaks
# 3. Should have testing and validation:
# -> checking for responses that are out of topic
# -> testing for inconsistent behavior
# 4. should have communication with external systems
# -> calling external functions and APIs
# -> optimizing response times

### Components of an OpenAI API request
# -> better to output the model response in a format that is easily recognized, such as a JSON
# -> this way others in the pipeline can easily access the results of the output
# -> can be done by mentioning in the content part of the call to ''... in json format''
# -> then ressponse_format = {"type":"json_object"}
# -> this allows us to extract relevant data when communicating with external applications

from pyexpat import model
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"user", "content":"Please write down five trees with their scientific names in json format."}
    ],
    response_format={"type":"json_object"}
)

print(response.choices[0].message.content)

### Handling Errors
# important for simpllifying user experience
## 1. Connection Errors: Errors that occur due to connection errors on either side
# -> InternalServerError
# -> APIConnectionError
# -> APITimeoutError
# SOLUTIONS:
# -> checking your connection configuration, if there are any firewalls blocking access
# -> reaching out to support if error persists
# 2. Resource Limits Errors: due to the frequency of requests or the amount of text passed 
# -> ConflictError
# -> RateLimitError
# SOLUTIONS:
# -> checking limit restrictions
# -> ensure requests are within limits by reducing amount of text in requests
# 3. Authentication Error: when API key is expired, invalid, or revoked
# 4. Bad Request Error: request was not formed or missing key parameters

### Handling Exceptions: use try and except blocks
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content":"list five data science professions"}]
    )
except openai.AuthenticationError as e:
    print(f"OpenAI API failed to authenticate: {e}")
    pass
except openai.RateLimitError as e:
    print(f"OpenAI API request exceeded rate limit: {e}")
    pass

### BATCHING
# Rate limits: regulate flow of data, can prevent malicious attacks
# Rate Limit Error: -> due to too many requests, -> or too much text in a request
# Avoiding rate limits:
# -> Retry: short wait between request
# set the function to automaticall retry after a given time using a decorator
# decorator: a way to slightly modify the function without changing its inner code
# retry decorator: used to control the extent to which the function should be run again when failing

from tenacity import (retry, stop_after_attemot, wait_random_exponential)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attemot(6))

def get_response(model, message):
    response = client.chat.completions.creat(
        model=model,
        messages=[message],
        response_format={"type":"json_object"}
    )
    return response.choices[0].message.content

# -> Batching: processing multiple messages in one request at more staggered time intervals
# much more effecient approach than looping through the chat completions endpoint
countries=["united states", "ireland", "india"]
message=[
    {
        "role":"system",
        "content": """You are given a series of countries and are asked to return the country and the 
        capital city. Provide each of the questions with an answer in the response as separate content"""
    }
]
[message.append({"role":"user", "content":i}) for i in countries]
##############################################################################################################
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

messages = []
# Provide a system message and user messages to send the batch
messages.append({
            "role": "system",
            "content": "Convert each measurement, given in kilometers, into miles, and reply with a table of all measurements."
        })
# Append measurements to the message
[messages.append({"role": "user", "content": str(i) }) for i in measurements]

response = get_response(messages)
print(response)

# -> Reducing Tokens: quantifying and cutting down the number of tokens
# tokens: chunks of words
# can use tiktoken library

import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
prompt = "Tokens can be full words, or groups of characters commonly grouped together: tokenization."

num_tokens=len(encoding.encode(prompt))
print("number of tokens in prompt:", num_tokens)
#############################################################################################
client = OpenAI(api_key="<OPENAI_API_TOKEN>")
input_message = {"role": "user", "content": "I'd like to buy a shirt and a jacket. Can you suggest two color pairings for these items?"}

# Use tiktoken to create the encoding for your model
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
# Check for the number of tokens
num_tokens = len(encoding.encode(input_message['content']))

# Run the chat completions function and print the response
if num_tokens <= 100:
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[input_message])
    print(response.choices[0].message.content)
else:
    print("Message exceeds token limit")


### Function Calling
# OpenAI has tools that can be used to get better, more specific responses
# up till now we were using JSON structure for outputs
# however, this is purely based on the model's interpretation, which can be inconsistent sometimes
# Function calling addresses this issue
# -> enables OpenAI models to take user-defined functions as input
# -> this results in consistent responses without the need for complex text processing techniques
# Advantages:
# 1. go from unstructured to consistent structured output
# 2. call multiple functions to provide complex responses
# 3. define functions that enhance responses by calling external APIs

### Extracting Structured Data From Text
## 1. Set up function calling
# the format of the function is a list of dictionaries
# -> in the parameters, each key is the key that will be included in the resulting JSON schema
# -> each value is a dictionary containing the type of data it will hold + a description to help the model extract it

function_definition = [{
    'type':'function',           # specify type of tool; function type is used when we want to call our own user-defined func
    'function':{
        'name':'extract_job_info',
        'description':'Get the job information from the body of the input text',
        'parameters':{
            'type':'object',
            'properties':
                'job':{'type':'string',
                    'description':'Job title'},
                'location':{'type':'string',
                    'description':'Office location'}
        }
    }
}]

response=client.chat.completions.create(
    model = model,
    messages = messages,
    tools = function_definition
)

print(response.choices[0].message.tool_calls[0].function.arguments) # toole_calls is a list as there is an option to call multiple funcs
####################################################################################################################################

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Append the second function
function_definition.append({'type': 'function', 'function':{'name': 'reply_to_review', 'description': 'Reply politely to the customer who wrote the review', 'parameters': {'type': 'object', 'properties': {'reply': {'type': 'string','description': 'Reply to post in response to the review'}}}}})

response = get_response(messages, function_definition)

# Print the response
print(response)
#####################################################################################################

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

response = get_response(messages, function_definition)

# Define the function to extract the data dictionary
def extract_dictionary(response):
  return response.choices[0].message.tool_calls[0].function.arguments

# Print the data dictionary
print(extract_dictionary(response))
##############################################################################################################

### Working with Multiple Functions
## Parallel Function Calling
# -> ability to call multiple functions
# -> improves communication with the model
# -> use indexing to select the different responses from the tool_calls list

print(response.choices[0].message.tool_call[1].function.arguments)

# we may end up having a lot of function in the tools
# default behavior of the model is to choose which one to use based on the message and the function definitions
# equivalent to setting tool_choice to auto

tool_choice='auto'

# if we want the model to pick a specific function from the list
# change from 'auto' to a dictionary containing the name of the function we'd like the model to use

tool_choice = {'type':'function',
                'function':{'name':'extract_job_info'}
            }

### Doble-checking the Response
# specify the model to not make any assumptions
# do this in the system message

messages=[]
messages.append({"role":"system", "content":"Don't make assumptions about what values to plug into functions."})
messages.append({"role":"system", "content":"Ask for clarifiication if needed"})

# it may return an empty dictionary if it cannot find anything suitable to respond to the prompt

### Calling External APIs
# use requests library to call API by providing its URL and query parameters
# then specify the type of request and pass the URL and parameters to the request() function to get the reponse

import requests

url = "https://api.artic.edu/api/v1/artworks/search"
querystring={"q", keyword}      # parameters required as input to the API call for the external API
response = requests.request("GET", url, params=querystring)
######################################################################################################

# package the API call as a function
# returns the recommended artwork based on an input keyword

import requests

def get_artwork(keyword):
    url = "https://api.artic.edu/api/v1/artworks/search"
    querystring={"q", keyword}      # parameters required as input to the API call for the external API
    response = requests.request("GET", url, params=querystring)

    return response.text

# set up a Chat Completions request
# make sure it uses the user message to generate one keyword that will then be used as input for calling the external API
# so we provide a specific system message asking to interpret the prompt
# based on it extract one keyword for recommending artwork related to their preference
# also provide a user message as an example

import json         # to convert the response to a dictionary

function_definition=[{
    "type":"function",
    "function":{
        "name":"get_artwork",
        "description":"This function calls the Art Institute of Chicago API to find artwork that matches the keyword",
        "parameters":{
            "type":"object",
            "properties":{
                "artwork_keyword":{
                    "type":"string",
                    "description":"The keyword to be passed to the get_artwork function"
                }
            }
        },
    "result":{"type":"string"}
    }
}]

response=client.chat.completions.create(
    model=model,
    messages=[
        {
            "role":"system",
            "content":"You are an AI assistant, a specialist in history of art. You should interpret the user prompt, and based on it extract one keyword for recommending artwork to their preference."
        },
        {
            "role":"user",
            "content":"I don't have much time to visit the museum and would like some nice recommendations. I like the seaside and quiet places."
        }     
    ],
    tools=function_definition
)

# 1. check if there was a call to the API in tools
# -> check the finish_reason in rsponse.choices = tool_calls
# -> if yes, we extract the function that was called using function in message.tool_calls
# -> if no, print a message to the user stating that the request could not be understood

# 2. next step is to extract which function was called
# -> use the name from function_call
# --> if the function called is 'get_artwork
# -> we extract the keyword from the function call arguments
# -> once keyword is extracted we can proceed to call the external API using the 'get_artwork' function

# 3. if external API returns a response 
# -> we extract our recommendations
# -> we use a dictionary comprehension to extract the response due to the format of the output in the external API

if response.choices[0].finish_reason == 'tools_calls':
    function_call = response.choices[0].message.tool_calls[0].function
    if function_call.name == "get_artwork":
        artwork_keyword = json.loads(function_call.arguments)["artwork keyword"]
        artwork = get_artwork(artwork_keyword)
        if artwork:
            print(f"Here are some recommendations:{[
                i['title'] for i in json.loads(artwork)['data']
            ]}")
        else:
            print("Apologies, I couldn't make any recommendations based on the request")
    else:
        print("I couldn't find any artwork")
else:
    print("I am sorry, but I could not understand your request.")

#############################################################################################################

### Moderation in OpenAI API
## Moderation: process of analyzing input to determine if it contains any content that violates predefined policies or guidelines
# -> critical aspect of managing user-generated content
# -> OpenAI provides a moderations endpoint as a part of its API o help developers flag and filter out such content
# -> uses the model to asssess the content and assign a probability for each category of content violation
# -> categories include hate, harassment, self-harm, sexually explicit content, violent content

moderations_response = client.moderations.create(input="""... Exploding kitten... they are now dead....""")
print(moderations_response.results[0].categories.violence)

# game's instructions has been classified as violent
# so output to the 'violence' category is 'True'
# -> however if we give it the entire context and tell it that this is a game, it no longer classifies it as violent

## Prompt Injection Attack
# when text increases, it becomes harder to classsify the text
# -> opens it up to prompt injection attack
# -> malicious actors manipulate AI models to produce undesirable outcomes
# MITIGATIONS:
# -> limiting the amount of text in prompts
# -> Limiting the number of output tokens generated
# -> Using pre-selected content as validated input and output

## ADDING GUARDRAILS
# give it a system message to help it avoid going off-topic

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

user_request = "Can you recommend a good restaurant in Berlin?"

# Write the system and user message
messages = [{"role":"system", "content":"You are a chatbot that provides advice for tourists visiting Rome. Keep the topics limited to only covering questions about food and drink, attractions, history and things to do around the city. For any other topic, apologize and say 'Apologies, but I am not allowed to discuss this topic.'."},
{"role":"user", "content":user_request}]

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages
)

# Print the response
print(response.choices[0].message.content)

### Validation
# -> test model performance to uncover areas where the model might be prone to making mistakes
## Potential Errors
# 1. Misinterpreting context
# 2. Amplifying biases in its outputs if the training data is biased
# 3. Output outdated information
# 4. Being manipulated to generate harmful or unethical content
# 5. Inadvertently revealing sensitive information

## Testing Methods
# 1. Adversarial Testing
# -> provide model with prompts that are specifically designed to identify its areas of weakness so that they can be addressed before release
# -> used with other AI systems where even a small change in input can produce an unwanted or wrong output
# 2. Use libraries and datasets of standardized use cases that measure the model's performances in a variety of domains

### Safety with the OpenAI API
# 1. Ethics and fairness
# 2. Alignment with the scope of the product
# 3. Privacy of the data
# 4. Security of system against attacks
# -> use moderation API
# -> adversarial testing
# -> limiting number of output tokens
# -> human oversight or human-in-th-loop
# KEEP API KEY safe
# USE END-USER IDs

import uuid
uniques_id = str(uuid.uuid4())

response = client.chat.completions.create(
    model =model,
    messages=messages,
    user=uniques_id
)
print(uniques_id)