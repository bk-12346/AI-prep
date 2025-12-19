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

response = get_response(messages, function_definition)

# Define the function to extract the data dictionary
def extract_dictionary(response):
  return response.choices[0].message.tool_calls[0].function.arguments

# Print the data dictionary
print(extract_dictionary(response))
##############################################################################################################

### Working with Multiple Functions
#
# 
# 
# 
# 
