# learning openai api
# summary
# API - application programming interface
# takes request to a system and gets a response
# sent in request - 1. model we want, 2. data for model to use, 3. any other params to customize model behavior
# response - text, image, audio, video, etc.

##### Make requests to API #####
# API's have different access points called endpoints
# endpoints may require authentication

from openai import OpenAI
# this a client - configures the enviornment for communication with the API
client = OpenAI('Your_OpenAI_API_key')

# reponse is the request to the API
# response is a chat completion object that has attributes for accessing different info
# chat completions endpoint is used to send a series of messages that represent a convo to a model
# messages argument takes as input a list of dictionaries where content sent from the iuser role allows us to prompt the model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

# response message gives the model response as a string
print(response.choices[0].message.content)

##### Text Editing #####
# make a prompt
prompt = "Write a story about a cat"
# send it to the messages
messages = [{"role": "user", "content": prompt}]
print (response.choices[0].message.content)

##### Text Summarization #####
text = "text to summarize"
prompt = f"summarize the text: {text}"
# now pass this prompt to messages
# use max_completion_tokens to limit the length of the response

##### Cost ######
# determined by input tokens and output tokens
# input tokens found in API response
print(response.usage.prompt_tokens)
# output tokens is usually max_completion_tokens

##### Text Generation #####
# control the randomness of the response using the temperature param
# 0 - highly deterministic
# 2 - very random
# can be used for content creation, product descriptions etc

##### Shot Prompts #####
# providing examples to the model to generate better responses
### Zero shot - no examples, just instruction
### One shot - one example
### Few shot - multiple responses

##### Single Turn Tasks #####
# one input - one output

##### Multi Turn Tasks #####
# follow up prompt - build on previous conversations

##### Roles #####
# how thw chat  model performs
### Role: System
# allows us to specify a message to control the behavior of the assistant
### Role: User
# provides an instruction to the assistant
### Role: Assistant
# response to the user interaction
# can also be provided in requests to help the model understand how we want the results

##### Mult-Turn Conversation #####
# we want a mechanism so that:
# -> when a user message is sent
# -> an assistant response is generated
# -> it is fed back into the messages
# -> stored to be sent with the next messages
# this basically creates history and gives the model context

from openai import OpenAI

client = OpenAI("YOUR_OPENAI_API_KEY")

messages = [{"role":"system",
"content": "You are a helpful Python tutor."}]
# user questions that we want responses to
user_qs = ["Why do we need to use dictionariies?", "Summarize the previous response."]

# we want a response for each question so we loop over the user_qs list
for q in user_qs:
    # print the question
    print("User": q)

    # convert user questions into messages for the API as a dictionary
    user_dict = {"role":"user", "content": q}

    # add it to the list of messages
    messages.append(user_dict)

    # now just send the messages to the model
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages
    )
    
    # dictionary for assistant so that we store all the previous responses
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)

    # print the response to the question
    print("Assistant:", response.choices[0].message.content)
