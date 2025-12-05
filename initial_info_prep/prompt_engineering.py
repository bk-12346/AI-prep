from openai import OpenAI
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

###### Prompt Engineering #####
### Create a get_response() function so we don't have to write the code every time
# just make changes to the prompt

def get_response(prompt):
    response = client.chat.completions.create(
        model = 'gpt_4o_mini'
        messages = [{
            "role":"user",
            "content": prompt
        }]
    )
    return response.choices[0].message.content

### Key Principles:
# use action verbs: write, complete etc
# provide specific and detailed instructions
# limit output length

# Create a request to complete the story
prompt = """Complete the given story in the triple backticks with only 2 paragraphs, in the style of Shakespeare. ```{story}```"""
# Get the generated response
response = get_reponse(prompt)
print("\n Original story: \n", story)
print("\n Generated story: \n", response)

### Prompt Components:
# instructions and output data to operate on

### Structures Outputs:
# output structures are tables, lists, structured paragraphs or custom format
# to generate a table clearly mention expected columns

# Create a prompt that generates the table
prompt = """Generate a table of 10 books, with columns for Title, Author and Year, that you should read given that you are a science fiction lover."""
# Get the response
response = get_response(prompt)
print(response)

# for list, add specific requirements for how the output list should be formatted
# for structured paragraphs tell it clear headings and subheadings etc
# for custom format, break it into instructions, output format, prompt

# Create the instructions
instructions = "Determine the language and generate a suitable title for the pre-loaded text excerpt that is provided using triple backticks."
# Create the output format
output_format = "Include the text, language, and title, each on a separate line, using 'Text:', 'Language:' and 'Title:' as prefixes for each line."
# Create the final prompt
prompt = instructions + output_format + f"```{text}```"
response = get_response(prompt)
print(response)

### Conditional Prompts: Have if/else structure

### One shot prompt
# Create the instructions
instructions = "Infer the language and the number of sentences of the given delimited text by triple backticks; then if the text contains more than one sentence, generate a suitable title for it, otherwise, write 'N/A' for the title."
# Create the output format
output_format = "Include the text, language, number of sentences and title, each on a separate line, and ensure to use 'Text:', 'Language:', and 'Title:' as prefixes for each line."
prompt = instructions + output_format + f"```{text}```"
response = get_response(prompt)
print(response)

########################################################################

### Few shot prompt
# consider task complexity to decide how many examples are needed
# fewer shots -> basic tasks
# diverse shots -> complex tasks

# Create a one-shot prompt
prompt = """Find the odd numbers in the set {1,3,7,12,19. The response should be like {1, 3, 7, 19}. Find the odd numbers in the set {3, 5, 11, 12, 16}.} """

response = get_response(prompt)
print(response)

response = client.chat.completions.create(
  model = "gpt-4o-mini",
  # Provide the examples as previous conversations
  messages = [{"role": "user", "content": "The product quality exceeded my expectations"},
              {"role": "assistant", "content": "1"},
              {"role": "user", "content": "I had a terrible experience with this product's customer service"},
              {"role": "assistant", "content": "-1"},
              # Provide the text for the model to classify
              {"role": "user", "content": "The price of the product is really fair given its features"}
             ],
  temperature = 0
)
print(response.choices[0].message.content)

### Single-step prompt
# Create a single-step prompt to get help planning the vacation
prompt = "Help me plan a beach vacation"

### Multi-step prompt
# break down the end goal into a series of steps 
# model goes through each step to give final output
# used for : 1. sequential tasks, 2. cognitive tasks such as analyzing solution correctness

# Create a prompt detailing steps to plan the trip
prompt = """make a plan for a beach vacation, which should include: four potential locations, each with some accommodation options, some activities, and an evaluation of the pros and cons.
"""

### Analyzing solution correctness
code = '''
def calculate_rectangle_area(length, width):
    area = length * width
    return area
'''
# Create a prompt that analyzes correctness of the code
prompt = f"assess the function provided in the delimited code string according to the three criteria: correct syntax, receiving two inputs, and returning one output. {code}"

### Prompting Techniques
## Chain of thought prompting
# requires llms to give reasoning steps before giving the answer
# used for complex reasoning tasks
# helps reduce errors

# Create the chain-of-thought prompt
prompt = "determine my friend's father's age in 10 years, given that he is currently twice my friend's age, and my friend is 20. Show the thinking step-by-step."

## One-Shot chain-of-thought prompt
# Define the example 
example = """Q: Sum the even numbers in the following set: {9,10,13,4,2}.
             A: Even numbers: {10,4,2}. Adding them: 10+4+2=16"""
# Define the question
question = """Q: Sum the even numbers in the following set: 15, 13, 82, 7, 14}.
              A:"""
# Create the final prompt
prompt = example + question
response = get_response(prompt)

### Chain-of-thought vs. Multi-step
# Multi-step incorporates steps inside the prompt
# chain of thought asks the model to generate intermediate steps 
# limitation of chain of thought is that one flawed thought can lead to a flawed final answer.

### Self-consistency prompts
# generates multiple chain of thoughts by prompting the model several times
# can be done by defining multiple prompts

# Create the self_consistency instruction
self_consistency_instruction = "Solve the problem with three experts and combine the results with a majority vote."
# Create the problem to solve
problem_to_solve = "If you own a store that sells laptops and mobile phones. You start your day with 50 devices in the store, out of which 60% are mobile phones. Throughout the day, three clients visited the store, each of them bought one mobile phone, and one of them bought additionally a laptop. Also, you added to your collection 10 laptops and 5 mobile phones. How many laptops and mobile phones do you have by the end of the day?"
# Create the final prompt
prompt = self_consistency_instruction + problem_to_solve

### Iterative Prompt Engineering and Refinement
# build a model -> feed it to the model -> observe and analyze the output -> reiterate to make the prompt better
# For few-shot prompts -> refine examples
# For multi-step prompts -> refine guiding steps
# For chain-of-thought and self-consistency prompts -> refine problem description
##################################################################################

### Text Summarization
# condenses text into shorter format while still preserving its essential meaning
# used to streamline business processes
# to make the prompt more effective, specify: 
# output limits -> specify no. of words/sentences
# output structure -> mention the format e.g bullet points
# summarization focus

# Craft a prompt to summarize the report
prompt = f"""Summarize the {report} in maximum 5 sentences, while focusing on aspects related to AI and data privacy"""

### Text Expansion
# generates text from ideas/bullets
# to improve prompt effeciency:
# ask model to expand delimited text
# highlight aspects to focus on
# provide output requirements

# Craft a prompt to expand the product's description
prompt = f"""Expand the {product_description} and write a one paragraph comprehensive overview capturing the key information of the product: unique features, benefits, and potential applications.
"""

### Text Transformation
# transforms given text to create new text
## language translation -> specify input + output language
# Craft a prompt to change the language
prompt = f"""Translate the {marketing_message} from English to French,Spanish, and Japanese"""

## tone adjustment -> specify target audience, use multistep prompt
# Craft a prompt to change the email's tone
prompt = f"""Change the tone of the {sample_email} to be proffessional, positive and user-centric"""

# Craft a prompt to transform the text
prompt = f"""Transform the text delimited by triple backticks with the following two steps:
Step 1 - Proofread it without changing its structure
Step 2 - Change the tone to be formal and friendly
 ```{text}```"""

### Text Analysis: Examine text to extract info
## Text Classification: assign categories to text
# -> specify known categories
# -> mention output requirements
# -> if no known labels, model uses its own understanding

# Craft a prompt to classify the ticket
prompt = f"""Classify the {ticket} as technical issue, billing inquiry, or product feedback, without providing anything else in the response."""

## Entity Extraction: e.g names, places etc
# specify entitities to extract
# specify output requirements

# Craft a few-shot prompt to get the ticket's entities
prompt = f"""Ticket: {ticket_1} -> Entities: {entities_1}
            Ticket: {ticket_2} -> Entities: {entities_2}
            Ticket: {ticket_3} -> Entities: {entities_3}
            Ticket: {ticket_4} -> Entities: """

### Code Generation and Explanation
## For Generation prompts can specify:
# -> problem description
# -> programming language
# -> format e.g script, function

# Craft a prompt that asks the model for the function
prompt = """write a Python function that receives a list of 12 floats representing monthly sales data as input and, returns the month with the highest sales value as output."""

# can give the model input-output examples to generate a program that maps a function
examples="""input = [10, 5, 8] -> output = 23
input = [5, 2, 4] -> output = 11
input = [2, 1, 3] -> output = 6
input = [8, 4, 6] -> output = 18
"""
# Craft a prompt that asks the model for the function
prompt = f"""infer the Python function that maps the inputs to the outputs in the provided {examples}"""

## For Explanations
function = """def calculate_area_rectangular_floor(width, length):
					return width*length"""
# Craft a multi-step prompt that asks the model to adjust the function
prompt = """test if the inputs to the functions {function} are positive, and if not, display appropriate error messages, otherwise return the area and perimeter of the rectangle."""

# Craft a chain-of-thought prompt that asks the model to explain what the function does
prompt = f"""explain the given function {function} step by step"""

#############################################################################################################
### PE for Chatbot Development
# use system messages to build chatbots because these guide model behaviour when answering users
# update the get_response() function to get system_prompts as well
def get_response(system_prompt, user_prompt):
  # Assign the role and content for each message
  messages = [{"role": "system", "content": system_prompt},
      		  {"role": "user", "content": user_prompt}]  
  response = client.chat.completions.create(
      model="gpt-4o-mini", messages= messages, temperature=0)
  return response.choices[0].message.content

# Try the function with a system and user prompts of your choice 
response = get_response("You are a finance expert", "What is money transfer?")
print(response)

## System Prompt
# define the purpose of the chatbot
# define response guidlines like audience, tone, length, structure
# define behavior guidelines like asking for missing info, providing context, etc

# Define the purpose of the chatbot
chatbot_purpose = "You are a customer support chatbot for an e-commerce company specializing in electronics. You will assist users with inquiries, order tracking, and troubleshooting common issues."
# Define audience guidelines
audience_guidelines = "The target audience is tech-savvy individuals interested in purchasing electronic gadgets"
# Define tone guidelines
tone_guidelines = "use a professional and user-friendly tone while interacting with customers.
system_prompt = chatbot_purpose + audience_guidelines + tone_guidelines
response = get_response(system_prompt, "My new headphones aren't connecting to my device")

# Define the order number condition
order_number_condition = "Please ask for the order number if they submitted a query about an order without specifying an order number"
# Define the technical issue condition
technical_issue_condition = "if the user is reporting a technical issue, start the response with 'I'm sorry to hear about your issue with ...."
# Create the refined system prompt
refined_system_prompt = base_system_prompt + order_number_condition + technical_issue_condition
response_1 = get_response(refined_system_prompt, "My laptop screen is flickering. What should I do?")
response_2 = get_response(refined_system_prompt, "Can you help me track my recent order?")

## Role-Playing Prompt
# tell the chatbot to adopt a specific role
# tailored language and content to fit the persona
# learns role from training data
# tell the model to act as a specific role

# Craft the system_prompt using the role-playing approach
system_prompt = "You are a personalized learning advisor chatbot that recommends textbooks for users. Your role is to receive queries from learners about their background, experience, and goals, and accordingly, recommend a learning path of textbooks, including both beginner-level and more advanced options."
user_prompt = "Hello there! I'm a beginner with a marketing background, and I'm really interested in learning about Python, data analytics, and machine learning. Can you recommend some books?"
response = get_response(system_prompt, user_prompt)
print(response)

base_system_prompt = "Act as a learning advisor who receives queries from users mentioning their background, experience, and goals, and accordingly provides a response that recommends a tailored learning path of textbooks, including both beginner-level and more advanced options."
# Define behavior guidelines
behavior_guidelines = "ask the user about their background, experience, and goals, whenever any of these is not provided in the prompt."
# Define response guidelines
response_guidelines = "recommend no more than three textbooks."
system_prompt = base_system_prompt + behavior_guidelines + response_guidelines
user_prompt = "Hey, I'm looking for courses on Python and data visualization. What do you recommend?"
response = get_response(system_prompt, user_prompt)

## External Context 
# need to provide the chatbot context of what we are building it for because most are trained on generic data
# give context as examples or in system prompt using triple backticks
# Define the system prompt
system_prompt = "You are a customer service chatbot delivery service that responds in a gentle way."
context_question = "What types of items can be delivered using MyPersonalDelivery?"
context_answer = "We deliver everything from everyday essentials such as groceries, medications, and documents to larger items like electronics, clothing, and furniture. However, please note that we currently do not offer delivery for hazardous materials or extremely fragile items requiring special handling."
# Add the context to the model
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[{"role": "system", "content": system_prompt},
            {"role": "user", "content": context_question},
            {"role": "assistant", "content": context_answer },
            {"role": "user", "content": "Do you deliver furniture?"}])
response = response.choices[0].message.content
print(response)

# Define the system prompt
system_prompt = f"""You are a customer service chatbot for MyPersonalDelivery whose service description is delimited by triple backticks. You should respond to user queries in a gentle way.
 ```{service_description}```
"""
user_prompt = "What benefits does MyPersonalDelivery offer?"
# Get the response to the user prompt
response = get_response(system_prompt, user_prompt)

