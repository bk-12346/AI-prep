##### LANGCHAIN #####
### LANGCHAIN ECOSYSTEM ###
# Langsmith for deploying applications into production
# LangGraph for creating AI agents

## BUILDING LLM APPS ##
# -> LLM 
# -> a mechanism to help the model make decisions 
# -> knowledge database for the model to use 
# -> a mechanism for finding relevant data and integrating it into the chatbot

## PROMPTING OPENAI MODELS ##
from ast import Or
from langchain_openai import ChatOpenAI

# define a model to use the langchain app
# makes a request to the OpenAI API and returns the response back to the application
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key="key",
    max_completion_tokens=100,
    temperature=0
)

# to prompt the model call the .invole() method on a prompt string
llm.invoke("What is Langchain?")

# OR
prompt = 'Three reasons for using LangChain for LLM application development.'
response = llm.invoke(prompt)

print(response.content)

## PROMPTING HUGGING FACE MODELS ##
from langchain_huggingface import HuggingFacePipeline

# .from_model_id() can be used to download a model for a particular task
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B-Intruct",
    task="text_generation",
    pipeline_kwargs={"max_new_tokens":100}
)

llm.invoke("What is Hugging Face?")

### PROMPT TEMPLATES ###
# fundamental LangChain component that act as reusable recipes for defining prompts for LLMs
# can include: instructions. examples, additional context

from langchain_core.prompts import PromptTemplate

template = "Explain this concept simply and concisely: {concept}"

# -> covert this string into a prompt template use .from_template()
prompt_template = PromptTemplate.from_template(
    template = template
)

# -> to insert a variable call the invoke method on the prompt template
# pass it a dictionary to insert the values where there are input variables
prompt = prompt_template.invoke({"concept": "Prompting LLMs"})
print(prompt)

# -> now integrate it with an llm
# define the llm
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation"
)

# use langchain expression language to integrate the prompt template and the model
llm_chain = prompt_template | llm

# -> to pass an input into the chain use the invole() method again
concept = "Prompting LLMs"
print(llm_chain.invoke({"concept": concept}))

## CHAT MODELS ##
# support prompting with roles
# allow us to specify a series of messages from these roles
# roles: system, human, ai
# use ChatPromptTemplate class

from langchain_core.prompts import ChatPromptTemplate

# .from_messages method used to create the template
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a calculator that responds with math."),
        ("human", "Answer this math question: What is two plus two?"),
        ("ai", "2+2=4"),
        ("human", f"Answer this math question: {math}")
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", api_key="key")

llm_chain = template | llm
math = 'What is five times two?'

response = llm_chain.invoke({"math":math})
print(response.content)

### FEW SHOT PROMPTING ###

# 1. use prompt_template to specify how the questions and answers should be formatted

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = ()
example_prompt = PromptTemplate.from_template(f"Question: {question}\n{answer}")

prompt = example_prompt({"question":"What is the capital of Italy?""answer":"Rome"})
print(prompt.text)

# 2. few shot prompt template takes the example lists of dictionaries we created + template for formatting example
prompt_template = FewShotPromptTemplate(
    examples = examples,    # list of dictionaries
    example_prompt=example_prompt,   # formatted example
    suffix=f"Question:{input}",         # can add a suffix -> used to fromat ths input
    input_variables=["input"]
)

# 3. invoke prompt template with an example user input, and extract the text from the resulting prompt
prompt = prompt_template.invoke({"input":"What is the name of Henry Campbell's dog?"})
print(prompt.text)

# 4. instantiate a model
llm = ChatOpenAI(model="gpt-4o-mini", api_key="key")

# 5. chain the model
llm_chain = prompt_template | llm
response = llm_chain.invoke({"input":"What is the name of Henry Campbell's dog?"})
print(response.content)

## EXAMPLE ##
# 1. Create the examples list of dicts
examples = [
  {
    "question": "How many courses has Jack completed?",
    "answer": "36"
  },
  {
    "question": "How much XP does Jack have from completing courses?",
    "answer": "284,320XP"
  },
  {
    "question": "What technology does Jack learn about most?",
    "answer": "Python"
  }
]

# 2. Complete the prompt for formatting answers
example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

# 3. Create the few-shot prompt
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

prompt = prompt_template.invoke({"input": "What is Jack's favorite technology?"})
print(prompt.text)

# 4. Create an OpenAI chat LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key='<OPENAI_API_TOKEN>')

# 5. Create and invoke the chain
llm_chain = prompt_template | llm
print(llm_chain.invoke({"input": "What is Jack's favorite technology?"}))
