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
from SWE_practices import Document
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

### SEQUENTIAL CHAINS ###
# some problems can only be solved sequentially
# e.g. a chatbot that takes plans your travel intinarary 
# output from one chain becomes the input to the next chain

# -> create 2 prompt templates
# ->> 1 to generate suggestions for activities from inputs destination
# ->> another to create an itinerary for one day of activities from the model's top 3 suggestions

destination_prompt = PromptTemplate(
    input_variables=["destination"],
    template=f"I am planning a trip to {destination}. Can you suggest some activities to do there?"
)

activities_prompt = PromptTemplate(
    input_variables=["activities"],
    template = f"I only have one day, so can you create an itinerary from your top three activities: {activities}"
)

llm = ChatOpenAI(model="gpt-4o-mini", api_key="key")

# start by defining a dictionary that passes the destination promt template to the llm and parses the output to a string
# assigned to the activities key
# -> important because this is the input variable to the second prompt template
# we then pipe the first chain into the second prompt template
# then into the llm
# then parse again
seq_chain = ({"activities": destination_prompt | llm | StrOutputParser()}
    | activities_prompt
    | llm
    | StrOutputParser()) 

print(seq_chain.invoke({"destination":"Rome"}))

## Example ##
# Create a prompt template that takes an input activity
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?"
)

# Create a prompt template that places a time constraint on the output
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template="I only have one week. Can you create a plan to help me hit this goal: {learning_plan}."
)

# Invoke the learning_prompt with an activity
print(learning_prompt.invoke({"activity": "biking"}))

# Complete the sequential chain with LCEL
seq_chain = ({"learning_plan": learning_prompt | llm | StrOutputParser()}
    | time_prompt
    | llm
    | StrOutputParser())

# Call the chain
print(seq_chain.invoke({"activity": "biking"}))

### LANGCHAIN AGENTS ###
# Agents: use LLMs to take actions
# Tools: functions called by agents
# -> can be high-level utilities to transform inputs or can be task-specific

## ReAct AGENT ##
# Reasoning + Acting
# prompts model using a repeated loop of thinking, acting and observing
# use LangGraph for agentic systems
# example is creating a ReAct Agent that solves math problems

from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools

# initialize the llm 
# load the llm-math tool using load_tools() function
llm = ChatOpenAI(model="gpt-4o-mini", api_key="key")
tools = load_tools(["llm-math"], llm=llm)

# create agent using creat_react_agent
agent = create_react_agent(llm, tools)

messages = agent.invoke({"messages": [("human", "What is the square root of 101?")]})
print(messages)

# if we just want final response without metadata
print(messages['messages'][-1].content)

### TOOLS ###
# we can design our own
# must follow a format, name, description as llm uses this to know when to call the tool

from langchain_community.agent_toolkits.oad_tools import load_tools

tools = load_tools(["llm-math"], llm=llm)
print(tools[0].name)
print(tools[0].description)
# .return_direct() used to determine if the agent should stop invoking this tool if True
print(tools[0].return_direct)

## Custom python function to generate a financial report for a company
# use the @tool decorator
# -> modifies the function so that it is in the correct format used by the tool
from langchain_core.tools import tool

@tool
def financial_report(company_name: str, revenue: int, expenses: int) -> str:
    """Generate a financial report for a company that calculates net income."""
    net_income = revenue - expenses

    report = f"Financial Report for {company_name}:\n"
    report += f"Revenue: ${revenue}\n"
    report += f"Expenses: ${expenses}\n"
    report += f"Net Income: ${net_income}\n"
    return report

# examine the tool
print(financial_report.name)
print(financial_report.description)
print(financial_report.return_direct)
print(financial_report.args)

# integrating the tool
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", api_key="key")
agent = create_react_agent(llm, [financial_report])

messages = agent.invoke({"messages": [("human", "TechStack generated made $10 million with $8 million of costs. Generate a financial report.")]})
print(messages)

### RAG ###
# user query is embedded and used to retrieve the most relevant docs from the database
# these docs are added to the model's prompts so that the model has extra context to inform its response
# 3 steps:
# -> 1. Document Loader -> load the documents into LangChain with document loaders
# ->>> classes designed to load and configure docs for integration with AI systems
# ->>> document loaders for common class types: .pdf, .csv
# ->>> PDF Document Loader
# ->>>> instantiate the PyPDFLoader class, passing in the path to the PDF file we are loading

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredHTMLLoader

loader1 = PyPDFLoader("path/to/file/file.pdf")
loader2 = CSVLoader('file.csv')

data = loader1.load()
print(data[0])

# -> 2. Splitting -> split the documents into chunks 
# ->>> chunks are units of information that we can index and process individually
# ->>> done to fit doc into the LLM context window

## CHUNK OVERLAP
# ->> implemented to counteract lost context during chunk splitting
# ->>  the overlap helps retain context
# ->> can be increased of the model show signs of losing context
 
# STRATEGIES FOR CHUNK SPLITTING
# 1. CharacterTextSplitter
# -> based on separator first then evaluates chunk_size and chunk_overlap to check 

from langchain_text_splitters import CharacterTextSplitter

quote = '''Hello. this is me.\n thankyou'''
chunk_size=24
chunk_overlap=3

ct_splitter = CharacterTextSplitter(
    separator = '.',
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# apply the splitter to the quote using the .split_text() method
docs = ct_splitter.split_text(quote)
print(docs)
print([len(doc) for doc in docs])

# 2. RecursiveCharacterTextSplitter
# -> takes a list of separators to split on and works through the list from left to right, splitting the document using each separator

from langchain_text_splitters import RecursiveCharacterTextSplitter

rc_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap
)

docs = rc_splitter.split_text(quote)
print(docs)

# 3. Storage + Retrieval -> encoding and storing the chunks for retrieval, which can utilize a vector database if necessary
# -> use a vector database to store the documents and make them available for retrieval
# -> a user query can be embedded to retrieve the most similar docs from the database and insert them into the model prompt

docs = [
    Document(
        page_content = "In all marketing copy, TechStack should always be written with the T and S capitalized. Incorrect: techstack, Techstack, etc.",
        metadata={"guideline": "brand-capitalization"}
    ),
    Document(
        page_content = "Our users should be referred to as techies in both internal and external communications.",
        metadata={"guideline": "referring-to-users"}
    )
]

# embed the doc using an embedding model 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding_function = OpenAIEmbeddings(api_key='key', model='text-embedding-3-small')

# create a chroma db from a set of documents by calling .from_documents
vectorstore = Chroma.from_documents(
    docs,
    emnedding=embedding_function,
    persist_directory = "path/to/directory"
)

# to integrate the database with other LangChain components, we convert it into a retriever with the .as_retriever() method
retriever = vectorstore.as_retriever(
    search_type = "similarity",     # specifying that we want to do a similarity 
    search_kwargs = {"k":2}         # return the top 2 most similar docs for each user query
)

# now the model knows what to do so we construct a prompt template
# this one starts with the instruction to review and fix the copy provided
# insert the retrieved guidelines and copy to review

from langchain_core.prompts import ChatPromptTemplate

message = f"""
Review and fix the following TechStack marketing copy with the following guidelines in consideration:

Guidelines: {guidelines}

Copy: {copy}

Fixed Copy:
"""

prompt_template=ChatPromptTemplate.from_messages([("human", message)])

# chain it all together
from langchain_core.runnables import RunnablePassthrough

rag_chain = ({"guidelines":retriever, "copy": RunnablePassthrough()}
    | prompt_template
    | llm)

response = rag_chain.invoke("Here at teckstack, our users are the best in the world!")
print(response.content)

## EXAMPLE ##
loader = PyPDFLoader('rag_vs_fine_tuning.pdf')
data = loader.load()

# Split the document using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50)
docs = splitter.split_documents(data) 

# Embed the documents in a persistent Chroma vector database
embedding_function = OpenAIEmbeddings(api_key='<OPENAI_API_TOKEN>', model='text-embedding-3-small')
vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding_function,
    persist_directory=os.getcwd()
)

# Configure the vector store as a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Add placeholders to the message string
message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# Create a chain to link retriever, prompt_template, and llm
rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm)

# Invoke the chain
response = rag_chain.invoke("Which popular LLMs were considered in the paper?")
print(response.content)