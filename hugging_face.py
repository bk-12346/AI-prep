##### Hugging Face #####4
# Hub - centralized space to find the best models
# Model - pre-trained, comes with amodel card for info

### Running Models ###
# can be done as 1. local inference, 2. use inference provider

## 1. Local Inference - use the model directly on your machine
# 1. import pipeline class from transformers library
from transformers import pipeline
# 2. instantiate it
gp2_pipeline = pipeline(task="text-generation", model="openai_community/gpt2")
# 3. call model
print(gpt2_pipeline("What is AI?"))
# OR
results=gpt2_pipeline("What is AI?", max_new_tokens=10, num_return_sequences=2)
# max_new_tokens to limit length of response, num_return_sequences to generate 2 sequences
# produces a list of dictionaries so loop over them to get the generated response
for result in results:
    print(result["generated_text"])

## 2. For Inference Providers
# 1. Create an inference client - configures the envo=ironment for communicating with the inference API
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider = "together",  # provider we want to use
    api_key = os.environ["HF_TOKEN"]    # specify API key
)

completion = client.chat.completions.create(
    model = "deepseek-ai/DeepSeek-V3",
    messages = [
        {
            "role":"user",
            "content": "What is the capital of France?"
        }
    ]
)

print(completion.choices[0].message)

### HuggingFace Datasets ###
# under datasets tab
# data studio for more detailed look + manipulate the dataset using SQL queries
# HuggingFace developed a python package to interact with datasets called datasets
# allows us to access, download, use, and share datasets
pip install datasets

# download the dataset
from datasets import load_dataset
data = load_dataset("IVN-RIN/BioBERT_Italian")

# Split parameter to specify partitions to download such as train, test, validate
# Check dataset card to see which splits are available
data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

# most datasets use APache Arrow
# datasets that use columnar-based storage instead of row-based data storage for faster querying
# Data manipulation for Apache is different from pandas DataFrames
data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

# to filter, use .filter() method with a lambda funnction that applies the defined criteria to each row
# filter for pattern " bella ", basically checking if bella is in each row
filtered = data.filter(lambda row: " bella " in row['text'])
print(filtered)

# to select rows from an indices, use .select() method
# select the first 2 rows
sliced = filtered.select(range(2))
print(sliced)

# to extract the 'text' for a row, pass the row index, and the column
print(sliced[0]['text'])

### Text Classification ###
# assigning predefined categories to text
# sentiment analysis is an example
# example of using a model that has been pre-trained for sentiment analysis
from transformers import pipeline

my_pipeline = pipeline(
    task="text-classification",
    model = "distilbert-base-uncased-finetuned-sst-2-english"
)
print(my_pipeline("Wifi is slower than a snail today!"))

# another type of text classification is grammatical correctness
# labels text as Acceptable/Unacceptable
from transformers import pipeline

# Create a pipeline for grammar checking
grammar_checker = pipeline(
  task="text-classification", 
  model="abdulmatinomotoso/English_Grammar_Checker"
)

# Check grammar of the input text
output = grammar_checker("I will walk dog")
print(output)


# another type is QNLI; question natural language inference
# checks if a premise answers a question
# we evaluate question/premise pairs
# when calling the classifier provide both the question and the premise
# label_0 is entailment
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model = "cross-encoder/qnli-electra-base"
)
print(my_pipeline("Where is Seattle located?, Seattle is located in Washington state."))

# another is dynamic category assignment
# dynamically assigns categories based on content
# e.g classifying the request 'I want to know more about your pricing' as 'sales->high confidence', 'marketing', 'support'
# model assigns confidence score to category
# used in content moderation and recommendation systems

# we use zero-shot classification
# this allows model to assign predefined categories to text even if it hasn't been trained on those categories

text = "AI-powered robots assist in complex brain surgeries with precision."
# Create the pipeline
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
# Create the categories list
categories = ["politics", "science", "sports"]
# Predict the output
output = classifier(text, categories)
# Print the top label and its score
print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")

### Challenges of text classification
# Ambiguity, sarcasm/irony, multilingual -> requiring tailored processing for diverse linguistic structures
# Addressing these issues requires robust preprocessing and diverse languange models

### Text Summarization ###
## can be extracted:
# -> select key sentences from text
# -> Efficient, needs fewer resources
# -> lacks flexibility, may result in less cohesive summaries
# -> for legal clauses or financial

# output is a dictionary containing summarized text
from transformers import pipeline
summarizer = pipeline(task="summarization", model="nyamuda/extractive-summarization")

text = "this is a large text..."
summary = summarizer(text)
print(summary[0][summary])

## can be abstractive
# -> generates new text
# -> captures main ideas while rephrasing for clarity and readability
# -> more resource-intensive
# -> news articles etc

# new parameters -> min_new_tokens, max_new_tokens -> control length

# output is a dictionary containing summarized text
from transformers import pipeline
summarizer = pipeline(task="summarization", 
model="sshleifer/distilbart-cnn-12-6",
min_new_tokens=10,
max_new_tokens=100)

text = "this is a large text..."
summary = summarizer(text)
print(summary[0][summary])

### Auto Models and Tokenizers ###
## Auto Classes -> flexible way to load models, tokenizers, and other components without manual setup
# offer more control than pipelines
# import AutoModel Class for your task
from transformers import AutoModelForSequenceClassification

# download a pre-trained classification model using .from_pretrained()
model = AutoModelForSequenceClassification.from_pretrained("model_name")

## AutoTokenizers -> used to prepare the text input data
# recommended to use the tokenizer paired with the model to make sure input is processed
# exactly how it was during training
# this happens automatically in pipeline but here you do it yourself

# to retrieve the tokenizer for the model, import AutoTokenizer from Transformers
from transformers import AutoTokenizer

# call .from_pretrained()
tokenizer = AutoTokenizer.from_pretrained("model_name")

# tokenizers work by first cleaning the input
# then divide the text into chunks -> use .tokenize()
tokens = tokenizer.tokenize("AI: Helping robots think")
print(tokens)

# building a custom pipeline with AutoClasses
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

my_model=AutoModelForSequenceClassification.from_pretrained("model_name")
my_tokenizer = AutoTokenizer.from_pretraines("mosel_name")

my_pipeline = pipeline(
    task="sentiment-analysis",
    model=my_model,
    tokenizer=my_tokenizer
)

### Q&A Models ###
# requires 2 inputs -> a document, typically a pdf, and a question
# content is analyzed and an answer is generated either quoting directly from doc or rephrasing
# example

from pypdf import PdfReader

# load the PDF file
reader = PdfReader("path/to/file")

# extract from all pages -> use .pages attribute
document_text = ""
# loop over the doc
# text from all pages is appended to a variable, combining them into a single string + preparing the PDF for processing
for page in reader.pages:
    document_text += page.extract_text()

# Q&A pipeline
qa_pipeline = pipeline(
    task="question-answering",
    model="model_name"
)

question="How many volunteer days are offered?"

# get the answer from the qa pipeline
result = qa_pipeline(question=question, context=document_text)
print(f"Answer: {result['answer']}")




