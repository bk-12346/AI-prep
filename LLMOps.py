# LLMs are different from trasitional ML because:
# 1. typically pre-trained
# 2. have a huge number of parameters
# 3. require significant computational resources
# 4. can sometimes have unpredictable outcomes
# 
### LLMOps 
# ensures the seamless integration of LLms into organizations
# ensures smooth transition across lifecycle phases, from ideation to deployment
# provide scalable, risk mitigating solutions

##### Lifecycle of LLMs #####
##### 1. Ideation Phase
## Data Sourcing: identifying data needs
# is the data relevant?
# is the data available? -> transform data, set up additional databases, evaluate costs, consider limitations
# does the data meet standards?

## Select base model selection
# whether to use propiritry or open-source model
# evaluate model on: 
# 1. performance -> response quality + speed
# 2. characteristics -> data used to train the model
# 3. practical considerations -> type of liscense associated with the model
# 4. important secondary factors -> popularity etc


##### 2. Development Phase
### Prompt Engineering
# important b/c: 1. improves performance, 2. more control over output, 3. avoid bias and hallucinations
# Elements of a prompt: 1. instructions, 2. examples/context 3. input data 4. output indicator
# Experiment with: 1. LLM settings like temperature or max tokens, 2. in-context learning and other prompt design patterns
# Prompt Management: to generate prompt templates
# -> crucial for efficiency, reproducibility, and collaboration
# -> important to track: prompt, output, model and settings
# -> use a prompt manager or version control
# -> begin generating a good collection of input-output pairs for regeneration

### Chains and Agents
# provide flow and structure
# to use a template we need input and examples
# we go through a few steps:
# -> 1. recieving input
# -> 2. searchinng examples
# -> 3. prompt creation
# -> 4. output retrieval
# -> 5. output parsing
# we use chains and agents to build this functionality

## Chain: pipeline or flow of connected steps that take an input and give an output
# important b/c: 
# -> 1. develop sophisticated systems
# -> 2. establish a modular design, enhancing scalibilty and operational efficiency
# -> 3. unlock endless possibilities for customization

## Agents consist of:
# -> multiple actions (tools)
# -> LLM that decides which tool to pick
# important when: 1. there are many actions, 2. optimal sequence of steps is unknown, 3. we are uncertain about inputs
# an action can happen many times and we cannot know that
# ----------------------------------------------------------------------------------
#               |   Chains      |   Agent
# Nature        | Deterministic | Adaptive
# Complexity    |   Low         | High
# Flexibility   |   Low         | High
# Risk          |   Low         | High
# ----------------------------------------------------------------------------

### RAG vs. Fine-tuning: methods to incorporate external info
# --- Use RAG:
# -> when including factual knowledge
# -> keeps capabilities of original LLM, easy to implement, always up-to-date
# -> issue: adds extra components and requires careful engineering

# --- Use Fine-tuning:
# -> when specializing in a specific domain
# -> when we need full control as extra components are added
# -> issue: needs labelled data, bias amplification and catastrophic forgetting

## RAG: common LLM design pattern
# -> combines model's reasoning abilities with external factual knowledge
# -> 3 step chain: 1. Retrieve related docs, 2. Augment prompt with examples, 3. Generate output
## RAG with vector databases
# 1. Retrieve
# -> convert input into embedding (numerical representation) to capture its meaning
# -> similar meanings have similar embeddings
# -> embeddings created using pre-trained models
# -> search vector database containing all the embeddings
# -> compare input embedding with these embeddings using cosine similarity
# -> retrieve the MOST SIMILAR docs
# 2. Augment
# -> combine input with top-k docs and create augmented prompt
# 3. Generate
# -> uses prompt to create output

## Fine-tuning
# -> adjusts the LLM weights based on our own data
# -> expand to specific tasks and domains such as: different languages, specialised fields

## 1. Supervised Fine-tuning (sort of transfer learning):
# -> type of data needed: transformation data (inputs with desired outputs)
# -> Approach: retrains parts of the model

## 2. RL from Human Feedback (usually done after supervised fine-tuning):
# type of data needed: rankings or quality scores obtained from likes and dislikes for example
# Approach: train an extra reward model to predict output quality + optimize the original LLM to maximise this


### Testing
# crucial b/c LLMs make mistakes
# also important for testing the readiness of the LLM for deployment
# evaluate the LLM output

## for testing we need:
# just need data, not necessarily labelled
# quality of output checked using a variety of metrics

## 1. Build a test set
# -> test set data must closely resemble real-world scenarios or unlabeled text data to simulate typical inputs

## 2. Choose the right metric
#    ------------
#   | LLM Output |  ---------->     Do we have correct output?  ----------->    Use ML metrics like accuracy to access correctness
#    ------------                           |
#                                           | No
#                                           |
#                                   Do we have a reference answer?  ------->    Use text comparison methods -> 1. statistical methods that compare
#                                           |                                                                      overlap b/w predicted and output text
#                                           | No                                                               2. model-based methods: LLM judges
#                                           |
# Use unsupervised metrics <------  Do we have human feedback?      ------->    Use feedback score metrics

## 3. Optional secondary metrics
# 1. Output characteristics: bias, toxicity, helpfulness
# 2. Operational characteristics: latency, total incuurred cost, memory usage

##### 3. Operational Phase
### Deployment
## 1. where to host component
# -> public or private
## 2. API design
# -> which parts of our system need to be open for communication
# -> design effects scalability, cost, speed, infrastructure
## 3. How to run
# -> containers/serverless functions/ cloud managed services

### CI/CD: automates integration, testing and deployment
##          CI                                              |       CD
#   1. Source: retrieve source code                         | 1. Retrieve: retrieve container from registry
#   2. Build: create a container image containg the code    | 2. Test: perform deployment tests
#   3. Test: perform integration tests                      | 3. Deploy: Deploy container to environments through 1. Staging 2. Production
#   4. Register: store the container in a registry          | 

### Scaling
# 2 strategies: 1. Horizontal: add more machines; for traffic. 2. Vertical: boost one machine; for speed

### Monitoring and Observability

## Monitoring: continously watches a system
# 1. Input monitoring:
# -> tracking changes, errors or melicious content in application inputs
# Data Drift: change in the input data distribution over time
# -> address it by: 1. monitoring data distribution, 2. periodically updating the model
# 2. Functional Monitoring:
# -> monitoring application overall health like down time, request volume,response time, error rates
# -> For LLMs, monitor chain/agent executions, costs, system resource usage
# 3. Output Monitoring:
# -> assessing response application generates
# -> metrics like bias, toxicity, helpfulness
# Model Drift: relationship between input and output changes due to external factors
# -> use human feedback loops

## Observability: reveals internal states to external observers
# Data sources: 
# 1. logs: provide detailed chronological event records
# 2. metrics: offer quantitative system performance measurements
# 3. traces: show the flow of requests across the system components

### Cost Management
# hosting and usage lead to costs
# 1. Choose the modt cost-effective model
# 2. Optimize prompts
# 3. Reduce no. of calls by: batching, response caching

### Implementation of Governance and Security
## Governance includes: policies, guidelines and frameworks

## Security involves measures to prevent:
# -> unauthorized access -> data breaches -> adversarial attacks -> potential misuse of model capabilities

## Access Roles
# Role Based Access Control: permissions are assigned to roles and users are assigned to those roles
# APIs must adhere to security standards and only requests from users with appropriate permissions should be allowed
# use zero trust security model
# application assumes the correct role

## THREAT 1: Prompt Injection:
# attackers manipulate input fields or prompts within an application to execute unauthorized commands or actions
# can lead to defamation etc
# Mitigation: 
# 1. assume inputs can be overridden and contents uncovered
# 2. Treat LLM as untrusted user
# 3. Identify and block known adversarial prompts

## THREAT 2: Output Manipulation
# alters LLM otput
# Mitigations:
# 1. Do not give LLM unnecessary permissions
# 2. censor and block unspecified outputs

## THREAT 3: Denial-of-Service Attacks
# users flood our application with requests, causing substantial cost, availibility and performance issues
# Mitigations:
# 1. limiting request rates
# 2. capping resource usage per request

## THREAT 4: Data Integrity and Poisoning
# injects false or misleading data into training set
# Mitigations:
# 1. use trusted sources only
# 2. use filters
# 3. output censoring