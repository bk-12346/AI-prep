from openai import OpenAI
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Prompt Engineering
# Create a get_response() function so we don't have to write the code every time
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

# Key Principles:
# use action verbs: write, complete etc
# provide specific and detailed instructions
# limit output length

# Create a request to complete the story
prompt = """Complete the given story in the triple backticks with only 2 paragraphs, in the style of Shakespeare. ```{story}```"""
# Get the generated response
response = get_reponse(prompt)
print("\n Original story: \n", story)
print("\n Generated story: \n", response)

#Prompt Components:
# instructions and output data to operate on

# Structures Outputs:
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

# Conditional Prompts: Have if/else structure

# Create the instructions
instructions = "Infer the language and the number of sentences of the given delimited text by triple backticks; then if the text contains more than one sentence, generate a suitable title for it, otherwise, write 'N/A' for the title."
# Create the output format
output_format = "Include the text, language, number of sentences and title, each on a separate line, and ensure to use 'Text:', 'Language:', and 'Title:' as prefixes for each line."
prompt = instructions + output_format + f"```{text}```"
response = get_response(prompt)
print(response)