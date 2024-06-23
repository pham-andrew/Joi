from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import warnings
import re

# Define paths for conversation history
history_file = "conversation_history.json"

def load_conversation():
  """Loads conversation history from a JSON file."""
  try:
    with open(history_file, "r") as f:
      return json.load(f)
  except FileNotFoundError:
    return []

def save_conversation(history):
  """Saves conversation history to a JSON file."""
  with open(history_file, "w") as f:
    json.dump(history, f, indent=4)

llm = LlamaCpp(
  model_path="llama-2-7b-chat.Q8_0.gguf",
  n_gpu_layers=40,
  n_batch=512,
  verbose=False,
)

template = """
History: {history}

[INST] <<SYS>>
You are a female companion and assistant named Joi. You always fulfill the user's requests to the best of your ability.
<</SYS>>

{input}[/INST]
"""
prompt = PromptTemplate(template=template, input_variables=["input", "history", "response"])

# Load conversation history
conversation_history = load_conversation()

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("Joi ready, go ahead!")
while True:
  user_input = input("> ")
  # Update conversation history
  conversation_history.append(f"### HUMAN: {user_input}")

  # Ensure even empty history is passed as a string
  history = "\n".join(conversation_history) if conversation_history else ""

  # Handle empty response for the first iteration
  response = conversation_history[-1].split(": ")[-1] if len(conversation_history) > 1 else ""

  # Call llm_chain.run with keyword arguments
  warnings.filterwarnings("ignore", category=DeprecationWarning) 
  answer = llm_chain.run(input=user_input, history=history, response=response)

  # Remove leading/trailing whitespace first
  cleaned_answer = answer.strip()
  # Find all occurrences of text between asterisks
  matches = re.findall(r"\*\s*(.*?)\s*\*", cleaned_answer)

  # Print any emotes TODO HAVE THIS UPDATE HER PICTURE ON THE GUI FOR A MOMENT
  if matches:
    print(f"Emotes: {', '.join(matches)}")

  # Remove text between asterisks
  cleaned_answer = re.sub(r"\*\s*(.*?)\s*\*", "", cleaned_answer)

  conversation_history.append(f"### JOI: {cleaned_answer}")

  print(cleaned_answer, '\n')

  # Save conversation history on each interaction
  save_conversation(conversation_history)
