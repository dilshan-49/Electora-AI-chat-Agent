#"mongodb+srv://shehanperera903:cP4weZs4pCdPfKKV@cluster0.5dvap.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json
import re

NVIDIA_API = "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"

# Initialize the ChatNVIDIA client
client = ChatNVIDIA(
    model="meta/llama-3.1-405b-instruct",
    api_key=NVIDIA_API, 
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

def chat_generate(user_input,problem_data):

    # Initialize the conversation history
    conversation_history = []
    pd = problem_data
    while True:
        #user_input = input("You: ")
        #if user_input.lower() == 'exit':
            #print("Chatbot: Goodbye!")
           # break

        # Add user input to the conversation history
        conversation_history.append({"role": "user", "content": user_input})
        text = "{user_input}{pd}try to use conversation history to answer the user inputs {conversation_history} "
        prompt = text.format(user_input=user_input,pd=pd,conversation_history=conversation_history)
        # Get response from the chatbot
        response = ""
        for chunk in client.stream(prompt):
            response += chunk.content
        
        # Add chatbot response to the conversation history
        conversation_history.append({"role": "ai", "content": response})

        return response

def calc_weight(user_input,problem_data):
   
    response = generate(user_input,problem_data)
    print("response"+ response)
    response = response.strip()  # Remove leading/trailing spaces
    response = response.replace("```python", "").replace("```", "").replace("'", '"') # Replace single quotes with double quotes for valid JSON

    
    try:
        json_string = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        print("json=== "+json_string)
        # Attempt to parse the result as JSON
        response_dict = json.loads(json_string)
        print(type(response_dict))
    except json.JSONDecodeError as e:
        # If there's a parsing error, print the error and return a default value
        print(f"Error parsing JSON: {e}")   

        response_dict = {}
    #response = json.loads(response)
    return response_dict

def generate(user_input,problem_data):
    text = "{user_input}{pd}  "
    prompt = text.format(user_input=user_input,pd=problem_data)
    # Get response from the chatbot
    response = ""
    for chunk in client.stream(prompt):
        response += chunk.content
    return response