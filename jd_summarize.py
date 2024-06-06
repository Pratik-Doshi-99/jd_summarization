import csv
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
import sys

# Check if the user provided an input file
if len(sys.argv) != 2:
    print("Usage: python script.py <input_csv_file>")
    sys.exit(1)

input_file = sys.argv[1]

# Load the CSV file
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Initialize the LangChain model and tokenizer
llm = Ollama(model="llama3:70b")

# Loop through each row in the CSV file
for row in data:
    # Extract the URL from the 3rd column
    url = row[2]

    # Visit the URL and extract the HTML content
    response = requests.get(url)
    html_content = response.text

    # Extract the text content from the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()

    # Preprocess the text content (e.g. remove extra whitespace)
    text_content = ' '.join(text_content.split())

    # Use LangChain to prompt the content and get a response
    input_prompt = f"Here is the raw job description. produce a comma separated list of required skills and experience: {text_content}"
    output = llm.predict(input_prompt)

    # Print the output (you can also store it in a new CSV file or database)
    print(output)