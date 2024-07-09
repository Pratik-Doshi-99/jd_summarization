import csv
import requests
from bs4 import BeautifulSoup
#from langchain_community.llms import Ollama
import sys
import os

# Check if the user provided an input file
if len(sys.argv) != 3:
    print("Usage: python jd_scrape.py <input_csv_file> <output_directory>")
    sys.exit(1)

input_file = sys.argv[1]
output_folder = sys.argv[2]


def append_to_csv(file_path, data_tuple):
    """
    Appends a tuple to the bottom of a CSV file.

    :param file_path: str, path to the CSV file
    :param data_tuple: tuple, data to append to the CSV file
    """
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_tuple)

def save_to_file(file_path, text_content):
    """
    Appends a tuple to the bottom of a CSV file.

    :param file_path: str, path to the CSV file
    :param data_tuple: tuple, data to append to the CSV file
    """
    with open(file_path, mode='w',encoding="utf-8") as file:
        file.write(text_content)


# Load the CSV file
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

# Initialize the LangChain model and tokenizer
#llm = Ollama(model="llama3:70b")
#append_to_csv('output.csv',('Company','Role','Description','Link'))
# Loop through each row in the CSV file
data = data[1:]
for row in data:
    # Extract the URL from the 3rd column
    url = row[3]
    print('Processing',row[0],row[1],url)
    # Visit the URL and extract the HTML content
    response = requests.get(url)
    html_content = response.text
    print('Got response')
    # Extract the text content from the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    print('Extracted Response')
    # Preprocess the text content (e.g. remove extra whitespace)
    text_content = ' '.join(text_content.split())

    print(text_content)
    save_to_file(os.path.join(output_folder,f'{row[0]}.txt'),text_content)
    #print('Summarizing')
    # Use LangChain to prompt the content and get a response
    #input_prompt = f"Here is the raw job description. produce a comma separated list of required skills and experience. Do not produce any other response: {text_content}"
    #output = llm.predict(input_prompt)

    # Print the output (you can also store it in a new CSV file or database)
    #print(output)
    #append_to_csv('output.csv',(row[0],row[1],output,row[2]))