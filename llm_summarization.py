import csv
from langchain_community.llms import Ollama
import sys
import os


def read_files_in_directory(directory_path):
    """
    Iterates over all files in a directory and reads their contents.
    
    :param directory_path: str, path to the directory
    :return: dict, a dictionary with filenames as keys and file contents as values
    """
    file_contents = {}
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                file_contents[filename] = file.read()
    
    return file_contents



def append_to_csv(file_path, data_tuple):
    """
    Appends a tuple to the bottom of a CSV file.

    :param file_path: str, path to the CSV file
    :param data_tuple: tuple, data to append to the CSV file
    """
    with open(file_path, mode='a') as file:
        file.write('|'.join(data_tuple) + '\n')


# Check if the user provided an input file
if len(sys.argv) != 3:
    print("Usage: python script.py <input csv> <jd directory>")
    sys.exit(1)

input_file = sys.argv[1]
input_folder = sys.argv[2]


# Initialize the LangChain model and tokenizer
llm = Ollama(model="llama3:70b")
append_to_csv('output.csv',('Srno','Company','Role','Description','Link'))


# Load the CSV file
with open(input_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

files = read_files_in_directory(input_folder)
data = data[1:]
for row in data:
    text_content = files[f'{row[0]}.txt']
    print('_' * 30)
    print('Summarizing',row[0],row[1],row[2])
    print(text_content[:200])
    # Use LangChain to prompt the content and get a response

    if text_content is None or text_content == '':
        print('Skipping...')
        continue

    input_prompt = f"Here is the raw job description. produce a comma separated list of required skills and experience. Do not produce any other response before or after the summary: {text_content}"
    output = llm.predict(input_prompt)

    # Print the output (you can also store it in a new CSV file or database)
    print(output)
    append_to_csv('output.csv',(row[0], row[1], row[2], output, row[3]))
    print('\n\n')