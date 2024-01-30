from sentence_transformers import SentenceTransformer
import torch
import os
import nltk

nltk.download('punkt')

def list_files_and_contents(dir_path):
    """List all files in the directory and print their contents"""
    # Check if the directory exists
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist.")
        return

    # Check if the given path is actually a directory
    if not os.path.isdir(dir_path):
        print(f"{dir_path} is not a directory.")
        return

    # List files in the directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            print(f"\n==== Contents of {filename} ====\n")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                print(file.read())


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def generate_embeddings(sentences, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def generate_embedded_sentences_with_headers(file_path):
    # Read the lines from the text file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = None
    sentences_to_embed = []
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        # Assuming headers don't contain periods (.), but this criterion might change based on your data
        if '.' not in line:
            header = line
        else:
            # Tokenize the long text into sentences
            for sentence in nltk.sent_tokenize(line):
                sentences_to_embed.append(f"{header} {sentence}")

    # Generate embeddings for the list of sentences with headers
    embeddings = generate_embeddings(sentences_to_embed)

    return embeddings

def main():
    # Generate embeddings
    embeddings = generate_embedded_sentences_with_headers('./docs/cisco_docs.txt')
    
    # Save embeddings
    torch.save(embeddings, 'embeddings.pt')

'''

def main():
    # Load your text file
    sentences = read_text_file('./docs/cisco_docs.txt')
    #sentences = [s.strip() for s in sentences if s.strip()]


    # Generate embeddings
    embeddings = generate_embeddings(sentences)

    # Save embeddings if needed
    torch.save(embeddings, 'embeddings.pt')

    #list_files_and_contents("./docs/pdfs")
'''

if __name__ == '__main__':
    main()
