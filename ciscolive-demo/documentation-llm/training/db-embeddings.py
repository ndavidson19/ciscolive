import torch
import psycopg2
import os
import nltk
from sentence_transformers import SentenceTransformer
from psycopg2 import sql
import re

# Database connection parameters
# TODO: add env variables
DATABASE_HOST = os.environ.get('DATABASE_HOST')
DATABASE_PORT = os.environ.get('DATABASE_PORT')
DATABASE_USER = os.environ.get('DATABASE_USER')
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD')
DATABASE_NAME = os.environ.get('DATABASE_NAME')


def create_database(db_name, host="localhost", port="5432"):
    '''
    Creates a database if one has not been created yet
    For initialization
    '''
    # Connect to the PostgreSQL server
    conn = psycopg2.connect(database="postgres", host=host, port=port)
    conn.autocommit = True  # Enable autocommit mode to execute the CREATE DATABASE command

    # Create a new cursor
    cur = conn.cursor()

    # Execute the CREATE DATABASE command
    try:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"Database {db_name} created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the cursor and connection
        cur.close()
        conn.close()

db_params = {
    'dbname': 'cisco_embeddings',
    'host': '127.0.0.1',
    'port': '5432'
}

create_database('cisco_embeddings', '127.0.0.1', '5432')

 
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def generate_embeddings(sentences, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings


def segment_and_embed_content(file_path):
    '''
    This function does the main preprocessing.
    The idea is our format in cleaned_docs.txt is Source \n Header \n Content \n\n
    This is the logic behind seperating the text:
    1. we want a shortform text without the source and header to give to the model as context
        this reduces context size and unnessessary repition
    2. we want a longform text that includes the source and header for ease of search and for more accurate doc retrieval
    '''
    with open(file_path, 'r') as f:
        lines = f.readlines()

    full_texts = []
    short_texts = []
    i = 0
    while i < len(lines):
        source = lines[i].strip() if i < len(lines) else None
        header = lines[i+1].strip() if i+1 < len(lines) else None
        content = lines[i+2].strip() if i+2 < len(lines) else None

        if source and header and content:
            # Segment the content
            segmented_contents = segment_content(content)
            for segment in segmented_contents:
                full_texts.append(f"{source} - {header}: {segment}")
                short_texts.append(segment)

        i += 4  # increment to skip the empty line and move to the next source

    # Generate embeddings for the full texts 
    embeddings = generate_embeddings(full_texts)

    return full_texts, short_texts, embeddings


def segment_content(content, max_sentences=3):
    '''
    This function works to find step patterns and preserve relative positioning i.e. 1. 2. 3. types of steps
    '''
    # Define step pattern
    step_pattern = r'(?:\d+\.\s|(?<=\s)[a-zA-Z]\.\s)'

    # If the content seems to be composed mainly of steps
    if len(re.findall(step_pattern, content)) > len(content) / 200: # 200 characters assumption
        # Split based on step pattern
        segments = re.split(step_pattern, content)
        segments = [segment.strip() for segment in segments if segment]
        segments = [f"{idx+1}. {segment}" for idx, segment in enumerate(segments)]
    else:
        # Tokenize content into sentences
        sentences = nltk.sent_tokenize(content)
        segments = []
        i = 0

        while i < len(sentences):
            segment = sentences[i:i+max_sentences]
            segments.append(' '.join(segment))
            i += len(segment)

    return segments


#sentences = generate_paragraph_embeddings('./cleaned_file.txt')
full_texts, short_texts, embeddings = segment_and_embed_content('./cleaned_file.txt')

# Initialize a connection without connecting to a particular database
initial_db_params = db_params.copy()
initial_db_params['dbname'] = 'postgres'  # connect to the default 'postgres' database

# Connect to the PostgreSQL instance
conn = psycopg2.connect(**initial_db_params)
conn.autocommit = True
cur = conn.cursor()

# Check if the cisco_embeddings database already exists
cur.execute("SELECT 1 FROM pg_database WHERE datname='cisco_embeddings'")
exists = cur.fetchone()
if not exists:
    cur.execute("CREATE DATABASE cisco_embeddings")

# Close the initial connection
cur.close()
conn.close()

# Connect to the PostgreSQL instance itself
with psycopg2.connect(**db_params) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Create the cisco_embeddings database if it doesn't exist
        cur.execute("SELECT 1 FROM pg_database WHERE datname='cisco_embeddings'")
        exists = cur.fetchone()
        if not exists:
            cur.execute("CREATE DATABASE cisco_embeddings")
            conn.commit()

# Connect to the PostgreSQL database
with psycopg2.connect(**db_params) as conn:
    with conn.cursor() as cur:
        # Create pgvector extension if not exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Drop the existing table (if it exists)
        drop_table_query = """
        DROP TABLE IF EXISTS embeddings;
        """
        cur.execute(drop_table_query)
        
        # Create the embeddings table (if it doesn't exist)
        print(embeddings.shape)
        print(embeddings.shape[1])
        embedding_dim = embeddings.shape[1]  # Assuming embeddings shape is [N, D]
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            id serial PRIMARY KEY,
            full_text TEXT,
            short_text TEXT,
            vector VECTOR({embedding_dim})
        );
        """
        cur.execute(create_table_query)

        # Insert embeddings into the table
        for full_text, short_text, embedding in zip(full_texts, short_texts, embeddings):
            embedding_list = embedding.tolist()
            print(full_text)
            cur.execute("INSERT INTO embeddings (full_text, short_text, vector) VALUES (%s, %s, %s);", (full_text, short_text, embedding_list))

        conn.commit()

print("Embeddings inserted into the database successfully!")
