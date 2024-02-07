import torch
import psycopg2
import os
import nltk
from sentence_transformers import SentenceTransformer
from psycopg2 import sql
import re

def create_database(db_name, user, password, host="localhost", port="5433"):
    '''
    Creates a database if one has not been created yet
    For initialization
    '''
    # Connect to the PostgreSQL server
    conn = psycopg2.connect(database="postgres", user=user, password=password, host=host, port=port)
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
    'user': 'postgres',
    'password': 'secret',
    'host': 'localhost',
    'port': '5433'
}
 
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def generate_embeddings_for_parts(sources, headers, contents):
    source_embeddings = generate_embeddings(sources)
    header_embeddings = generate_embeddings(headers)
    content_embeddings = generate_embeddings(contents)
    return source_embeddings, header_embeddings, content_embeddings

def generate_embeddings(sentences, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def segment_and_embed_content(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sources = []
    headers = []
    contents = []

    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Source:"):
            source = lines[i].strip()
            sources.append(source)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("Source:"):
                header = lines[i].strip()
                headers.append(header)
                i += 1
                content = []
                while i < len(lines) and not lines[i].strip().startswith("Source:") and not lines[i].strip() == '':
                    content.append(lines[i].strip())
                    i += 1
                contents.append(' '.join(content))
        else:
            i += 1

    return sources, headers, contents

def segment_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    data = []

    i = 0
    while i < len(lines):
        # Debugging: Print the current line being processed
        print(f"Processing line {i}: {lines[i]}")

        # Check for the source
        if lines[i].startswith("Source:"):
            source = lines[i] + " " + lines[i+1]
            print(f"Found source: {source}")
            i += 2  # Move past the source lines
            
            headers_for_current_source = []
            contents_for_current_source = []

            # Continue with headers and their content until the next "Source:"
            while i < len(lines) and not lines[i].startswith("Source:") and len(lines[i].split()) <= 10:
                header = lines[i]
                print(f"Found header: {header}")
                i += 1
                
                content = lines[i] if i < len(lines) else ""  # Ensure we don't go out of index
                print(f"Found content: {content}")
                i += 1

                headers_for_current_source.append(header)
                contents_for_current_source.append(content)
            source_embeddings, header_embeddings, content_embeddings = embed_parts(source, headers_for_current_source, contents_for_current_source)
            data.append({
                'source': source,
                'headers': headers_for_current_source,
                'contents': contents_for_current_source,
                'source_embedding': source_embeddings,  
                'header_embeddings': header_embeddings,  
                'content_embeddings': content_embeddings  
            })
        else:
            i += 1  # Move past any unexpected lines

    return data



def embed_parts(sources, headers, contents):
    return generate_embeddings(sources), generate_embeddings(headers), generate_embeddings(contents)

def insert_into_db(data):
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            # Create pgvector extension if not exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Drop the existing table (if it exists)
            cur.execute("DROP TABLE IF EXISTS embeddings;")
            
            # Create the embeddings table (if it doesn't exist)
            embedding_dim = len(data[0]['source_embedding'])
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id serial PRIMARY KEY,
                source TEXT,
                header TEXT,
                content TEXT,
                source_vector VECTOR({embedding_dim}),
                header_vector VECTOR({embedding_dim}),
                content_vector VECTOR({embedding_dim})
            );
            """
            cur.execute(create_table_query)

            for item in data:
                s = item['source']
                se = item['source_embedding']
                for h, c, he, ce in zip(item['headers'], item['contents'], item['header_embeddings'], item['content_embeddings']):
                    cur.execute(
                        "INSERT INTO embeddings (source, header, content, source_vector, header_vector, content_vector) VALUES (%s, %s, %s, %s, %s, %s);", 
                        (s, h, c, se.tolist(), he.tolist(), ce.tolist())
                    )
            conn.commit()
            print("Successfully inserted embeddings into DB")

def main():
    create_database('cisco_embeddings', 'postgres', 'secret', 'localhost', '5433')

    # 1. Segment the file content
    data = segment_file('./cleaned_file.txt')

    # 3. Insert into the database
    insert_into_db(data)
    
if __name__ == "__main__":
    main()
