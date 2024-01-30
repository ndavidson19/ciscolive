import torch
import psycopg2
import os
from psycopg2 import sql

# Load the embeddings
embeddings = torch.load('embeddings.pt')

# Database connection parameters
# TODO: add env variables
DATABASE_HOST = os.environ.get('DATABASE_HOST')
DATABASE_PORT = os.environ.get('DATABASE_PORT')
DATABASE_USER = os.environ.get('DATABASE_USER')
DATABASE_PASSWORD = os.environ.get('DATABASE_PASSWORD')
DATABASE_NAME = os.environ.get('DATABASE_NAME')

def create_database(db_name, user, password, host="localhost", port="5432"):
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
    'host': '127.0.0.1',
    'port': '5432'
}

create_database('cisco_embeddings', 'postgres', 'secret', '127.0.0.1', '5432')

 
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

sentences = read_text_file('./cisco_docs.txt')
sentences = [s.strip() for s in sentences if s.strip()]

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
        print(embeddings.shape[1])
        embedding_dim = embeddings.shape[1]  # Assuming embeddings shape is [N, D]
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            id serial PRIMARY KEY,
            text TEXT,
            vector VECTOR({embedding_dim})
        );
        """
        cur.execute(create_table_query)

        # Insert embeddings into the table
        for text, embedding in zip(sentences, embeddings):
            embedding_list = embedding.tolist()
            cur.execute("INSERT INTO embeddings (text, vector) VALUES (%s, %s);", (text, embedding_list))

        conn.commit()

print("Embeddings inserted into the database successfully!")
