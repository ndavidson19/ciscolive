from flask import Flask, jsonify, request
from ctransformers import AutoModelForCausalLM
from transformers import LlamaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask_cors import CORS, cross_origin
import psycopg2
import nltk
import re

'''
Adding function trying to do module imports
TODO: Fix module imports 
Added repeated functions that should be imported 
'''

def find_similar_sentences_within_headers(top_sentences, user_query, limit=10):
    # Extract headers from the sentences
    headers = [sentence.split(":")[0].strip() for sentence in [entry[1] for entry in top_sentences]]
    
    # Identify the most relevant header based on user query
    tfidf = TfidfVectorizer()
    all_texts = headers + [user_query]
    vectors = tfidf.fit_transform(all_texts)
    
    user_vector = tfidf.transform([user_query])
    header_similarities = [(header, cosine_similarity(user_vector, vectors[i])[0][0]) for i, header in enumerate(headers)]
    header_similarities.sort(key=lambda x: x[1], reverse=True)
    top_header = header_similarities[0][0]
    
    # Filter sentences based on the top header
    relevant_sentences = [sentence for sentence in [entry[1] for entry in top_sentences] if sentence.startswith(top_header + ":")]
    
    # Filter out sentences that are too similar to each other
    vectors = tfidf.fit_transform(relevant_sentences)
    cosine_matrix = cosine_similarity(vectors)

    threshold = 0.85  # Threshold for similarity
    indices_to_keep = set()

    for i in range(len(relevant_sentences)):
        if i not in indices_to_keep:
            indices_to_keep.add(i)
            for j in range(i+1, len(relevant_sentences)):
                if cosine_matrix[i, j] > threshold:
                    indices_to_keep.add(j)

    unique_sentences = [relevant_sentences[i] for i in indices_to_keep]

    # Rank the unique sentences based on their similarity to the user's query
    user_vector = tfidf.transform([user_query])
    sentence_similarities = [(sentence, cosine_similarity(user_vector, vectors[i])[0][0]) for i, sentence in enumerate(unique_sentences)]
    sentence_similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_sentences = [entry[0] for entry in sentence_similarities[:limit]]

    return top_n_sentences


def compute_similarity(query, documents):
    """
    Compute the cosine similarity of the query to the list of documents.
    Returns a sorted list of (index, similarity_score) in descending order of similarity.
    """
    vectorizer = TfidfVectorizer().fit(documents + [query])
    vectors = vectorizer.transform(documents)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, vectors).flatten()
    sorted_similarities = sorted(enumerate(similarities), key=lambda x: -x[1])
    
    return sorted_similarities

def process_user_query(user_query, db_params, st_model, limit):
    # Convert user query into an embedding
    query_embedding = get_embedding_from_text(user_query, st_model)
    
    # Get top similar sentences
    top_sentences = find_similar_embeddings(query_embedding, db_params)

    # Find other sentences under these headers
    refined_sentences = find_similar_sentences_within_headers(top_sentences, user_query, limit=20)
    
    # Extract headers from these sentences
    headers = extract_and_remove_headers_from_sentences(refined_sentences)

    return headers

def extract_and_remove_headers_from_sentences(sentences):
    unwanted_phrases = ["For more information,", "Refer to", "See also", "All rights reserved.", "N/A"]  # Add any other unwanted phrases

    processed_sentences = []

    for sentence in sentences:
        # Skip sentences with unwanted phrases
        if any(phrase in sentence for phrase in unwanted_phrases):
            continue

        # Remove the header
        #_, content = sentence.split(":", 1)
        #content = content.strip()

        processed_sentences.append(sentence)

    return processed_sentences


# Your main function might look something like:


def get_embedding_from_text(text, model):
    return model.encode(text, convert_to_tensor=True).tolist()

def find_similar_embeddings(target_embedding, db_params, limit=50):
    '''
    Finds the embeddings and corresponding text for the user text query from vector db
    '''
    results = []

    # Connect to the PostgreSQL database
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            
            # Query the embeddings based on cosine similarity or L2 distance
            # Adjust the query based on your requirements
            query = """
            SELECT id, text, vector, vector <-> CAST(%s AS vector) AS distance
            FROM embeddings
            ORDER BY vector <-> CAST(%s AS vector)
            LIMIT %s;
            """
            
            cur.execute(query, (target_embedding, target_embedding, limit))
            results = cur.fetchall()

    return results

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#model_13b = "../llm/ggml-llama-2-13b-chat-q4_0.bin"
model_7b = "../llm/ggml-llama-2-7b-chat-q4_0.bin"

llm = AutoModelForCausalLM.from_pretrained(model_7b, model_type="llama")

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

db_params = {
    'dbname': 'cisco_embeddings',
    'user': 'postgres',
    'password': 'secret',
    'host': 'localhost',
    'port': '5432'
}

def fetch_similar_embeddings(query_text, db_params=db_params, model=st_model, limit=5):
    # Connect to db
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    '''
    embedding = get_embedding_from_text(query_text, model)
    similar_entries = find_similar_embeddings(embedding, db_params, limit=limit)

    # Extract the text entries
    similar_texts = [entry[1] for entry in similar_entries]
    '''
    similar_texts = process_user_query(query_text, db_params, st_model, limit=limit)
    # Close connection
    cur.close()
    conn.close()

    return similar_texts


@app.route("/get_message", methods=['POST'])
@cross_origin()
def get_message():
    user_input = request.json.get('text')
    
    # Convert user input into an embedding
    embedding = st_model.encode(user_input, convert_to_tensor=True)
    
    # Get the top N similar embeddings from the database (let's say top 10 for this example)
    similar_texts = fetch_similar_embeddings(user_input, limit=10)

    # Formulate a context based on the results
    context = "You are an expert in Cisco documentation. Use the documentation provided to help the user's question:"
    
    # Combine all similar texts into a single context
    combined_texts = ' '.join(similar_texts)

    # Feed the context and the user query to LLM
    input_text = f"{context} Documentation: {combined_texts} User Question: {user_input}"
    print(input_text)

    # Feed the input_text to LLM and get output (assuming LLM processes the input_text directly)
    output = llm(input_text)

    return jsonify({"message": output, "context": combined_texts})

if __name__ == '__main__':
    app.run(debug=True)
