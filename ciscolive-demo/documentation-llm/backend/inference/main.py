from flask import Flask, request, jsonify, make_response
from llama_cpp import Llama
from transformers import LlamaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from sentence_transformers import SentenceTransformer
from flask_cors import CORS, cross_origin
import psycopg2
from collections import Counter

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../llm/rocket-3b.Q4_K_M.gguf")
args = parser.parse_args()

llm = Llama(model_path=args.model, chat_format="chatml", n_ctx=2048)

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

db_params = {
    'dbname': 'cisco_embeddings',
    'user': 'postgres',
    'password': 'secret',
    'host': 'db',
    'port': '5432'
}
# TODO: Move to /util file

def split_into_components(full_text):
    ''' 
    Takes in the full_text ({source - header: content}) and splits it
    full_text: str
    '''
    # Splitting the full text into source, header, and content
    try:
        source_and_rest, content = full_text.split(':', 1)
        source, rest_of_header = source_and_rest.rsplit(' - ', 1)  # Use rsplit to split from the rightmost instance
        header = rest_of_header.rsplit(' - ', 1)[0]
    except ValueError:
        # Handle texts that do not conform to the expected pattern
        source, header, content = None, None, full_text

    return source, header, content

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

def get_embedding_from_text(text, model):
    '''
    Uses sentence transformers to encode text to embeddings for similiarity search.
    text:
    model:
    '''
    return model.encode(text, convert_to_tensor=True).tolist()

def fetch_similar_embeddings(query_text, db_params=db_params, model=st_model, limit=5):
    '''
    Function connects to DB and starts our post-processing pipeline.
    Calls process_user_query function
    '''
    # Connect to db
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    similar_texts = process_user_query(query_text, db_params, st_model, limit=limit)
    # Close connection
    cur.close()
    conn.close()

    return similar_texts


def process_user_query(user_query, db_params, st_model, limit):
    '''
    Main function that performs the postprocessing
    '''
    # Convert user query into an embedding
    query_embedding = get_embedding_from_text(user_query, st_model)
    
    # Get top similar sentences
    top_sentences = find_similar_embeddings(query_embedding, db_params)

    return top_sentences

def find_similar_embeddings(target_embedding, db_params, limit=2):
    '''
    Finds the embeddings and corresponding text for the user text query from vector db
    '''
    results = []

    # Connect to the PostgreSQL database
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            
            # Query the embeddings based on cosine similarity or L2 distance
            query = """
            SELECT id, source, header, content, content_vector <-> CAST(%s AS vector) AS content_distance
            FROM embeddings
            ORDER BY content_distance ASC
            LIMIT %s;
            """
            
            cur.execute(query, (target_embedding, limit))
            results = cur.fetchall()

            # Select only the top 2 results 
            results = results[:1]
            
    return [{'id': result[0], 'source': result[1], 'header': result[2], 'content': result[3]} for result in results]



@app.route("/get_message", methods=['POST', 'OPTIONS'])
def get_message():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all domains, adjust if necessary
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'  # Add any headers your POST requests need
        return response
    
    if request.method == 'POST':
        user_input = request.json.get('text')
        
        # Get the top N similar embeddings from the database 
        similar_texts = fetch_similar_embeddings(user_input, limit=10)

        # Formulate a context based on the results
        context = "You are an expert in Cisco documentation. Use the documentation provided to help the user's question:"

        # Feed the context and the user query to LLM
        input_text = f"{context} Documentation: {similar_texts} User Question: {user_input}"
        print(input_text)

        # Feed the input_text to LLM and get output 
        output = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": "You are an assistant who perfectly helps with answering questions with the provided documentation."},
                    {
                        "role": "user",
                        "content": input_text
                    }
                ]
            )
        answer = output['choices'][0]['message']['content']

        print(output)
        return jsonify({"question":  user_input, "message": answer, "context": "ND Release Notes"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
