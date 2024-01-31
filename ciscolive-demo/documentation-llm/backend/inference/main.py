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

app = Flask(__name__)
CORS(app)

model_7b = "../llm/llama-2-7b.ggmlv3.q4_0.bin"

llm = AutoModelForCausalLM.from_pretrained(model_7b, model_type="llama")

st_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

db_params = {
    'dbname': 'cisco_embeddings',
    'host': 'localhost',
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


def segment_with_steps(sentences):
    '''
    Function to maintain relative positioning of stepwise text i.e. 1. xyz 2. abc 3. etc.
    '''
    segments = []

    i = 0
    while i < len(sentences):
        # If we find a sentence starting with a step (e.g., "1.")
        if re.match(r"^\d+\.", sentences[i].strip()):
            current_step = int(re.match(r"^\d+", sentences[i].strip()).group())
            segment = sentences[i]
            i += 1

            # Check if the current sentence is a continuation of the step
            # by looking ahead for the next step number
            while i < len(sentences) and not (re.match(r"^\d+\.", sentences[i].strip()) and int(re.match(r"^\d+", sentences[i].strip()).group()) == current_step + 1):
                segment += ' ' + sentences[i]
                i += 1

            segments.append(segment.strip())

        else:
            segments.append(sentences[i].strip())
            i += 1

    return segments

def process_user_query(user_query, db_params, st_model, limit):
    '''
    Main function that performs the postprocessing
    '''
    # Convert user query into an embedding
    query_embedding = get_embedding_from_text(user_query, st_model)
    
    # Get top similar sentences
    top_sentences = find_similar_embeddings(query_embedding, db_params)

    # Find other sentences under these headers
    refined_sentences = find_similar_sentences_within_headers(top_sentences, user_query, limit=20)
    
    # Extract headers from these sentences
    headers = extract_and_remove_headers_from_sentences(refined_sentences)

    return headers

def find_similar_embeddings(target_embedding, db_params, limit=50):
    '''
    Finds the embeddings and corresponding text for the user text query from vector db
    '''
    results = []

    # Connect to the PostgreSQL database
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            
            # Query the embeddings based on cosine similarity or L2 distance
            query = """
            SELECT id, full_text, short_text, vector, vector <-> CAST(%s AS vector) AS distance
            FROM embeddings
            ORDER BY vector <-> CAST(%s AS vector)
            LIMIT %s;
            """
            
            cur.execute(query, (target_embedding, target_embedding, limit))
            results = cur.fetchall()

    return results

def find_similar_sentences_within_headers(top_sentences, user_query, limit=10):
    sources, headers, contents = zip(*[split_into_components(entry[1]) for entry in top_sentences])

    print("Sample Sources:", sources[:5])
    print("Sample Headers:", headers[:5])

    header_counts = Counter(headers)
    source_counts = Counter(sources)

    # Calculate cosine similarity scores using TfidfVectorizer
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(contents)
    user_vector = tfidf.transform([user_query])
    cosine_scores = cosine_similarity(user_vector, vectors).flatten()

    # Sort contents based on cosine scores
    sorted_contents = [(score, source, header, content) for score, source, header, content 
                       in sorted(zip(cosine_scores, sources, headers, contents), key=lambda pair: pair[0], reverse=True)]
    
    # Weight the most common headers and sources
    most_common_header = header_counts.most_common(1)[0][0]
    most_common_source = source_counts.most_common(1)[0][0]
    print(f"The most common header is {most_common_header} and source is {most_common_source}")

    weighted_sorted_contents = []
    for score, source, header, content in sorted_contents:
        weight = 1.0
        if source == most_common_source:
            weight += 0.1
        if header == most_common_header:
            weight += 0.1
        weighted_sorted_contents.append((score * weight, content))

    # Sort contents based on weighted cosine scores
    top_responses = [content for _, content in sorted(weighted_sorted_contents, key=lambda pair: pair[0], reverse=True)[:limit]]

    return top_responses

def extract_and_remove_headers_from_sentences(sentences):
    unwanted_phrases = ["For more information,", "Refer to", "See also", "All rights reserved.", "N/A", "© 2023 Cisco and/or its affiliates.", "and later"]  # Add any other unwanted phrases

    processed_sentences = []

    for sentence in sentences:
        # Skip sentences with unwanted phrases
        if any(phrase in sentence for phrase in unwanted_phrases):
            continue

        processed_sentences.append(sentences)

    return processed_sentences

def remove_repetitive_phrases_and_incomplete_sentences(text, shingle_length=5):
    '''
    Shingles: a k-shingle is a consequetive set of k-words i.e. 1 shingle is the same as a bag_of words model
    '''
    # Create shingles from the text
    shingles = [text[i:i+shingle_length] for i in range(len(text) - shingle_length + 1)]
    
    # Detect repetitive shingles
    shingle_counts = {}
    for shingle in shingles:
        shingle_counts[shingle] = shingle_counts.get(shingle, 0) + 1

    # Filter for shingles that occur more than once
    repetitive_shingles = [shingle for shingle, count in shingle_counts.items() if count > 1]
    # Replace instances of repetitive shingles with a single instance
    for shingle in repetitive_shingles:
        while shingle*2 in text:
            text = text.replace(shingle*2, shingle)

    # Split the cleaned text into sentences
    sentences = [sent.strip() for sent in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)]
    
    if sentences and not re.search(r'[.?!]$', sentences[-1]):
        sentences.pop()

    cleaned_text = ' '.join(sentences)

    # If the cleaned text is empty, return a default response or indication
    if not cleaned_text:
        return "Sorry, I couldn't generate a response."

    return cleaned_text




@app.route("/get_message", methods=['POST'])
def get_message():
    if processing_lock.locked():
        return jsonify({"error": "Server is processing another request. Please try again later"}), 503

    with processing_lock:
        user_input = request.json.get('text')
        
        # Convert user input into an embedding
        embedding = st_model.encode(user_input, convert_to_tensor=True)
        
        # Get the top N similar embeddings from the database 
        similar_texts = fetch_similar_embeddings(user_input, limit=10)

        # Formulate a context based on the results
        context = "You are an expert in Cisco documentation. Use the documentation provided to help the user's question:"
        
        # Combine all similar texts into a single context
        seen_texts = set()
        flattened_texts = []

        for sublist in similar_texts:
            for item in sublist:
                if item not in seen_texts:
                    flattened_texts.append(item)
                    seen_texts.add(item)
        
        flattened_texts = segment_with_steps(flattened_texts)

        first_4_texts = flattened_texts[:2]
        sources = ' '.join(first_4_texts)

        # Feed the context and the user query to LLM
        input_text = f"{context} Documentation: {sources} User Question: {user_input}"
        print(input_text)

        # Feed the input_text to LLM and get output 
        output = llm(input_text)
        clean_output = remove_repetitive_phrases_and_incomplete_sentences(output)

        return jsonify({"question":  user_input, "message": clean_output, "source": sources})

if __name__ == '__main__':
    app.run(debug=True)
