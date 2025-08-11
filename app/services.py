import os
import sqlite3
import torch
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Paths
MODEL_PATH = 'model/best_model.pt'
VECTOR_STORE_PATH = 'data/vector_store'
UPLOADS_PATH = 'data/uploads'

# Ensure directories exist
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(UPLOADS_PATH, exist_ok=True)

# Initialize models
device = torch.device("cpu")  # Use CPU as specified by user
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Load the fine-tuned model if it exists
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

# Initialize the sentence transformer for embeddings
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)

# Initialize FAISS index
embedding_size = 384  # Default size for 'BAAI/bge-small-en-v1.5'
index = faiss.IndexFlatL2(embedding_size)
db_ids = []  # To keep track of database IDs corresponding to vectors

# Load existing index if it exists
index_path = os.path.join(VECTOR_STORE_PATH, 'faiss_index.bin')
db_ids_path = os.path.join(VECTOR_STORE_PATH, 'db_ids.txt')

if os.path.exists(index_path) and os.path.exists(db_ids_path):
    try:
        index = faiss.read_index(index_path)
        with open(db_ids_path, 'r') as f:
            db_ids = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading index: {e}")
        # Initialize new index if loading fails
        index = faiss.IndexFlatL2(embedding_size)
        db_ids = []

def extract_schema(db_path, db_id):
    """
    Extract schema from a SQLite database file and format it in the same way as the training data.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    table_schema_text = []
    
    for table in tables:
        # Get columns and their types
        cursor.execute(f"PRAGMA table_info({table});")
        columns_info = cursor.fetchall()
        
        # Get primary keys
        primary_keys = [col[1] for col in columns_info if col[5] > 0]  # col[5] is the pk flag
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        foreign_keys = cursor.fetchall()
        
        columns = []
        for col in columns_info:
            col_name = col[1]
            col_type = col[2].lower()
            
            # Check if it's a primary key
            pk_marker = " [PK]" if col_name in primary_keys else ""
            
            # Check if it's a foreign key
            fk_info = ""
            for fk in foreign_keys:
                if col_name == fk[3]:  # fk[3] is the column name
                    fk_table = fk[2]  # fk[2] is the referenced table
                    fk_col = fk[4]    # fk[4] is the referenced column
                    fk_info = f" [FK -> {fk_table}.{fk_col}]"
                    break
            
            columns.append(f"{col_name} ({col_type}){pk_marker}{fk_info}")
        
        # Create the table schema text
        if columns:
            cols_text = ", ".join(columns)
            table_schema_text.append(f"Table: {table} ({cols_text})")
    
    conn.close()
    
    # Join all table schemas with newlines
    schema_text = "\n".join(table_schema_text)
    return schema_text

def add_to_vector_store(db_id, schema_text):
    """
    Add a database schema to the vector store.
    """
    global index, db_ids
    
    # Generate embedding for the schema text
    embedding = embedding_model.encode([schema_text])[0]
    embedding = np.array([embedding]).astype('float32')
    
    # Add to FAISS index
    index.add(embedding)
    db_ids.append(db_id)
    
    # Save the updated index and db_ids
    faiss.write_index(index, index_path)
    with open(db_ids_path, 'w') as f:
        for id in db_ids:
            f.write(f"{id}\n")
    
    return True

def get_all_databases():
    """
    Get a list of all available databases.
    """
    return db_ids

def get_schema_for_db(db_id):
    """
    Get the schema text for a specific database ID.
    """
    db_path = os.path.join(UPLOADS_PATH, f"{db_id}.sqlite")
    if os.path.exists(db_path):
        return extract_schema(db_path, db_id)
    return None

def find_most_similar_schema(question):
    """
    Find the most similar schema to the question using the vector store.
    """
    if len(db_ids) == 0:
        return None
    
    # Generate embedding for the question
    question_embedding = embedding_model.encode([question])[0]
    question_embedding = np.array([question_embedding]).astype('float32')
    
    # Search in FAISS index
    D, I = index.search(question_embedding, 1)  # Get the closest match
    
    if I[0][0] < len(db_ids):
        return db_ids[I[0][0]]
    return None

def get_sql_from_question(question, db_id=None):
    """
    Generate SQL from a natural language question.
    """
    
    original_db_id = db_id
    # If db_id is not provided, find the most similar schema
    if db_id is None:
        db_id = find_most_similar_schema(question)
        if db_id is None:
            return "No database found. Please upload a database first.", original_db_id, ""
    
    # Get the schema for the database
    schema_text = get_schema_for_db(db_id)
    if schema_text is None:
        return f"Database '{db_id}' not found.", original_db_id, ""
    
    # Format the input for the model
    input_text = f"translate to SQL: {question} \n{schema_text}"
    
    # Tokenize the input
    input_ids = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    # Generate SQL
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode the prediction
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return sql_query, db_id, input_text



def delete_database(db_id):
    global index, db_ids

    if db_id not in db_ids:
        return False, "Database not found"

    try:
        idx_to_remove = db_ids.index(db_id)
        num_vectors = index.ntotal

        if num_vectors == 1:
            index = faiss.IndexFlatL2(embedding_size)
            db_ids = []
        else:
            new_index = faiss.IndexFlatL2(embedding_size)
            new_db_ids = []
            for i in range(num_vectors):
                if i == idx_to_remove:
                    continue
                vec = index.reconstruct(i)
                new_index.add(np.array([vec]))
                new_db_ids.append(db_ids[i])
            index = new_index
            db_ids = new_db_ids

        faiss.write_index(index, index_path)
        with open(db_ids_path, 'w') as f:
            for _id in db_ids:
                f.write(f"{_id}\n")

        db_file_path = os.path.join(UPLOADS_PATH, f"{db_id}.sqlite")
        if os.path.exists(db_file_path):
            os.remove(db_file_path)

        return True, "Database deleted successfully"

    except Exception as e:
        return False, str(e)