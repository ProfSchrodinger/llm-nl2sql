import os
from flask import render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app
from app.services import (extract_schema, add_to_vector_store, get_sql_from_question, 
                          get_all_databases, get_schema_for_db, delete_database)

# Configure upload folder
UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure vector store folder
VECTOR_STORE_FOLDER = 'data/vector_store'
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/upload-db', methods=['POST'])
def upload_db():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.sqlite'):
        filename = secure_filename(file.filename)
        db_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(db_path)
        
        # Extract schema and add to vector store
        db_id = os.path.splitext(filename)[0]  # Remove .sqlite extension
        schema_text = extract_schema(db_path, db_id)
        add_to_vector_store(db_id, schema_text)
        
        return jsonify({'success': True, 'db_id': db_id, 'schema': schema_text})
    
    return jsonify({'error': 'Only .sqlite files are allowed'}), 400

@app.route('/databases', methods=['GET'])
def get_databases():
    databases = get_all_databases()
    return jsonify(databases)

@app.route('/database/<db_id>', methods=['GET'])
def get_database_schema(db_id):
    """
    Return the formatted schema for the requested database.
    """
    schema = get_schema_for_db(db_id)
    if schema:
        return jsonify({'db_id': db_id, 'schema': schema})
    return jsonify({'error': 'Database not found'}), 404

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    db_id = data.get('db_id', None)  # Optional, if not provided, RAG will find the best match
    
    sql_query, used_db_id, prompt = get_sql_from_question(question, db_id)

    # If an error message was returned, send the appropriate response
    if "not found" in sql_query or "No database" in sql_query:
        return jsonify({'error': sql_query}), 404
    
    return jsonify({
        'question': question,
        'sql': sql_query,
        'db_id': used_db_id,
        'prompt': prompt
    })

@app.route('/database/<db_id>', methods=['DELETE'])
def delete_db(db_id):
    """
    Delete a database from the system (vector store + file).
    """
    success, message = delete_database(db_id)
    if success:
        return jsonify({'success': True, 'message': message})
    return jsonify({'error': message}), 404
