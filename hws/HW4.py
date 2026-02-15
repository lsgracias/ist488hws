__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import os
from bs4 import BeautifulSoup

st.title("🏫 HW4 - Syracuse University Student Org Chatbot")
st.write("Ask me anything about Syracuse University student organizations!")

# Get API key from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Extract text from HTML file
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Remove script and style elements so we only get visible text
    for tag in soup(['script', 'style']):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)

    # Clean up excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)

# Helper function for chunking text
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# Function to create ChromaDB collection
def create_vector_db():
    html_dir = "./hw4htmls"
    
    # Check if directory exists
    if not os.path.exists(html_dir):
        st.error(f"Directory {html_dir} not found.")
        return None

    # Retrieve all PDF files
    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    
    if not html_files:
        st.error(f"No PDF files found in {html_dir}.")
        return None
        
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create or get collection
    # We use a new name to force a fresh start if the code changes
    collection_name = "HW4Collection_v2" 
    
    try:
        # Try to delete if it exists to ensure freshness (optional, but good for dev)
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    
    documents = []
    metadatas = []
    ids = []

    id_counter = 0
    
    # Process each HTML
    for filename in html_files:
        file_path = os.path.join(html_dir, filename)
        try:
            full_text = extract_text_from_html(file_path)
            # Chunk the text
            chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
                
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{filename}_chunk_{i}") # Unique ID per chunk
                metadatas.append({"filename": filename, "chunk_id": i})
                    
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    if not documents:
        st.error("No text extracted from HTML files.")
        return None

    # Generate embeddings using batches to avoid API limits
    embeddings = []
    batch_size = 100
    
    progress_bar = st.progress(0, text="Generating embeddings...")
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch_docs, model="text-embedding-3-small")
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            progress_bar.progress((i + len(batch_docs)) / len(documents), text=f"Generated {i + len(batch_docs)}/{len(documents)} embeddings")
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}: {e}")
            return None # Stop if embedding fails
            
    progress_bar.empty()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

# --- Initialize Vector Database ---
if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Creating vector database from HTML files..."):
        st.session_state.HW4_VectorDB = create_vector_db()

# Initialize chat history
if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = []

# --- Sidebar Info ---
st.sidebar.header("RAG Info")
st.sidebar.write(f"**Embedding Model:** text-embedding-3-small")
st.sidebar.write(f"**LLM Model:** gpt-4o-mini")
st.sidebar.write(f"**Data Source:** Student org HTML pages")

st.sidebar.divider()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.hw4_messages = []
    st.rerun()

# --- Main Chat Interface ---

# Display chat history
for message in st.session_state.hw4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the courses..."):
    # Add user message to history
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Retrieve relevant documents
    context_text = ""
    retrieved_docs = []

    if st.session_state.HW4_VectorDB:
        with st.spinner("Searching course documents..."):
            query_response = client.embeddings.create(input=prompt, model="text-embedding-3-small")
            query_embedding = query_response.data[0].embedding
            
            results = st.session_state.HW4_VectorDB.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    filename = results['metadatas'][0][i]['filename']
                    context_text += f"\n\n--- Document Snippet {i+1} (Source: {filename}) ---\n{doc}"
                    retrieved_docs.append(filename)

    system_prompt = f"""You are a helpful Syracuse University student engagement assistant.
IMPORTANT RULES:
1. If you find relevant information, clearly state which student organization HTML it comes from.
2. Be helpful, clear, and concise.
3. When using information from the context, mention that you found it in the student organization materials.
4. If the answer to the question isn't provided in the context, use your general knowledge but state that this wasn't from the student organization material.

Here are the relevant course documents:
{context_text}
"""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add current query
    messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Add assistant response to history
    st.session_state.hw4_messages.append({"role": "assistant", "content": response})