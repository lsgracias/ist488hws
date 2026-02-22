__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb
import os
import json
from bs4 import BeautifulSoup

st.title("🏫 HW5 - Syracuse University Student Org Chatbot")
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

    for tag in soup(['script', 'style']):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)
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

    if not os.path.exists(html_dir):
        st.error(f"Directory {html_dir} not found.")
        return None

    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]

    if not html_files:
        st.error(f"No HTML files found in {html_dir}.")
        return None

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "HW5Collection_v1"

    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    documents = []
    metadatas = []
    ids = []

    for filename in html_files:
        file_path = os.path.join(html_dir, filename)
        try:
            full_text = extract_text_from_html(file_path)
            chunks = chunk_text(full_text, chunk_size=1000, overlap=200)

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{filename}_chunk_{i}")
                metadatas.append({"filename": filename, "chunk_id": i})

        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    if not documents:
        st.error("No text extracted from HTML files.")
        return None

    embeddings = []
    batch_size = 100

    progress_bar = st.progress(0, text="Generating embeddings...")

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch_docs, model="text-embedding-3-small")
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            progress_bar.progress(
                (i + len(batch_docs)) / len(documents),
                text=f"Generated {i + len(batch_docs)}/{len(documents)} embeddings"
            )
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}: {e}")
            return None

    progress_bar.empty()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return collection

def relevant_club_info(query):
    if not st.session_state.HW5_VectorDB:
        return "The knowledge base is not available right now."

    # 1. Embed the query
    query_response = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = query_response.data[0].embedding

    # 2. Vector search
    results = st.session_state.HW5_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # 3. Build context from retrieved chunks
    context_text = ""
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            filename = results['metadatas'][0][i]['filename']
            context_text += f"\n\n--- Document Snippet {i+1} (Source: {filename}) ---\n{doc}"

    if not context_text:
        return "I couldn't find relevant information about that in the student organization materials."

    # 4. Call LLM with retrieved context (no tool calling at this level)
    synthesis_messages = [
        {
            "role": "system",
            "content": f"""You are a helpful Syracuse University student engagement assistant.
IMPORTANT RULES:
1. If you find relevant information, clearly state which student organization HTML it comes from.
2. When using information from the context, mention that you found it in the student organization materials.
3. If the answer to the question isn't provided in the context, use your general knowledge but state that this wasn't from the student organization material.
4. Be helpful, clear, and concise.

Here are the relevant course documents:
{context_text}"""
        },
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=synthesis_messages,
    )

    return response.choices[0].message.content

# Tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the Syracuse University student organization knowledge base. "
                "Call this when the user asks about clubs, organizations, membership, "
                "meetings, officers, contact info, or activities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the knowledge base.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Initialize Vector Database
if "HW5_VectorDB" not in st.session_state:
    with st.spinner("Creating vector database from HTML files..."):
        st.session_state.HW5_VectorDB = create_vector_db()

# Initialize chat history (short-term memory buffer)
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = []

# Sidebar Info
st.sidebar.header("RAG Info")
st.sidebar.write(f"**Embedding Model:** text-embedding-3-small")
st.sidebar.write(f"**LLM Model:** gpt-4o-mini")
st.sidebar.write(f"**Data Source:** Student org HTML pages")
st.sidebar.write(f"**Memory:** Last 10 messages")

st.sidebar.divider()

if st.sidebar.button("Clear Chat"):
    st.session_state.hw5_messages = []
    st.rerun()

# Main Chat Interface 
for message in st.session_state.hw5_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about SU student organizations..."):
    # Add user message to history and display it
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages: system prompt + last 10 messages (short-term memory) + new user message
    system_msg = (
        "You are a helpful Syracuse University student engagement assistant. "
        "You have access to a tool called `relevant_club_info` that searches a database of SU student organizations. "
        "Use it whenever the user asks about organizations, membership, meetings, officers, or activities. "
        "Remember the full conversation to answer follow-up questions naturally."
    )

    messages = [{"role": "system", "content": system_msg}]
    messages += st.session_state.hw5_messages[-10:]  # short-term memory buffer

    # First API call — model decides whether to use the tool
    with st.spinner("Searching student org documents..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    response_msg = response.choices[0].message
    tool_calls = response_msg.tool_calls

    if tool_calls:
        # Append assistant's tool-call request to messages
        messages.append(response_msg)

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            query = args.get("query", prompt)
            tool_result = relevant_club_info(query)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": tool_result,
            })

        # Second API call — produce final answer using tool result
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            final_answer = st.write_stream(final_response)

    else:
        # No tool called — stream direct response
        messages.append({"role": "user", "content": prompt})
        direct_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        with st.chat_message("assistant"):
            final_answer = st.write_stream(direct_response)

    # Add assistant response to history
    st.session_state.hw5_messages.append({"role": "assistant", "content": final_answer})