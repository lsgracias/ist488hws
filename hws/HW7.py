
__import__ ('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st 
import chromadb
from openai import OpenAI
import json
import pandas as pd
import os

st.title("HW 7 - Law Firm News Intelligence Bot")
st.write("Ask me anything about news topics or articles!")

# OpenAI Setup
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Build ChromaDB from csv
def create_vector_db():
    csv_path = "./news.csv"
    
    if not os.path.exists(csv_path):
        st.error("news.csv not found!")
        return None
    
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["Document"]).copy()
    df["Document"] = df["Document"].astype(str).str.strip()
    df = df[df["Document"] != ""].reset_index(drop=True)

    chroma_client   = chromadb.EphemeralClient()
    collection_name = "HW7Collection_v1"

    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name)

    documents  = []
    metadatas  = []
    ids        = []

    for i, row in df.iterrows():
        company = str(row.get("company_name", "Unknown"))
        date = str(row.get("Date", "N/A"))
        url = str(row.get("URL", ""))
        content = str(row["Document"])
        title   = f"{company} — {date}"

        # Embed company + date + content together for better retrieval
        documents.append(f"{title}\n\n{content}")
        metadatas.append({"title": title, "source": company, "date": date, "url": url, "content": content})
        ids.append(f"article_{i}")

    if not documents:
        st.error("No text could be extracted from the CSV.")
        return None
    
    # Generate embeddings in batches using text-embedding-3-small
    embeddings = []
    batch_size = 100

    progress_bar = st.progress(0, text="Generating embeddings...")

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-3-small")
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            progress_bar.progress(
                (i + len(batch)) / len(documents),
                text=f"Generated {i + len(batch)}/{len(documents)} embeddings"
            )
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}: {e}")
            return None

    progress_bar.empty()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return collection

# RAG Function
def search_news(query: str) -> str:
    if not st.session_state.HW7_VectorDB:
        return "The knowledge base is not available right now."

    # 1. Embed the query
    query_response  = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embedding = query_response.data[0].embedding

    # 2. Vector search
    results = st.session_state.HW7_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    # 3. Build context from retrieved articles
    context_text = ""
    if results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            context_text += (
                f"\n\n--- Article {i+1} ---\n"
                f"Title:  {meta['title']}\n"
                f"Source: {meta['source']}\n"
                f"Date:   {meta['date']}\n"
                f"URL:    {meta['url']}\n"
                f"Content:\n{meta['content']}"
            )

    if not context_text:
        return "I couldn't find any relevant articles in the uploaded news data."

    # 4. Synthesize with LLM (using selected model)
    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "You are a professional news analyst for a large global law firm.\n"
                "IMPORTANT RULES:\n"
                "1. Answer ONLY using the articles provided below. Never invent facts.\n"
                "2. For ranking requests, return a numbered list with title, source, date, "
                "and one sentence explaining why the story is noteworthy.\n"
                "3. For topic searches, return all matching articles with a 2-3 sentence summary each.\n"
                "4. If the question cannot be answered from the articles, say so clearly.\n"
                "5. Maintain a professional, concise tone appropriate for a law firm.\n\n"
                f"Here are the relevant news articles:\n{context_text}"
            ),
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=st.session_state.hw7_model,
        messages=synthesis_messages,
    )

    return response.choices[0].message.content

# Tool Creation
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search the uploaded news article knowledge base. "
                "Call this whenever the user asks about news stories, topics, companies, "
                "rankings of interesting articles, or any question answerable from the articles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the news knowledge base.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# Session State
if "HW7_VectorDB" not in st.session_state:
    with st.spinner("Loading news knowledge base..."):
        st.session_state.HW7_VectorDB = create_vector_db()
if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = []
if "hw7_model" not in st.session_state:
    st.session_state.hw7_model = "gpt-4o-mini"

# Sidebar
st.sidebar.header("Settings")

model_choice = st.sidebar.radio(
    "LLM Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
)
st.session_state.hw7_model = "gpt-4o-mini" if "mini" in model_choice else "gpt-4o"
st.sidebar.caption(f"`{st.session_state.hw7_model}`")

st.sidebar.divider()

st.sidebar.header("RAG Info")
st.sidebar.write("**Embedding Model:** text-embedding-3-small")
st.sidebar.write(f"**LLM Model:** {st.session_state.hw7_model}")
st.sidebar.write("**Memory:** Last 10 messages")

st.sidebar.divider()

if st.sidebar.button("Clear Chat"):
    st.session_state.hw7_messages = []
    st.rerun()

# Chat History Display
for message in st.session_state.hw7_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if "pending_query" in st.session_state:
    prompt = st.session_state.pop("pending_query")
else:
    prompt = st.chat_input("Ask about the news... e.g. 'Find the most interesting news'")

if prompt:
    st.session_state.hw7_messsages.append({"role":"user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # System prompt + last 10 messages (short-term memory)
    system_msg = (
        "You are a professional news analyst for a large global law firm. "
        "You have access to a tool called `search_news` that searches a database of uploaded news articles. "
        "Use it whenever the user asks about news stories, topics, rankings, or anything answerable from the articles. "
        "Remember the full conversation to answer follow-up questions naturally."
    )

    messages = [{"role": "system", "content": system_msg}]
    messages += st.session_state.hw7_messages[-10:]  # short-term memory buffer

    # First API call — model decides whether to use the tool
    with st.spinner("Searching news articles..."):
        response = client.chat.completions.create(
            model=st.session_state.hw7_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    response_msg = response.choices[0].message
    tool_calls   = response_msg.tool_calls

    if tool_calls:
        messages.append(response_msg)

        for tc in tool_calls:
            args        = json.loads(tc.function.arguments)
            query       = args.get("query", prompt)
            tool_result = search_news(query)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         tc.function.name,
                "content":      tool_result,
            })

        # Second API call — stream final answer using tool result
        final_response = client.chat.completions.create(
            model=st.session_state.hw7_model,
            messages=messages,
            stream=True,
        )

        with st.chat_message("assistant"):
            final_answer = st.write_stream(final_response)

    else:
        # No tool called — stream direct response
        direct_response = client.chat.completions.create(
            model=st.session_state.hw7_model,
            messages=messages,
            stream=True,
        )
        with st.chat_message("assistant"):
            final_answer = st.write_stream(direct_response)

    st.session_state.hw7_messages.append({"role": "assistant", "content": final_answer})