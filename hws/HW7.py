
__import__ ('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st 
import chromadb
from openai import OpenAI
import json
import pandas as pd

st.title("HW 7 - Law Firm News Intelligence Bot")
st.write("Ask me anything about news topics or articles!")

# OpenAI Setup
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Column name aliases
CONTENT_ALIASES = ["document", "content", "body", "text", "article", "description", "summary"]
TITLE_ALIASES   = ["title", "headline", "head"]
SOURCE_ALIASES  = ["company_name", "source", "publisher", "outlet", "publication"]
DATE_ALIASES    = ["date", "published_date", "publish_date", "published_at", "pubdate"]
URL_ALIASES     = ["url", "link", "href"]

def find_col(df: pd.DataFrame, aliases: list) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None

# Build ChromaDB from csv
def create_vector_db(df: pd.DataFrame):
    content_col = find_col(df, CONTENT_ALIASES)
    title_col   = find_col(df, TITLE_ALIASES)
    source_col  = find_col(df, SOURCE_ALIASES)
    date_col    = find_col(df, DATE_ALIASES)
    url_col     = find_col(df, URL_ALIASES)

    if not content_col:
        st.error(
            f"Could not find a content column in your CSV. "
            f"Expected one of: {CONTENT_ALIASES}. "
            f"Found: {list(df.columns)}"
        )
        return None

    df = df.dropna(subset=[content_col]).copy()
    df[content_col] = df[content_col].astype(str).str.strip()
    df = df[df[content_col] != ""].reset_index(drop=True)

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
        content = str(row[content_col])
        source  = str(row[source_col]) if source_col else "Unknown"
        date    = str(row[date_col])   if date_col   else "N/A"
        url     = str(row[url_col])    if url_col    else ""
        title   = str(row[title_col])  if title_col  else f"{source} — {date}"

        # Embed company + date + content together for better retrieval
        documents.append(f"{title}\n\n{content}")
        metadatas.append({"title": title, "source": source, "date": date, "url": url, "content": content})
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
    st.session_state.HW7_VectorDB = None
if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = []
if "hw7_csv_name" not in st.session_state:
    st.session_state.hw7_csv_name = None
if "hw7_model" not in st.session_state:
    st.session_state.hw7_model = "gpt-4o-mini"

# Sidebar
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload articles CSV", type=["csv"])

if uploaded_file is not None and uploaded_file.name != st.session_state.hw7_csv_name:
    with st.spinner("Building vector database from uploaded articles..."):
        df = pd.read_csv(uploaded_file)
        st.session_state.HW7_VectorDB = create_vector_db(df)
        st.session_state.hw7_csv_name = uploaded_file.name
        st.session_state.hw7_messages = []  # reset chat on new upload
    if st.session_state.HW7_VectorDB:
        n = st.session_state.HW7_VectorDB.count()
        st.success(f"✅ Loaded **{n} articles** from `{uploaded_file.name}`")

st.sidebar.divider()

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
st.sidebar.write("**Data Source:** Uploaded articles CSV")
st.sidebar.write("**Memory:** Last 10 messages")

st.sidebar.divider()

if st.sidebar.button("Clear Chat"):
    st.session_state.hw7_messages = []
    st.rerun()

# Upload CSV if not taken
if st.session_state.HW7_VectorDB is None:
    st.info("👈 Upload a CSV of news articles in the sidebar to get started.")
    st.stop()

# Chat History Display
for message in st.session_state.hw7_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about the news... e.g. 'Find the most interesting news'"):
    st.session_state.hw7_messages.append({"role": "user", "content": prompt})

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