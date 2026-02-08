import streamlit as st
from openai import OpenAI
import anthropic
import requests
from bs4 import BeautifulSoup

st.title("HW 3 - URL Chatbot")

#Description of how the chatbot works
st.write("""
### How This Chatbot Works

This chatbot allows you to have a conversation about the content of up to **two URLs**. 

**Features:**
- 🔗 **URL Context:** Enter up to 2 URLs in the sidebar. The content becomes the chatbot's knowledge base.
- 🧠 **Conversation Memory:** Uses a **6-message buffer** (3 user questions + 3 assistant responses). 
  Older messages are removed to stay within limits, but the URL context is always preserved.
- 🤖 **Multiple LLMs:** Choose between OpenAI GPT-4o or Claude Sonnet 4 for responses.

**Memory Implementation:** The system prompt containing URL content is never discarded. 
Only the conversation history is buffered to the last 6 messages to manage token usage efficiently.

---
""")

# Get API keys from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY")

# Function to read content from URL
def read_url_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        # Limit content length to avoid token limits
        return text[:8000] if len(text) > 8000 else text
    except requests.RequestException as e:
        return None

# Function to count tokens (approximate)
def count_tokens(text):
    """Approximate token count: ~4 characters per token."""
    return len(text) // 4

# Function to get response from OpenAI
def get_openai_response(messages, api_key):
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )
    return stream

# Function to get response from Claude
def get_claude_response(messages, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    
    # Extract system prompt and convert messages for Claude
    system_content = ""
    claude_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            claude_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Use streaming with Claude
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_content,
        messages=claude_messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

# --- SIDEBAR OPTIONS ---
st.sidebar.header("⚙️ Configuration")

# LLM Selection (requirement 3)
llm_choice = st.sidebar.selectbox(
    "Select LLM:",
    ["OpenAI GPT-4o", "Claude Sonnet 4"]
)

# Check for required API key
if llm_choice == "OpenAI GPT-4o" and not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to secrets.toml")
    st.stop()
elif llm_choice == "Claude Sonnet 4" and not anthropic_api_key:
    st.error("Anthropic API key not found. Please add ANTHROPIC_API_KEY to secrets.toml")
    st.stop()

st.sidebar.divider()

# URL Inputs (requirement 2)
st.sidebar.header("🔗 URL Inputs")
url1 = st.sidebar.text_input("URL 1:", placeholder="https://example.com/article1")
url2 = st.sidebar.text_input("URL 2 (optional):", placeholder="https://example.com/article2")

# Load URL content button
if st.sidebar.button("Load URL Content", type="primary"):
    st.session_state.url_content = ""
    st.session_state.urls_loaded = []
    
    if url1:
        with st.spinner("Loading URL 1..."):
            content1 = read_url_content(url1)
            if content1:
                st.session_state.url_content += f"\n\n--- Content from URL 1 ({url1}) ---\n{content1}"
                st.session_state.urls_loaded.append(url1)
                st.sidebar.success("✅ URL 1 loaded!")
            else:
                st.sidebar.error("❌ Failed to load URL 1")
    
    if url2:
        with st.spinner("Loading URL 2..."):
            content2 = read_url_content(url2)
            if content2:
                st.session_state.url_content += f"\n\n--- Content from URL 2 ({url2}) ---\n{content2}"
                st.session_state.urls_loaded.append(url2)
                st.sidebar.success("✅ URL 2 loaded!")
            else:
                st.sidebar.error("❌ Failed to load URL 2")
    
    # Clear chat when new URLs are loaded
    st.session_state.messages = []

# Display loaded URLs
if "urls_loaded" in st.session_state and st.session_state.urls_loaded:
    st.sidebar.divider()
    st.sidebar.write("**Loaded URLs:**")
    for i, url in enumerate(st.session_state.urls_loaded, 1):
        st.sidebar.write(f"{i}. {url[:40]}...")

# --- CONVERSATION MEMORY (requirement 4 & 5) ---
MAX_BUFFER_MESSAGES = 6  # 3 user + 3 assistant messages

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "url_content" not in st.session_state:
    st.session_state.url_content = ""
if "urls_loaded" not in st.session_state:
    st.session_state.urls_loaded = []

# Create system prompt with URL content (never discarded)
def get_system_prompt():
    base_prompt = """You are a helpful assistant that answers questions based on the provided URL content. 
Rules:
1. Answer questions using ONLY the information from the provided URL content.
2. If the answer is not in the provided content, say "I don't have that information in the provided URLs."
3. Be conversational and helpful.
4. Provide clear, well-structured answers.
5. If asked about something outside the URL content, politely redirect to the available topics."""
    
    if st.session_state.url_content:
        return f"{base_prompt}\n\nHere is the content from the URLs:\n{st.session_state.url_content}"
    else:
        return base_prompt + "\n\nNo URL content has been loaded yet. Please ask the user to load URLs first."

# Function to get buffered messages
def get_buffered_messages():
    """Returns system prompt + last MAX_BUFFER_MESSAGES from conversation history."""
    buffered = [{"role": "system", "content": get_system_prompt()}]
    
    # Buffer: keep only last 6 messages (3 exchanges)
    if len(st.session_state.messages) > MAX_BUFFER_MESSAGES:
        buffered.extend(st.session_state.messages[-MAX_BUFFER_MESSAGES:])
    else:
        buffered.extend(st.session_state.messages)
    
    return buffered

# --- MAIN CHAT INTERFACE ---

# Check if URLs are loaded
if not st.session_state.url_content:
    st.info("👈 Please enter at least one URL in the sidebar and click 'Load URL Content' to start chatting.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the URL content..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get buffered messages for API call
    buffered_messages = get_buffered_messages()
    
    # Generate response with streaming
    with st.chat_message("assistant"):
        try:
            if llm_choice == "OpenAI GPT-4o":
                stream = get_openai_response(buffered_messages, openai_api_key)
                response = st.write_stream(stream)
            else:  # Claude
                response = st.write_stream(get_claude_response(buffered_messages, anthropic_api_key))
        except Exception as e:
            response = f"Error generating response: {e}"
            st.error(response)
    
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- SIDEBAR INFO ---
st.sidebar.divider()
st.sidebar.header("📊 Chat Stats")
st.sidebar.write(f"**Model:** {llm_choice}")
st.sidebar.write(f"**Messages in history:** {len(st.session_state.messages)}")
st.sidebar.write(f"**Buffer limit:** {MAX_BUFFER_MESSAGES} messages")

if st.session_state.url_content:
    token_estimate = count_tokens(st.session_state.url_content)
    st.sidebar.write(f"**URL content tokens:** ~{token_estimate}")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()