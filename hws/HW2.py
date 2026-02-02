import streamlit as st
from openai import OpenAI
import requests
import anthropic
import io
from bs4 import BeautifulSoup

# Read URL Content
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements for cleaner text
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Summary Prompt
def create_summary_prompt(content, summary_type, language):
    if summary_type == "100 words":
        instruction = "Provide a concise summary in approximately 100 words."
    elif summary_type == "2 connecting paragraphs":
        instruction = "Provide a summary in exactly 2 well-connected paragraphs that flow logically from one to the other."
    else:  # 5 bullet points
        instruction = "Provide a summary as exactly 5 clear and informative bullet points."
    
    return f"Please summarize the following web page content in {language}.\n\n{instruction}\n\nContent:\n{content}"

# OpenAI Summary
def get_openai_summary(prompt, model, api_key):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates clear, accurate summaries."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Claude Summary
def get_claude_summary(prompt, model, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
        system="You are a helpful assistant that creates clear, accurate summaries."
    )
    return response.content[0].text

# Show title and description.
st.title("📄 URL Reader")
st.write(
    "Provide a URL and pick a model for URL summarization"
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Sidebar Options
st.sidebar.header("Summary Options")

# LLM Provider Selection
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider:",
    ["OpenAI", "Claude (Anthropic)"]
)

# Advanced model checkbox
use_advanced = st.sidebar.checkbox("Use advanced model", value=False)

# Set model based on provider and advanced checkbox
if llm_provider == "OpenAI":
    model = "gpt-4o" if use_advanced else "gpt-4o-mini"
    api_key = st.secrets.get("OPENAI_API_KEY")
    key_name = "OPENAI_API_KEY"
else:  # Claude
    model = "claude-sonnet-4-20250514" if use_advanced else "claude-3-haiku-20240307"
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    key_name = "ANTHROPIC_API_KEY"

st.sidebar.write(f"**Current model:** {model}")

# Summary type dropdown
summary_type = st.sidebar.selectbox(
    "Select summary type:",
    ["100 words", "2 connecting paragraphs", "5 bullet points"]
)

# Language selection dropdown
language = st.sidebar.selectbox(
    "Select output language:",
    ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Portuguese"]
)

if not api_key:
    st.error(f"API key not found. Please add {key_name} to your secrets.toml file.")
    st.stop()

# URL input at the top of the screen
url = st.text_input("Enter a URL to summarize:", placeholder="https://example.com/article")

if url:
    # Validate URL format
    if not url.startswith(("http://", "https://")):
        st.warning("Please enter a valid URL starting with http:// or https://")
    else:
        # Generate summary button
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Reading URL content..."):
                content = read_url_content(url)
            
            if content:
                # Truncate content if too long (to avoid token limits)
                max_chars = 15000
                if len(content) > max_chars:
                    content = content[:max_chars]
                    st.info(f"Content truncated to {max_chars} characters to fit model limits.")
                
                st.success(f"Successfully read {len(content)} characters from the URL.")
                
                # Show preview of content
                with st.expander("Preview extracted content"):
                    st.write(content[:2000] + "..." if len(content) > 2000 else content)
                
                # Generate summary
                with st.spinner(f"Generating summary using {model}..."):
                    try:
                        prompt = create_summary_prompt(content, summary_type, language)
                        
                        if llm_provider == "OpenAI":
                            summary = get_openai_summary(prompt, model, api_key)
                        else:
                            summary = get_claude_summary(prompt, model, api_key)
                        
                        st.subheader(f"Summary ({summary_type} in {language}):")
                        st.write(summary)
                        
                    except anthropic.AuthenticationError:
                        st.error("Invalid Anthropic API key. Please check your ANTHROPIC_API_KEY in secrets.")
                    except anthropic.APIError as e:
                        st.error(f"Anthropic API error: {e}")
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
            else:
                st.error("Could not read content from the URL. Please check the URL and try again.")
else:
    st.info("Please enter a URL above to get started.")