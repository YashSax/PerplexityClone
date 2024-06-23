import gradio as gr
from RAG import get_retriever_links, RAG_with_memory
import requests

rag_llm = None
searched = False
DEFAULT_FAVICON_URL = "https://www.nasa.gov/favicon.ico"

def get_favicon_url(link):
    from urllib.parse import urlparse
    domain = urlparse(link).netloc
    url = f"https://{domain}/favicon.ico"
    if favicon_exists(url):
        return url
    return DEFAULT_FAVICON_URL


def favicon_exists(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

def chatbot_response(message, history):
    if not searched:
        return "Please enter a query above before asking follow-up questions."
    return rag_llm.generate(message)

# Function to simulate a search operation
def search(query):
    global rag_llm
    global searched
    searched = True

    retriever, sources = get_retriever_links(query, "./api_keys.json")
    rag_llm = RAG_with_memory(retriever)
    response = rag_llm.generate(query)
    
    sources_html = ''.join([
    f"""
    <a href='{source['link']}' target='_blank' style='display: inline-block; text-decoration: none; color: black; margin-right: 10px; border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
        <div style='display: flex; align-items: center;'>
            <img src='{get_favicon_url(source['link'])}' alt='favicon' style='width: 16px; height: 16px; margin-right: 8px;'>
            <span>{source['title']}</span>
        </div>
    </a>
    """ for source in sources
    ])

    reponse_html = f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{response}</div>"

    return sources_html, reponse_html

# Define the Gradio interface
with gr.Blocks(css=".output-box { background-color: #1c1c1c; color: white; padding: 10px; border-radius: 10px; }") as demo:
    gr.Markdown(
        """
        # LLM-powered Search Engine
        RAG + Langchain-powered Search Engine. Uses the Google Search API to augment Cohere's Command-R LLM.
        """
    )
    
    query_input = gr.Textbox(label="Enter your query", placeholder="Type your search query here...", lines=1)
    search_button = gr.Button("Search")

    sources_box = gr.HTML("<div class='output-box' style='display: none;'><h3>Sources</h3><div id='sources-content'></div></div>", label="Sources")
    response_output = gr.Markdown("<div class='output-box' style='display: none;'><h3>Answer</h3></div>", label="Answer")

    search_button.click(
        fn=search, 
        inputs=query_input, 
        outputs=[sources_box, response_output]
    )

    gr.HTML("""
    <script>
        function showResults() {
            document.querySelectorAll('.output-box').forEach(el => el.style.display = 'block');
        }
        document.querySelector('button').addEventListener('click', showResults);
    </script>
    """)

    gr.Row([sources_box])
    gr.Row([response_output])

    gr.ChatInterface(chatbot_response, title="Follow-up Questions")

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch(share=True)
