# PerplexityClone

Chat Version Demo (multi-turn + retrieval for each step + query rewrite): [Google Drive Link](https://drive.google.com/file/d/1LCS4RdyI46WhfJabGGkIrgFJFWZq7cDK/view)

🎥 **Video Walkthrough**:  
[![Watch the demo](https://img.youtube.com/vi/7lQMV2ps50E/0.jpg)](https://www.youtube.com/watch?v=7lQMV2ps50E)

A clone of [perplexity.ai](perplexity.ai), an LLM and RAG-powered search engine using LangChain, Chroma Vector Store, and Cohere Text Generation and Embedding Models. Frontend designed using Gradio.

![image](https://github.com/YashSax/PerplexityClone/assets/46911428/014076e5-fa74-4821-b0b0-2298b3a89205)

### Overview

1) The user's query is passed through the Google Search API to create a corpus of the 10 most recent and relevant webpages.  
2) The documents are split and stored into a Chroma vector store  
3) Using Cohere's text embedding model, relevant snippets are collected using embedding similarity  
4) The relevant snippets are compiled into context for the RAG application  
5) The prompt and context are fed into Cohere's Command-R Text Generation LLM, and the output and documents are presented to the user.
