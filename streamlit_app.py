import streamlit as st
from RAG import get_retriever_links, RAG_with_memory

# In re-write query prompt, could try and get the AI to re-write the query in a more google-search like fashion
# Ex. Instead of "tell me more about the moon landing" say "moon landing"

st.title("RAG Web Search")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    rag_llm = RAG_with_memory()
    if len(st.session_state.messages) > 2:
        print("Rewriting query")
        rag_llm.load_history(st.session_state.messages)
        print(rag_llm.history)
        prompt = rag_llm.rewrite_query(prompt)

    retriever, sources = get_retriever_links(prompt, "./api_keys.json")
    rag_llm.add_retriever(retriever)
    response = rag_llm.generate(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    display_sources = [
        dict(
            title=s["title"],
            link=s["link"],
            snippet=s["snippet"]
        ) for s in sources
    ]

    st.json({
        "LLM Query:" : prompt,
        "Sources" : display_sources
    }, expanded=False)