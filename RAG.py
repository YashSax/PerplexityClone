from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieve_documents import retrieve_relevant_documents
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import json


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_vector_store(documents, embedding):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    return vectorstore


def execute_query(query, api_key_file):
    documents, links = retrieve_relevant_documents(query, api_key_file)

    with open("./api_keys.json", "r") as f: 
        api_keys = json.load(f)
        COHERE_API_KEY = api_keys["cohere"]
    embedding_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    vectorstore = create_vector_store(documents, embedding_model)

    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate(input_variables=['context', 'question'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Otherwise, if the question necessitates a lengthy response, break the answer down step-by-step. If the question is simple, keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
    llm = ChatCohere(model="command-r", cohere_api_key=COHERE_API_KEY)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query), links
