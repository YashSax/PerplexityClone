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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class RAG_with_memory:
    def __init__(self, retriever=None):
        with open("./prompts/search.txt", "r") as f:
            self.raw_prompt_text = f.read()
        with open("./prompts/rewrite_query.txt", "r") as f:
            self.raw_query_rewrite_text = f.read()
        self.retriever = retriever
        self.reset()
    
    def load_history(self, history):
        self.history = ""
        for item in history:
            if item["role"] == "user":
                self.history += "Human: " + item["content"] + "\n"
            else:
                self.history += "AI: " + item["content"] + "\n"
        self.history += "\n"

    def add_retriever(self, retriever):
        self.retriever = retriever
    
    def reset(self):
        with open("./api_keys.json", "r") as f:
            api_keys = json.load(f)
            COHERE_API_KEY = api_keys["cohere"]
        self.llm = ChatCohere(model="command-r", cohere_api_key=COHERE_API_KEY)
        self.history = ""

    def rewrite_query(self, original_query):
        print("Rewriting query")
        prompt_text = self.raw_query_rewrite_text.format(history=self.history, query="{query}")
        prompt_template = PromptTemplate(
            input_variables=['query'],
            template=prompt_text
        )
        chain = (
            {"query" : RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(original_query)

    def generate(self, query):
        # TODO: Generate should take a retriever, not constructor
        if self.retriever is None:
            raise Exception("Retriever must non-None")
        print("Prompting LLM")
        prompt_text = self.raw_prompt_text.format(history=self.history, context="{context}", question="{question}")
        prompt_template = PromptTemplate(
            input_variables=['context', 'question'],
            template=prompt_text
        )
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(query)
        self.history += "Human: " + query + "\n"
        self.history += "AI: " + response + "\n"
        print("Done")
        return response


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_vector_store(documents, embedding):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    return vectorstore

def get_retriever_links(query, api_key_file):
    documents, links = retrieve_relevant_documents(query, api_key_file)

    with open(api_key_file, "r") as f: 
        api_keys = json.load(f)
        COHERE_API_KEY = api_keys["cohere"]
    embedding_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    print("Splitting Documents + Loading into vectorstore")
    vectorstore = create_vector_store(documents, embedding_model)
    return vectorstore.as_retriever(), links
