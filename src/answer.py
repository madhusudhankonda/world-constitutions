from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores.chroma import Chroma

import config, model

llm = model.get_llm()
embeddings = model.get_embeddings()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_vector_store():
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings)

    return vector_store

def get_qa_chain():
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4})

    qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()    
    )

    return qa_chain

def answer(query):
    qa_chain = get_qa_chain()
    result = qa_chain.invoke(query)
    
    return result
