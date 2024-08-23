from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
import config, model

llm = model.get_llm()
embeddings = model.get_embeddings()

def get_vector_store(country):
    persist_directory = f"./chroma_db_{country.lower()}"
    return Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

def get_qa_chain(country, memory):
    vector_store = get_vector_store(country)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        verbose=True
    )

def answer(query, country, memory):
    qa_chain = get_qa_chain(country, memory)
    result = qa_chain(query)
    return result['answer']

def top_questions_retriever(country):
    vector_store = get_vector_store(country)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        verbose=True
    )

def top_5_ques(country):
    qa = top_questions_retriever(country)
    question = f"Give me top 10 questions that one could ask from the document about {country}"
    result = qa({"query": question})
    return result["result"]
