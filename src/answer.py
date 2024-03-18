from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
import config, model
from langchain.prompts import PromptTemplate

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


def get_qa_chain(memory):
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4})
    
    memory = memory
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, verbose = True)

    return qa_chain


def top_5_ques_retriever():
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4})
    qa_chain_ques = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose = True
)
    

    return qa_chain_ques

def top_5_ques():
    qa=top_5_ques_retriever()
    question = "Give me top 10 questions that one could ask from the document"
    result = qa({"query": question})
    return result["result"]




def answer(query,memory):
    qa_chain = get_qa_chain(memory)
    result = qa_chain(query)
    
    return result['answer']

