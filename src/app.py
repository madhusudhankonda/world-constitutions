import os

import streamlit as st
from country_list import countries_for_language
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from answer import answer

col1,col2,col3,col4,col5,col6= st.columns(6)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is your query?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)    
    
    response = answer(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
with col6:
    st.write("")
    st.button("clear history", type="primary")    
    if st.button:
        st.session_state.messages = []

with st.sidebar:
    all_countries = [country[1] for country in countries_for_language('en')]

    selected_country = st.selectbox("Select country:", all_countries)