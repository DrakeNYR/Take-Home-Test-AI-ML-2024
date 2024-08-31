import os
import langchain
# import langchain.chains.conversational_retrieval
import langchain.chains.conversational_retrieval.base
# import langchain.chains.conversational_retrieval.prompts
# import langchain.chains.retrieval_qa
# import langchain.chains.retrieval_qa.base
# import langchain.chains.retrieval_qa.prompt
# import langchain.llms
# import langchain.llms.openai
# import langchain.memory
# import langchain.retrievers
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
import streamlit as st

api_key = "sk-KODx8W-JqEQq471E2C3653ogdbhnfIcoP4cP_eD9X9T3BlbkFJCrcpO9shdswAhJUTVuGJPMo_WsvLnpnTlpqsaJTSkA"
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)

# chat_memory = langchain.memory.ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_custom_data():
    txt_file_path = 'C3 Generative AI/data/customdata.txt'
    loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    index = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(index, embedding=OpenAIEmbeddings())
    
    return vectorstore

vectorstore = load_custom_data()

chain = langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("What can I help you with today?")

if prompt:
    st.chat_message("user").markdown(prompt)
    
    # Append new user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Pass chat history to the chain
    response = chain({"question": prompt, "chat_history": st.session_state.chat_history})
    response_text = response['answer']  # Adjust based on the actual structure of the response

    st.chat_message("assistant").markdown(response_text)
    
    # Append new assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.chat_history.append(AIMessage(content=response_text))

# run the app with command: streamlit run "path\to\chatbot3.py"


