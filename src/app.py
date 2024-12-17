import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA

# Replace the existing ChromaDB import with this
client = chromadb.PersistentClient(path="./chroma_db")

chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv()

def get_vectorestore_from_url(url):
    #get the text in docuents form
    loader = WebBaseLoader(url)
    document = loader.load()
    #print("Extracted Documents:", [doc.page_content[:200] for doc in document])
    #split the documents into chunks
    text_spiltter = RecursiveCharacterTextSplitter() #split the text into chunks using recursive character text splitter
    document_chunks = text_spiltter.split_documents(document)
    
    #create a vector store from chuncks
    vector_store = Chroma.from_documents(
        document_chunks, 
        OpenAIEmbeddings(),
        persist_directory="./chroma_db"
        ) #using OpenaiEmbeddings to create vectors
    
    return vector_store

# This functions gives the documents which are relevant to the user query
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conservation, generate a search query to look up in order to get information relevant of the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

#This functions creates documents chain which takes context and answer and combining that with the retriever chain 
def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question as truthfully as possible using the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def is_query_relevant(user_query, vector_store):
    """
    Check if the user's query matches relevant content from the website.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_documents = retriever.get_relevant_documents(user_query)
    
    # If no relevant documents are found, return False
    if not relevant_documents:
        return False
    
    # If relevant documents are found, check that the content is from the website
    # For example, you can check the source URL of the document, or just return True if the document exists
    # Here, we'll assume the document contains useful content related to the website
    return len(relevant_documents) > 0

def get_response(user_query):
    """
    Get the AI response for the user's query from the website's content.
    """
    # First check if the query is relevant to the website's content
    if not is_query_relevant(user_query, st.session_state.vector_store):
        return "Sorry, I don't have information on this topic. Please ask about something related to the website content."

    # If the query is relevant, proceed with retrieving the answer
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)  # Get the context retriever chain
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Generate response based on the website content
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']
    
#app config
st.set_page_config(page_title="Chat with Websites", page_icon="🧠")
st.title("Chat with Websites")

#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL")
    st.markdown("""
    **Instructions:**
    1. Enter the URL of the website you want to chat with.
    2. Make sure the URL starts with `http://` or `https://`.
    3. Press Enter to process.
    """)
    st.markdown("""
    **Quick Tips:**
    - Ensure the URL is publicly accessible
    - Complex websites might have limited extraction
    - Some sites block web scraping
    """)

#main        
if website_url is None or website_url == "":
    st.info("Please Enter the URL")

else:
    #Session state
    if "chat_history" not in st.session_state: #session_state to keep track of the chat history (persistent)
        st.session_state.chat_history =[
            AIMessage(content="Hello! I'm a website chatbot. How can I help you?"),     #default message for the chatbot if using langchain
        ]
       
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorestore_from_url(website_url)
              
    #user_input
    user_query = st.chat_input("Enter your message here.....")
    if user_query is not None and user_query.strip() != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    #conversation to maintain the history of the user and ai instance
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.markdown(f"🧠 **AI:** {message.content}")
        elif isinstance(message, HumanMessage):
            st.markdown(f"🙋‍♂️ **You:** {message.content}")