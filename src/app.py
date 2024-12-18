import os
import requests
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
from typing import List, Optional
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain.chains import RetrievalQA

load_dotenv()

def extract_text_from_url(url: str) -> Optional[str]:
    """
    Enhanced text extraction from websites with multiple fallback methods
    """
    try:
        # Method 1: Requests with BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, and navigation elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Extract text from paragraphs and main content areas
        text_elements = soup.find_all(['p', 'article', 'div'])
        
        # Combine text, filtering out very short or empty strings
        texts = [elem.get_text(strip=True) for elem in text_elements 
                 if elem.get_text(strip=True) and len(elem.get_text(strip=True)) > 30]
        
        full_text = " ".join(texts)
        
        # Minimum text length check
        if len(full_text) < 100:
            st.warning("Limited content extracted. The website might have a complex structure.")
        
        return full_text
    
    except requests.RequestException as e:
        st.error(f"Error fetching website content: {e}")
        return None

def get_vectorestore_from_url(url: str):
    """
    Enhanced vector store creation with error handling and fallback mechanisms
    """
    try:
        # Extract text using custom method
        text_content = extract_text_from_url(url)
        
        if not text_content:
            st.error("Could not extract meaningful content from the website.")
            return None
        
        # Create a custom document
        from langchain_core.documents import Document
        document = [Document(page_content=text_content, metadata={"source": url})]
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust based on your needs
            chunk_overlap=200  # Some overlap to maintain context
        )
        document_chunks = text_splitter.split_documents(document)
        
        # Create vector store with error handling
        vector_store = Chroma.from_documents(
            document_chunks, 
            OpenAIEmbeddings(),
            persist_directory="./chroma_db"
        )
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# This functions gives the documents which are relevant to the user query
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Generate a search query to look up in order to get information relevant of the conversation. Otherwise say 'I don't know', dont act smart, give only relevant information")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

#This functions creates documents chain which takes context and answer and combining that with the retriever chain 
def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question as accurately as possible using the below context:\n\n{context}, if you don't know, say 'I don't know', dont give out answe which are not in the website content"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def is_query_relevant(user_query, vector_store):
    """
    Check if the user's query matches relevant content from the website.
    """
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 2})
    relevant_documents = retriever.get_relevant_documents(user_query)
    
    # Check if the retrieved documents have sufficient similarity
    if not relevant_documents:
        return False
    
    # Check if any document has content closely related to the query
    for doc in relevant_documents:
        # You might want to use more sophisticated similarity checking
        if len(doc.page_content) > 50:  # Ensure document has substantial content
            return True
    
    return False

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
st.set_page_config(page_title="WebChat-Assistant", page_icon="🧠")
st.title("WebChat-Assistant")

# Add a message for mobile users
st.info("""
✨ **Using Mobile?**  
To access settings, tap the **menu icon (>)** in the top-left corner.  
Enter the website URL there and start chatting!
""")

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
