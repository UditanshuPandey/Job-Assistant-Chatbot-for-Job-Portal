import http.client
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import re
from langchain_community.vectorstores import FAISS
import asyncio
import nest_asyncio
import warnings
from google import generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings('ignore', category=UserWarning, module='torch')

# Load environment variables
load_dotenv()

# Set tokenizers parallelism to false for compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Initialize asyncio event loop
def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Initialize the Gemini API (using the Client approach instead of configure)
genai_api_key = os.getenv('GEMINI_API_KEY')
genai_client = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    api_key=genai_api_key
)

# Initialize the Gemini model using LangChain's integration
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=genai_api_key
)

# Extract PDF text from multiple documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Using FAISS for vector storage
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

# Define conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    # Use Gemini model instead of OpenAI
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=genai_api_key
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=gemini_llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

# Extract job-related keywords from multiple resumes
def extract_job_features(text):
    skills = re.findall(r'\b(Java|Python|Data Science|Machine Learning|Deep Learning|Software Engineer|Data Engineer|AI|NLP|C\+\+|SQL|TensorFlow|Keras)\b', text, re.IGNORECASE)
    titles = re.findall(r'\b(Engineer|Data Scientist|Developer|Manager|Analyst|Consultant)\b', text, re.IGNORECASE)
    features = list(set(skills + titles))
    return features if features else ["General"]

# Get job recommendations from Jooble API based on features
def get_job_recommendations(features):
    host = "jooble.org"
    jooble_api_key = os.getenv("JOOBLE_API_KEY")

    connection = http.client.HTTPConnection(host)
    headers = {"Content-type": "application/json"}
    keywords = ", ".join(features)
    body = json.dumps({"keywords": keywords, "location": "Remote"})
    
    try:
        connection.request("POST", f"/api/{jooble_api_key}", body, headers)
        response = connection.getresponse()
        data = response.read()
        jobs = json.loads(data).get("jobs", [])
        
        job_listings = []
        for job in jobs:
            job_listings.append({
                "title": job.get("title", "Job Title"),
                "company": job.get("company", "Company Name"),
                "link": job.get("link", "#"),
                "description": clean_job_description(job.get("snippet", "Job description not available."))
            })
        return job_listings
    except Exception as e:
        st.error(f"Error fetching job data: {e}")
        return []

# Function to clean and format job description text
def clean_job_description(description):
    description = re.sub(r'&nbsp;|&#39;|<[^>]+>', '', description)
    relevant_info = re.findall(r'\b(?:Python|Java|TensorFlow|Keras|Machine Learning|AI|NLP|Deep Learning|Engineer|Data Scientist|Developer|Analyst)\b', description, re.IGNORECASE)
    for word in relevant_info:
        description = re.sub(r'\b' + re.escape(word) + r'\b', f"**{word}**", description)
    return description

# Function to directly use Gemini for custom queries
def ask_gemini_directly(question, context=""):
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = genai_client.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return "I encountered an error processing your request."

# Handle user question
def handle_userinput(user_question):
    if user_question:
        try:
            if st.session_state.conversation:
                response = st.session_state.conversation.invoke({
                    "question": user_question
                })
                st.write(response.get('answer'))
            else:
                # If no documents are processed yet, use direct Gemini API
                response = ask_gemini_directly(user_question)
                st.write(response)
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")

def main():
    st.set_page_config(page_title="Job Assistant Chatbot",
                      page_icon=":briefcase:")
    
    st.header("Job Assistant Chatbot 💬")
    
    # Initialize conversation if not exists
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    try:
        init_async()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "job_recommendations" not in st.session_state:
            st.session_state.job_recommendations = []

        tab_choice = st.sidebar.radio("Choose a tab", ["Chatbot", "Job Recommendations"])

        if tab_choice == "Chatbot":
            st.header("Chat with Job Assistant :books:")
            user_question = st.text_input("Ask a question about your Resume:")
            if user_question:
                handle_userinput(user_question)

            st.sidebar.subheader("Your documents")
            pdf_docs = st.sidebar.file_uploader("Upload your resumes in PDF form here and click on 'Process'", accept_multiple_files=True)
            if st.sidebar.button("Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        try:
                            raw_text = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)

                            job_features = extract_job_features(raw_text)
                            st.session_state.job_recommendations = get_job_recommendations(job_features)
                            st.success("Documents processed and job recommendations updated.")
                        except Exception as e:
                            st.error(f"Error processing documents: {e}")
                else:
                    st.warning("Please upload PDFs before processing.")

        elif tab_choice == "Job Recommendations":
            st.header("Recommended Jobs 💼")
            if st.session_state.job_recommendations:
                for job in st.session_state.job_recommendations:
                    st.markdown(f"**[{job['title']}]({job['link']})** at **{job['company']}**")
                    st.markdown(f"**Description:** {job['description']}", unsafe_allow_html=True)
                    
                # Add Gemini-powered job insights
                if st.button("Get AI Insights on Job Market"):
                    with st.spinner("Analyzing job market..."):
                        job_titles = [job['title'] for job in st.session_state.job_recommendations]
                        insights = ask_gemini_directly(f"Based on these job titles, provide brief insights about current job market trends: {', '.join(job_titles)}")
                        st.markdown("### AI Market Insights")
                        st.markdown(insights)
            else:
                st.info("Please upload and process your resume in the Chatbot tab to view job recommendations.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == '__main__':
    main()