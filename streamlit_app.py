import os
import tempfile
import streamlit as st
from pathlib import Path
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure page
st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

def load_pdf(pdf_path):
    """Load a PDF file and return document objects"""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def process_pdfs(uploaded_files, temp_dir):
    """Process uploaded PDF files"""
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to temp directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF
        pdf_docs = load_pdf(file_path)
        documents.extend(pdf_docs)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore, len(documents), len(splits)

class OpenRouterLLM:
    """Custom LLM class for OpenRouter"""
    
    def __init__(self, api_key, model="deepseek/deepseek-r1:free", temperature=0):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.temperature = temperature
        
    def __call__(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:8502",  # Replace with your site URL
                    "X-Title": "PDF RAG System",              # Replace with your site name
                },
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return f"Error generating response: {str(e)}"

def query_rag(vectorstore, question, model_name, temperature, api_key):
    """Query the RAG system"""
    # Define prompt template
    template = """
    You are an AI assistant that answers questions based on the provided context.
    
    Context: {context}
    
    Question: {question}
    
    Provide a comprehensive answer based solely on the context provided.
    If the context doesn't contain the information needed to answer the question, 
    just say "I don't have enough information to answer this question."
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format prompt
    formatted_prompt = prompt.format(context=context, question=question)
    
    # Initialize OpenRouter LLM
    openrouter_llm = OpenRouterLLM(api_key=api_key, model=model_name, temperature=temperature)
    
    # Get response
    response = openrouter_llm(formatted_prompt)
    
    # Return in same format as RetrievalQA for compatibility
    return {
        "result": response,
        "source_documents": docs
    }

def main():
    st.title("📚 PDF RAG System")
    st.markdown("""
    This application allows you to upload PDF files and ask questions about their content.
    The system uses RAG (Retrieval Augmented Generation) to provide accurate answers based on your documents.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenRouter API Key", type="password", 
                              help="Enter your OpenRouter API key")
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        
        # Model settings
        model_name = st.selectbox(
            "Select Model",
            ["deepseek/deepseek-r1:free", "anthropic/claude-3-opus", "google/gemini-pro",
             "meta-llama/llama-3-70b-instruct"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
        
        # File upload section
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            # Create temp directory
            if not st.session_state.temp_dir:
                st.session_state.temp_dir = tempfile.mkdtemp()
            
            if st.button("Process PDFs"):
                if not api_key:
                    st.error("Please enter your OpenRouter API Key")
                else:
                    with st.spinner("Processing PDFs..."):
                        try:
                            # Process PDFs
                            vectorstore, num_docs, num_chunks = process_pdfs(
                                uploaded_files, 
                                st.session_state.temp_dir
                            )
                            
                            st.session_state.vectorstore = vectorstore
                            st.success(f"Processed {num_docs} PDFs into {num_chunks} chunks")
                        except Exception as e:
                            st.error(f"Error processing PDFs: {str(e)}")
    
    # Main content area
    if st.session_state.vectorstore:
        st.header("Ask Questions")
        
        # Question input
        question = st.text_input("Enter your question:")
        
        if question:
            if not api_key:
                st.error("Please enter your OpenRouter API Key in the sidebar")
            else:
                if st.button("Get Answer"):
                    with st.spinner("Generating answer..."):
                        try:
                            # Get answer from RAG system
                            result = query_rag(
                                st.session_state.vectorstore, 
                                question, 
                                model_name, 
                                temperature,
                                api_key
                            )
                            
                            # Display answer
                            st.subheader("Answer")
                            st.write(result["result"])
                            
                            # Display sources
                            if "source_documents" in result:
                                st.subheader("Sources")
                                for i, doc in enumerate(result["source_documents"]):
                                    with st.expander(f"Source {i+1}"):
                                        st.write(f"**Page:** {doc.metadata.get('page', 'Unknown')}")
                                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                        st.write("**Content:**")
                                        st.write(doc.page_content)
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
    else:
        st.info("👈 Upload PDF files in the sidebar and click 'Process PDFs' to get started!")

if __name__ == "__main__":
    main() 
