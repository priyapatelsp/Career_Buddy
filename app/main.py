from rich import _console
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from utils import clean_text
import fitz  # PyMuPDF to extract text from PDF
import streamlit as st
import fitz  # PyMuPDF for PDF processing
from io import BytesIO

# Function to extract text from PDF (if using Streamlit file uploader)
def extract_text_from_pdf(uploaded_file):
    # Ensure the uploaded file is a BytesIO object
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail Generator")

    # User Inputs: Resume (PDF) and Job URL
    resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    job_url = st.text_input("Enter the Job URL:", value="https://jobs.nike.com/job/R-33460")
    
    submit_button = st.button("Submit")

    if submit_button:
        if resume_file is not None and job_url:
            try:
                # Extract text from the uploaded resume PDF
                resume_text = extract_text_from_pdf(resume_file)
                clean_resume_text = clean_text(resume_text)
                
                # Process the job URL to extract job information
                loader = WebBaseLoader([job_url])
                job_data = clean_text(loader.load().pop().page_content)
                
                # Extract jobs and generate emails
                jobs = llm.extract_jobs(job_data)
                for job in jobs:
                    email = llm.write_mail(job, clean_resume_text)
                    st.code(email, language='markdown')

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload a resume (PDF) and provide a job URL.")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, clean_text)
