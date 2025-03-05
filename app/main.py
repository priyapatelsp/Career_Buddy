import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from utils import clean_text
import fitz
from io import BytesIO

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail and Career Tools")
    st.markdown("Provide applicant details and job URL for various analyses.")

    st.subheader("Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        resume_text = ""
        if resume_file is not None:
            try:
                resume_text = extract_text_from_pdf(resume_file)
            except Exception as e:
                st.error(f"Error extracting text from PDF: {e}")

    with col2:
        linkedin_link = st.text_input("LinkedIn Profile URL")
        linkedin_text = f"LinkedIn profile link provided: {linkedin_link} (Processing not implemented)" if linkedin_link else ""
        if linkedin_link:
            st.warning("LinkedIn processing is a placeholder.")

    with col3:
        profession = st.selectbox("Select Profession",
                                    ("Software Engineer", "IT", "Cloud Engineer", "Data Analyst", "Project Manager"))
        profession_text = f"Profession selected: {profession}"

    job_url = st.text_input("Job URL", value="https://jobs.nike.com/job/R-33460")

    st.subheader("Analysis Options")

    button_col1, button_col2, button_col3, button_col4 = st.columns(4)
    output_area = st.empty() # create empty area for output

    if resume_text and job_url:
        loader = WebBaseLoader([job_url])
        job_data = clean_text(loader.load().pop().page_content)

        with button_col1:
            if st.button("Generate Email"):
                applicant_text = ""
                if resume_text:
                    applicant_text += resume_text + "\n"
                if linkedin_text:
                    applicant_text += linkedin_text + "\n"
                if profession_text:
                    applicant_text += profession_text + "\n"
                try:
                    clean_resume_text = clean_text(applicant_text)
                    jobs = llm.extract_jobs(job_data)
                    for job in jobs:
                        email = llm.write_mail(job, clean_resume_text)
                        # Inject CSS to force full width
                        output_area.markdown(f"""
                        <style>
                        .stCode {{
                            width: 100% !important;
                        }}
                        </style>
                        {st.code(email, language='markdown')}
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        with button_col2:
            if st.button("Match your skills with profile"):
                try:
                    skills_match = chain.match_skills(resume_text, job_data)
                    output_area.markdown(skills_match)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        with button_col3:
            if st.button("My Strengths"):
                try:
                    strengths = chain.my_strengths(resume_text)
                    output_area.markdown(strengths)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        with button_col4:
            if st.button("Most commonly asked questions"):
                try:
                    questions = chain.common_questions(job_data)
                    output_area.markdown(questions)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif not resume_text and job_url:
        st.error("Please upload a resume")
    elif resume_text and not job_url:
        st.error("Please provide a job url")
    elif not resume_text and not job_url:
        st.error("Please upload a resume and job url")

if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email and Career Tools", page_icon="📧")
    create_streamlit_app(chain, clean_text)