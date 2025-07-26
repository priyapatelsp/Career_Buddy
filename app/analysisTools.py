import os
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Set up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AnalysisTools:
    def __init__(self):
        groq_api_key = st.secrets["GROQ_API_KEY"]
        self.llm = ChatGroq(temperature=0.5, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        logger.info("Chain instance created with LLM: %s", self.llm.model_name)

    

    def my_strengths(self, resume_text):
        # Implement strengths analysis logic using llm
        prompt = f"Analyze the following resume:\n{resume_text}\n and identify the candidate's strengths."
        return self.llm.llm(prompt)

    def common_questions(self, job_data):
        # Implement common questions generation logic using llm
        prompt = f"Based on the job description:\n{job_data}\nGenerate a list of commonly asked interview questions."
        return self.llm.llm(prompt)
    
if __name__ == "__main__":
    logger.info("Starting the AnalysisTools module.")
    print(os.getenv("GROQ_API_KEY"))
