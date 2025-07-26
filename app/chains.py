import os
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import tenacity

load_dotenv()

# Set up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Chain:
    def __init__(self):
        groq_api_key = st.secrets["GROQ_API_KEY"]
        self.llm = ChatGroq(temperature=0.5, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        logger.info("Chain instance created with LLM: %s", self.llm.model_name)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, max=10))
    def _invoke_llm(self, prompt):
        return self.llm.invoke(prompt)

    def extract_jobs(self, cleaned_text):
        logger.info("Extracting jobs from cleaned text.")
        prompt_exact = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE : 
            {page_data}
            ### Instruction:
            Extract job postings as JSON with 'role', 'skills', 'description'. Limit to 200 words.
            ### VALID JSON:
            """
        )
        try:
            res = self._invoke_llm(prompt_exact.invoke(input={'page_data': cleaned_text}))
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
            logger.info("Job extraction successful.")
        except (OutputParserException, tenacity.RetryError) as e:
            logger.error("Error during job extraction: %s", e)
            raise e
        return res if isinstance(res, list) else [res]

    def match_skills(self, resume_text, job_data):
        logger.info("Matching skills.")
        prompt_match_skills = PromptTemplate.from_template(
            """
            ### RESUME: {resume_text}
            ### JOB: {job_data}
            ### Instruction:
            Match skills, list strengths, and improvements in bullet points. Add how skills contribute. Limit 200 words.
            ### MATCHED SKILLS:
            """
        )
        try:
            res = self._invoke_llm(prompt_match_skills.invoke(input={'resume_text': resume_text, 'job_data': job_data}))
            logger.info("Matched skills successful.")
            return res.content
        except (tenacity.RetryError, Exception) as e:
            logger.error("Error matching skills: %s", e)
            raise e

    def write_mail(self, job, resume_text):
        logger.info("Generating email.")
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB: {job_description}
            ### RESUME: {resume_text}
            ### Instruction:
            Write a personalized cold email, highlight skills, and show fit. No mobile numbers. Limit 200 words.
            Just write one email but that should be precise.
            ### EMAIL WITH GOOD FORMAT :
            """
        )
        try:
            res = self._invoke_llm(prompt_email.invoke(input={'job_description': str(job), 'resume_text': resume_text}))
            logger.info("Email generation successful.")
            return res.content
        except (tenacity.RetryError, Exception) as e:
            logger.error("Error generating email: %s", e)
            raise e

    def my_strengths(self, resume_text):
        logger.info("Analyzing strengths.")
        prompt_strengths = PromptTemplate.from_template(
            """
            ### RESUME: {resume_text}
            ### Instruction:
            Identify strengths in bullet points. Limit 200 words.
            ### STRENGTHS:
            """
        )
        try:
            res = self._invoke_llm(prompt_strengths.invoke(input={'resume_text': resume_text}))
            logger.info("Strengths analysis successful.")
            return res.content
        except (tenacity.RetryError, Exception) as e:
            logger.error("Error analyzing strengths: %s", e)
            raise e

    def common_questions(self, job_data,resume_text):
        logger.info("Generating questions.")
        prompt_questions = PromptTemplate.from_template(
            """
            ### JOB: {job_data}
            ### RESUME: {resume_text}
            ### Instruction:
            Generate interview questions for this JOB for the interviewer with RESUME . Limit 200 words.
            ### QUESTIONS:
            """
        )
        try:
            res = self._invoke_llm(prompt_questions.invoke(input={'job_data': job_data,'resume_text':resume_text}))
            logger.info("Questions generated successfully.")
            return res.content
        except (tenacity.RetryError, Exception) as e:
            logger.error("Error generating questions: %s", e)
            raise e

if __name__ == "__main__":
    logger.info("Starting Chain module.")
    print(os.getenv("GROQ_API_KEY"))
