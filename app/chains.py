import os
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

# Set up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.5, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
        logger.info("Chain instance created with LLM: %s", self.llm.model_name)


# This extract_jobs method perform following task : 
# Scrape data from the website and
# Extract Job relevant information 

    def extract_jobs(self, cleaned_text):
        logger.info("Extracting jobs from cleaned text.")
        prompt_exact = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE : 
            {page_data}
            ### Instruction:
            The scraped text is from the career page of a website. Your job is to extract the job posting and 
            return them in JSON format in English containing the following keys: 'role', 'skills', and 'description'.
            Only return the valid JSON. Keep answer short and professional and only give one answer that is accurate. Limit anwer to 200 words.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_exact | self.llm
        try:
            res = chain_extract.invoke(input={'page_data': cleaned_text})
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
            logger.info("Job extraction successful.")
        except OutputParserException as e:
            logger.error("Error during job extraction: %s", e)
            raise e
        return res if isinstance(res, list) else [res]
    
# This match_skills method perform following task : 
# Take resume data and job description data and matches them to 
# List matching skills and areas for improvement
    
    def match_skills(self, resume_text, job_data):
        # Implement skill matching logic using llm
        prompt_exact_match_skills = PromptTemplate.from_template(
            """
            ### MATCH THE SKILLS FROM RESUME AND MATCH TO JOB :
            resume - {resume_text} 
            job description -{job_data}
            ### Instruction:
            Provide a list of matching skills and areas for improvement in 2 sections and in bullet points each.
            Add how you can contribute to the company using this skills.
            Limit answer to 200 words only
            Keep answer short and professional and only give one answer that is accurate.
            ### VALID TEXT (NO PREAMBLE):
            """)
        matchSkills_extract = prompt_exact_match_skills | self.llm
        try:
            res = matchSkills_extract.invoke(input={'resume_text': resume_text,'job_data':job_data})
            logger.info("Matched Skills successfully.")
        except OutputParserException as e:
            logger.error("Error during Matching Skills: %s", e)
            raise e
        return res if isinstance(res, list) else [res]
    

# This write_mail method perform following task : 
# Take resume data and job description data and 
# Writes an email

    def write_mail(self, job, resume_text):
        logger.info("Generating email for the job: %s", job.get('role', 'Unknown'))
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION  : 
            {job_description}
            
            ### RESUME  : 
            {resume_text}
            
            ### Instruction:
            Extract the information about the applicant from the resume provided and . Based on the job description and the resume, write a personalized cold email to the company. 
            The email should highlight relevant skills from the resume and show why the applicant is a good fit for the role. 
            Make it short, professional, and tailored to the job description, along with a creative way to gains readers attention mantaining professionalism.
            But don't provide mobile number details in answer.
            ### EMAIL  (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        try:
            res = chain_email.invoke(input={'job_description': str(job), 'resume_text': resume_text})
            logger.info("Email generation successful.")
        except Exception as e:
            logger.error("Error during email generation: %s", e)
            raise e
        return res.content

if __name__ == "__main__":
    logger.info("Starting the Chain module.")
    print(os.getenv("GROQ_API_KEY"))
