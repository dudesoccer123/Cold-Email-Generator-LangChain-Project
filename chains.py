import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain: 
    def __init__(self) :
        self.llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
    )
    
    def extract_jobs(self,cleaned_text) :
        prompt_extract = PromptTemplate.from_template(
            '''
            ### SCRAPED TEXT FROM A WEBSITE: 
            {page_data}
            ### INSTRUCTION: 
            The scrapped text is from a career's page of a website.
            your job is to extract the job postings and return them in a JSON format containing the following keys 
            `role`,`experience`,`skills` and `description`,
            Only return a valid JSON.
            ### VALID JSON : (NO PREAMBLE):
            '''
        )

        chain_extract = prompt_extract | self.llm 
        res = chain_extract.invoke(input = {'page_data':cleaned_text})
        try :
            json_parser = JsonOutputParser()
            res= json_parser.parse(res.content) 
        except OutputParserException:
            raise OutputParserException("Content too big. Unable to parse jobs")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
            '''
            ### JOB DESCRIPTION
            {job_description}
            
            ###INSTRUCTIONS:
            You are John, A business development executive at InfraTechADV. InfraTechaDV is an AI and software consulting company dedicated to the seamless
            integration business processes through automated tools.
            Over our experience we have empowered numerous enterprises with tailored soultions fostering scalability, process optimization, cost reduction,
            and hightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of InfraTechADV in fullfilling their needs.
            Also add the most relevant ones from the following links to showcase InfraTechADV's portfolio: {link_list}
            Remember you are John, BDE at InfraTechADV.
            DO NOT provide a preamble..
            ### EMAIL (NO PREAMBLE):  
            
            '''
        )

        chain_email = prompt_email | self.llm 
        res= chain_email.invoke({"job_description":str(job) , "link_list":links})
        return res.content
        


if __name__ == "__main__" :
    print("hello")
    print(os.getenv("GROQ_API_KEY"))