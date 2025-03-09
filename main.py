import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import pdfplumber

# Use GPT-Neo model instead of LLaMA
model_name = "EleutherAI/gpt-neo-2.7B"  # Change this line to use GPT-Neo or GPT-J
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create the HuggingFace pipeline for text generation
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Wrap the Hugging Face pipeline using HuggingFacePipeline from LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the prompt template and LangChain setup for CV tailoring
prompt_template = """
You are a skilled CV tailor. Based on the following job description and my general CV, tailor my CV to match the job description.
Job Description:
{job_description}

General CV:
{general_cv}

Please ensure that you highlight my relevant experience, skills, and projects that align with the job role. Ensure the CV reads naturally and is ATS-friendly.
"""

template = PromptTemplate(input_variables=["job_description", "general_cv"], template=prompt_template)

# Set up LangChain's LLMChain for CV tailoring
llm_chain = LLMChain(prompt=template, llm=llm)

# Extract text from PDF
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Function to extract keywords from job description
def extract_keywords(job_description):
    nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = nlp(job_description)
    
    # Print the entities to inspect the structure
    print(entities)  # Debugging line to inspect the structure of entities
    
    # Check and extract the correct keywords based on the model's output
    keywords = []
    for entity in entities:
        # In some cases, the key might not be 'entity_group', so let's print and inspect
        if 'entity_group' in entity:
            keywords.append(entity['word'])
        elif 'label' in entity:  # Some models may use 'label' instead of 'entity_group'
            keywords.append(entity['word'])
    
    return keywords

# ATS check function
def ats_check(cv_text, job_description):
    keywords = extract_keywords(job_description)
    missing_keywords = [word for word in keywords if word not in cv_text]
    
    if missing_keywords:
        return f"Missing Keywords: {', '.join(missing_keywords)}"
    else:
        return "CV is ATS-friendly!"

# Streamlit interface
def main():
    st.title("CV Tailoring Tool")

    st.subheader("Upload Your General CV (PDF format)")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    st.subheader("Enter Job Description")
    job_description = st.text_area("Job Description")

    if uploaded_file and job_description:
        # Extract text from uploaded CV
        general_cv = extract_pdf_text(uploaded_file)

        # Extract keywords from the job description
        keywords = extract_keywords(job_description)
        st.subheader("Job Keywords")
        print('key words')
        st.write("Extracted Keywords from Job Description:")
        st.write(keywords)

        # Tailor the CV using LangChain
        tailored_cv = llm_chain.run({"general_cv": general_cv, "job_description": job_description})

        st.subheader("Tailored CV")
        st.text_area("Tailored CV", tailored_cv, height=400)

        # Perform ATS check
        ats_result = ats_check(tailored_cv, job_description)
        st.write(f"ATS Check: {ats_result}")

if __name__ == "__main__":
    main()
