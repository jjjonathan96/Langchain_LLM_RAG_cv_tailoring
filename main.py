import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import pdfplumber
import time

# Load the model
model_name = "EleutherAI/gpt-neo-2.7B"  # Change if using a different open-source model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create the Hugging Face pipeline
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# Wrap the Hugging Face pipeline with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the prompt template
prompt_template = """
You are a skilled CV tailor. Based on the following job description and my general CV, tailor my CV to match the job description.
Job Description:
{job_description}

General CV:
{general_cv}

Ensure that you highlight my relevant experience, skills, and projects. The CV should be ATS-friendly and read naturally.
"""

template = PromptTemplate(input_variables=["job_description", "general_cv"], template=prompt_template)
llm_chain = LLMChain(prompt=template, llm=llm)

# Function to extract text from a PDF
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Avoid NoneType issues
    return text

# Function to extract keywords using Named Entity Recognition (NER)
def extract_keywords(job_description):
    nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    entities = nlp(job_description)
    
    # Extract words from recognized entities
    keywords = [entity['word'] for entity in entities if 'entity_group' in entity or 'label' in entity]
    
    return list(set(keywords))  # Remove duplicates

# ATS check function
def ats_check(cv_text, job_description):
    keywords = extract_keywords(job_description)
    missing_keywords = [word for word in keywords if word not in cv_text]
    
    if missing_keywords:
        return f"Missing Keywords: {', '.join(missing_keywords)}"
    else:
        return "CV is ATS-friendly!"

# Streamlit UI
def main():
    st.title("📄 AI-Powered CV Tailoring")

    # File uploader for CV
    st.subheader("📂 Upload Your General CV (PDF format)")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    # Text input for job description
    st.subheader("📝 Enter Job Description")
    job_description = st.text_area("Paste the job description here")

    if job_description:
        # Extract and display keywords
        with st.spinner("🔍 Extracting keywords..."):
            keywords = extract_keywords(job_description)
            time.sleep(1)  # Simulate processing delay
        print('keyword', keywords)
        st.success("✅ Keywords extracted successfully!")
        st.write("**Extracted Keywords:**", ", ".join(keywords))

    # Add a "Tailor CV" button
    if uploaded_file and job_description:
        if st.button("🎯 Tailor CV"):
            # Extract text from CV
            general_cv = extract_pdf_text(uploaded_file)

            # Progress bar for CV tailoring
            progress_bar = st.progress(0)
            st.write("⏳ Tailoring your CV...")

            for percent_complete in range(1, 101):
                time.sleep(0.02)  # Simulate processing time
                progress_bar.progress(percent_complete)

            # Tailor the CV
            tailored_cv = llm_chain.run({"general_cv": general_cv, "job_description": job_description})

            st.success("✅ CV tailored successfully!")
            st.subheader("📜 Tailored CV")
            st.text_area("Your tailored CV:", tailored_cv, height=400)

            # ATS check
            ats_result = ats_check(tailored_cv, job_description)
            st.subheader("📊 ATS Check")
            st.write(f"📝 {ats_result}")

if __name__ == "__main__":
    main()
