from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Falcon model (7B Instruct version for better responses)
model_name = "tiiuae/falcon-7b-instruct"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# Create text generation pipeline
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# Function to extract keywords using Falcon LLM
def extract_keywords_llm(job_description):
    prompt = f"""
    Extract key skills, technologies, and responsibilities from this job description:

    {job_description}

    Keywords (comma-separated):
    """

    response = llm_pipeline(prompt)[0]["generated_text"]

    # Extract only the keywords from the response
    keywords_start = response.find("Keywords (comma-separated):") + len("Keywords (comma-separated):")
    keywords = response[keywords_start:].strip()

    return keywords.split(", ")

# Example job description
job_desc = """We are looking for a Data Scientist skilled in Python, Machine Learning, and NLP.
The candidate must have experience with cloud computing (AWS, GCP) and MLOps.
Experience with transformers and deep learning models like Llama or GPT is a plus."""

# Extract keywords using Falcon
keywords = extract_keywords_llm(job_desc)
print("Extracted Keywords:", keywords)
