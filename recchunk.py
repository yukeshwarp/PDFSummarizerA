import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import fitz
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Now retrieve the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("AZURE_OPENAI_API_ENDPOINT")
openai_api_version = os.getenv("OPENAI_API_VERSION")
model_name = os.getenv("MODEL_NAME")
deployment_name = os.getenv("DEPLOYMENT_NAME")

# Initialize AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    model_name=model_name
)

# Streamlit frontend
st.title("PDF Summarizer with Metadata Extraction")

# Upload the PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Recursive Character Text Splitter for chunking the text
    chunk_text_rec = RecursiveCharacterTextSplitter(
        chunk_size=5000,  # Set chunk size based on model input limit
        chunk_overlap=500,  # Overlap for preserving context between chunks
        length_function=len,
        is_separator_regex=False,
    )

    # Read the uploaded file once and store the bytes
    file_data = uploaded_file.read()

    # Load PDF from the file bytes
    pdf = fitz.open(stream=file_data, filetype="pdf")
    full_text = ""

    # Extract text from each page of the PDF
    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)  # Load the current page
        full_text += page.get_text("text")  # Append text from each page

    pdf.close()

    # Split the text into chunks using the recursive character text splitter
    chunks = chunk_text_rec.split_text(full_text)

    # Template for summarization with metadata extraction
    summary_prompt_template = """
    Extract metadata from the chunk and summarize the chunk based on the metadata. The chunk text is given below:
    {chunk}
    """

    # List to store chunk summaries
    chunk_summaries = []

    # Process each chunk sequentially
    for chunk in chunks:
        summary_prompt = PromptTemplate.from_template(template=summary_prompt_template)
        formatted_summary_prompt = summary_prompt.format(chunk=chunk)
        chunk_summary = llm.predict(formatted_summary_prompt)
        chunk_summaries.append(chunk_summary)

    # Generate a final summary based on all chunk summaries
    st.subheader("Final Summary of the Document")

    final_summary_prompt_template = """
    You are an expert in summarization. Here is the summary of individual chunks of the document to be summarized:
    {chunk_summaries}
    Give a complete and comprehensive summary of the document by using chunk summaries.
    """
    
    final_summary_prompt = final_summary_prompt_template.format(
        chunk_summaries="\n".join(chunk_summaries)
    )

    # Get the final document summary from the LLM
    final_summary = llm.predict(final_summary_prompt)

    # Estimate and display the read time
    words = final_summary.split()
    word_count = len(words)
    reading_speed_wpm = 200  # Assuming average reading speed of 200 words per minute
    estimated_read_time = word_count / reading_speed_wpm

    st.write(f"Estimated read time: {estimated_read_time:.2f} minutes")
    st.write(final_summary)
