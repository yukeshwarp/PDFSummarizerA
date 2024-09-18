import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import fitz
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Now retrieve the environment variables
api_type = os.getenv("OPENAI_API_TYPE")
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

# Initialize AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model_name="gpt-4o-mini"
)

# Streamlit frontend
st.title("PDF Summarizer with Metadata Extraction")

# Upload the PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
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

    # Split the PDF content into chunks with overlapping (chunk_size=4000, overlap_size=500)
    def chunk_text(full_text, chunk_size=5000, overlap_size=500):
        paragraphs = full_text.split("\n\n")
        current_chunk = ""
        chunks = []
        overlap = ""  # Initialize overlap variable

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                chunks.append(current_chunk.strip())
                # Store the last 'overlap_size' characters to be used in the next chunk
                overlap = current_chunk[-overlap_size:]
                # Start the new chunk with the overlapping text
                current_chunk = overlap + paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # Chunk the text with a defined overlap
    chunks = chunk_text(full_text, chunk_size=5000, overlap_size=500)

    # Template for summarization
    summary_prompt_template = """
    Extract metadata from the chunk and summarize the chunk based on the metadata. The chunk text is given below:
    {chunk}
    """

    # Function to process each chunk (extract metadata and generate summary)
    def process_chunk(chunk):
        # Summarize the chunk based on the extracted metadata
        summary_prompt = PromptTemplate.from_template(template=summary_prompt_template)
        formatted_summary_prompt = summary_prompt.format(chunk=chunk)
        chunk_summary = llm.predict(formatted_summary_prompt)
        return chunk_summary

    # Use ThreadPoolExecutor to process chunks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        chunk_summaries = list(executor.map(process_chunk, chunks))

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
