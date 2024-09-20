import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import fitz
import os
from dotenv import load_dotenv
from io import BytesIO
import docx  # To create a .docx file

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
    # Read the uploaded file once and store the bytes
    file_data = uploaded_file.read()

    # Load PDF from the file bytes
    pdf = fitz.open(stream=file_data, filetype="pdf")
    full_text = ""
    total_pages = len(pdf)  # Get the total number of pages in the PDF

    # Extract text from each page of the PDF
    for page_num in range(total_pages):
        page = pdf.load_page(page_num)  # Load the current page
        full_text += page.get_text("text")  # Append text from each page

    pdf.close()

    # Target number of summary pages as 1/8th of the total PDF pages
    target_summary_pages = max(1, total_pages // 8)
    # Average number of words per page in the summary (assuming ~300 words per page)
    words_per_summary_page = 300
    target_summary_words = target_summary_pages * words_per_summary_page

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
    Extract summary and metadata from the chunk given below:
    {chunk}
    """

    # Function to process each chunk (extract metadata and generate summary)
    def process_chunk(chunk):
        # Summarize the chunk based on the extracted metadata
        summary_prompt = PromptTemplate.from_template(template=summary_prompt_template)
        formatted_summary_prompt = summary_prompt.format(chunk=chunk)
        chunk_summary = llm.predict(formatted_summary_prompt)
        return chunk_summary

    # Process chunks sequentially instead of in parallel
    chunk_summaries = []
    for chunk in chunks:
        chunk_summary = process_chunk(chunk)
        chunk_summaries.append(chunk_summary)

    # Generate a final summary based on all chunk summaries
    st.subheader("Final Summary of the Document")

    final_summary_prompt_template = """
    You are an expert in summarization. Here is the summary of individual chunks of the document to be summarized:
    {chunk_summaries}
    Give a complete and comprehensive summary of the document by using chunk summaries.
    Make sure that the total number of words in the summary does not exceed {target_summary_words} words.
    """

    final_summary_prompt = final_summary_prompt_template.format(
        chunk_summaries="\n".join(chunk_summaries),
        target_summary_words=target_summary_words
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

    # Functionality to download summary as a .txt file
    def download_txt(summary):
        return BytesIO(summary.encode('utf-8'))

    # Functionality to download summary as a .docx file
    def download_docx(summary):
        doc = docx.Document()
        doc.add_paragraph(summary)
        output = BytesIO()
        doc.save(output)
        output.seek(0)
        return output

    # Download buttons for .txt and .docx formats
    st.download_button(
        label="Download Summary as .txt",
        data=download_txt(final_summary),
        file_name="summary.txt",
        mime="text/plain"
    )

    st.download_button(
        label="Download Summary as .docx",
        data=download_docx(final_summary),
        file_name="summary.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
