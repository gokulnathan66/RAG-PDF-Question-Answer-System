import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load embedding and model pipeline
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

retriever = None
qa_chain = None

def process_pdf(file):
    global retriever, qa_chain
    try:
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embedding_model)
        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        return "✅ Document uploaded and processed successfully!"
    except Exception as e:
        return f" Failed to process document: {str(e)}"

def answer_question(query):
    if qa_chain is None:
        return " Please upload and process a document first!"
    try:
        result = qa_chain(query)
        answer = result['result']
        source_text = result['source_documents'][0].page_content[:500]
        return f"**Answer**: {answer}\n\n🔍 **Source Snippet**:\n{source_text}"
    except Exception as e:
        return f" Error during QA: {str(e)}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(" RAG-based Document QA System")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
        upload_btn = gr.Button("Process Document")
    upload_output = gr.Textbox(label="Upload Status")

    with gr.Row():
        question = gr.Textbox(label="Ask a Question", placeholder="e.g., What is the main topic?")
        answer = gr.Textbox(label="Answer", lines=10)
        ask_btn = gr.Button("Get Answer")

    upload_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[upload_output])
    ask_btn.click(fn=answer_question, inputs=[question], outputs=[answer])

demo.launch()
