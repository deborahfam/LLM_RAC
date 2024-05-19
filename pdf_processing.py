from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdf(file) -> str:
    pdf_reader = PdfReader(file)
    return "".join(page.extract_text() for page in pdf_reader.pages)

def process_file(file, splitter: RecursiveCharacterTextSplitter):
    file_content = read_pdf(file)
    book_documents = splitter.create_documents([file_content])
    return [
        Document(page_content=text.page_content.replace("\n", " ").replace(".", "").replace("-", ""))
        for text in book_documents
    ]

recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
