import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader
import markdown

class DocumentProcessor:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def process_file(self, file_path: str) -> List[Document]:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension.lower() == '.md':
            return self._process_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _process_pdf(self, file_path: str) -> List[Document]:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        return self.text_splitter.create_documents([text], [{"source": file_path}])

    def _process_markdown(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html = markdown.markdown(md_text)
            text = ''.join(html.split())
        return self.text_splitter.create_documents([text], [{"source": file_path}])