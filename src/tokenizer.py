import os
import pdb
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from docx import Document
from utils import config, logger

nltk.download('punkt')  # Ensure nltk tokenizer models are available

class TextTokenizer:
    def __init__(self, file_mode='recursive', folder_path=None):
        """
        Initialize the TextTokenizer with a file path.
        :param file_path: Path to the text file to be processed.
        """
        logger.info("Tokenisation started")
        self.folder_path = folder_path
        self.file_mode = file_mode
        self.tokenise_mode = config["chromadb"]["tokenise_mode"]
        self.max_words = config["chromadb"]["max_words"]
        
        if not self.folder_path or not os.path.isdir(self.folder_path):
            raise ValueError('Folder path is invalid')
        
        self.file_path = self.list_files_recursive(self.folder_path)

        logger.info("%s files identified", len(self.file_path))
        self.text, self.metadata = self._read_file()

    
    def list_files_recursive(self, directory):
        """
        List all files in a directory recursively.
        :param directory: Path to the directory.
        :return: List of file paths.
        """
        
        file_list = []
    
        if self.file_mode == 'Recursive':
            # os.walk for recursive directory traversal
            for root, _, files in os.walk(directory):
                for file in files:
                    file_list.append(os.path.join(root, file))
        else:
            # Non-recursive listing: only list files in the specified directory
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):  # Only include files, not directories
                    file_list.append(file_path)
    
        return file_list

    def _read_file(self):
        """
        Read the content of the file based on its extension.
        :return: Text content of the file.
        """
        text = []
        metadata = []
        for file in self.file_path:
            try:
                _, file_extension = os.path.splitext(file)
                base_metadata = {'file_name': os.path.split(file)[-1], 'file_path': os.path.split(file)[0]}
                if file_extension.lower() in ['.txt']:
                    with open(file, 'r', encoding='utf-8') as file:
                        curr_text = self.tokenize(file.read(), base_metadata)
                        curr_metadata = [base_metadata for i in text]
                elif file_extension.lower() in ['.pdf']:
                    curr_text, curr_metadata = self._read_pdf(file, base_metadata)
                elif file_extension.lower() in ['.docx']:
                    curr_text, curr_metadata = self._read_docx_by_pages(file, base_metadata)
                else:
                    logger.warning("Unsupported file type, skipping it: %s", file_extension)
                    continue
                text += curr_text
                metadata += curr_metadata
            except Exception as e:
                logger.error(f"Error {e} while processing file {file}")
        # print(metadata)
        return text, metadata

    def _read_pdf(self, file_path, base_metadata):
        """
        Extract text from a PDF file.
        :return: Combined text from all pages in the PDF.
        """
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        text = self.tokenize(text)
        return text, [base_metadata for i in text]
    
    def _read_docx_by_pages(self, file_path, base_metadata):
        """
        Reads a Word document and splits text into pages based on manual page breaks.
        :param file_path: Path to the .docx file.
        :return: List of strings, each representing the text of a page.
        """
        doc = Document(file_path)
        pages = []
        current_page = []

        for paragraph in doc.paragraphs:
            current_page.append(paragraph.text)
            # Check for manual page break
            if paragraph.text == '\f':
                pages.append('\n'.join(current_page).strip())
                current_page = []

        # Add the last page if it exists
        if current_page:
            pages.append('\n'.join(current_page).strip())

        pages = self.tokenize(pages)
        return pages, [base_metadata for i in pages]

    def tokenize(self, text):
        """
        Tokenize the text based on the specified mode.
        :param mode: Tokenization mode ('sentence', 'paragraph', 'page', 'max_words').
        :param max_words: Maximum number of words per token (only used in 'max_words' mode).
        :return: List of tokens.
        """
        if self.tokenise_mode == 'sentence':
            return self._tokenize_by_sentence(text)
        elif self.tokenise_mode == 'paragraph':
            return self._tokenize_by_paragraph(text)
        elif self.tokenise_mode == 'page':
            return self._tokenize_by_page(text)
        elif self.tokenise_mode == 'max_words':
            if self.max_words is None or not isinstance(self.max_words, int) or self.max_words <= 0:
                raise ValueError("max_words must be a positive integer when mode is 'max_words'.")
            return self._tokenize_by_max_words(text, self.max_words)
        else:
            raise ValueError(f"Unsupported tokenization mode: {self.tokenise_mode}")

    def _tokenize_by_sentence(self, text):
        """
        Tokenize the text by sentences.
        :return: List of sentence tokens.
        """
        if isinstance(text, list):
            text = '/n'.join(text)
        return sent_tokenize(text)

    def _tokenize_by_paragraph(self, text):
        """
        Tokenize the text by paragraphs (split by newline).
        :return: List of paragraph tokens.
        """
        print(text)
        if isinstance(text, list):
            text = '/n'.join(text)
        return [para.strip() for para in text.split('\n') if para.strip()]

    def _tokenize_by_page(self, text):
        """
        Tokenize the text by pages (only applicable for PDF files).
        :return: List of page tokens.
        """
        if not isinstance(text, list):
            raise ValueError("Page tokenization is only supported for PDF files.")
        return text

    def _tokenize_by_max_words(self, max_words):
        """
        Tokenize the text into chunks of a maximum number of words.
        :param max_words: Maximum number of words per token.
        :return: List of word-based tokens.
        """
        if isinstance(text, list):
            text = '/n'.join(text)
        words = text.split()
        return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
