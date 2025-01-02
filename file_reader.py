# import PyPDF2
from typing import List, Dict
from langchain.schema import Document
import re
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FileReader:
    def __init__(self, directory: str):
        self.directory = directory
        self.lines = []
        self.chunks = []
    
    def read_files(self) -> List[Dict[str, str]]:
        self.lines = partition( self.directory)
   
     
        
    def create_chunks(self,replace_str):
        entire_text = ''
        for e in self.lines:
            if not self.contains_only_special_chars(str(e)) and not re.search(r'^-?\d+$',str(e)) :
                for s in replace_str:
                    e = re.sub(s.lower(),'',str(e).lower())
                if e!='' and e!=' ':
                    entire_text+=e+' '

        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=3000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        self.chunks = text_splitter.create_documents([entire_text])
        return self.chunks


    def contains_only_special_chars(self,text):
        for char in text:
            if char.isalnum():
                return False
        return True 
    