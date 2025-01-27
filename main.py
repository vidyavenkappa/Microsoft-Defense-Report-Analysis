from file_reader import FileReader
from generator import LLMGenerator
from retrieval import ChromaDB_Retriever,Ensemble_Retriever,BM25_Retriever
from evaluator import BERTEvaluator
import json
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    # Directory containing text files
    with open('data_json.json', 'r') as file:
        data = json.load(file)
        start_time1 = time.time()
        for d in data:
            start_time = time.time()
            file_path = d['file_name']
            replace_str = d['replace_str']
            file_reader = FileReader(file_path)
            file_reader.read_files()
            documents = file_reader.create_chunks(replace_str)
            print(f"Analysing report: {file_path.split('/')[-1]}")
            
            # query = "Based on the context provided, identify the top 5 most significant security threats. Rank them in order of significance, with the most significant threat listed first. If multiple threats are described, combine and consolidate related ones to present a comprehensive view. If the context does not provide enough information to answer, respond with: 'The context does not provide enough information to answer this question.'"
            query = "Based on the context provided, identify the top 5 most significant security threats. Rank them in order of significance, with the most significant threat listed first. For each threat, provide a brief explanation of why it is considered significant based on the context. If multiple threats are described, combine and consolidate related ones to present a comprehensive view. If the context does not provide enough information to answer, respond with: 'The context does not provide enough information to answer this question.'"


            retriever = Ensemble_Retriever(documents)
            retrieved_context = retriever.retrieve(query)

            generator = LLMGenerator()
            generated_answer = generator.generate(query, retrieved_context)

            print(f"Generated Answer : {generated_answer}")
            
            end_time = time.time()
            print(f'Time taken: {(end_time - start_time)/60} minutes')
            print(' --------- END OF REPORT -------------')


        end_time1 = time.time()
        print(f'Time taken for all files to complete: {(end_time1 - start_time1)/60} minutes')
        
