from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Generator:
    def generate(self, query: str, context: List[str]) -> str:
        """
        Generate an answer based on query and context.
        Args:
            query (str): User query.
            context (List[str]): Retrieved documents.
        Returns:
            str: Generated answer.
        """
        raise NotImplementedError("Subclasses must implement this method")


class LLMGenerator(Generator):
    def __init__(self):
        self.model_name = "microsoft/Phi-3.5-mini-instruct"
        
    def generate(self, query: str, context: List[str]) -> str:
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        
        combined_context = ""
        for text in context:
            combined_context+=" "+text.page_content
#         prompt = f"""Use the provided context string to answer the given question. Do not generate follow-up questions or include information outside the context. Respond solely based on the context provided.\n
# Context: {combined_context}\n\nQuestion: {query}\n\nAnswer:"""
        prompt = f'''You are a highly intelligent AI assistant. Use only the provided context to answer the following question as accurately as possible."
### Context:
{context}


### Question:
{query} 

### Answer:'''



        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate response using the model
        outputs = model.generate(
            inputs["input_ids"],
            max_length=32000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

