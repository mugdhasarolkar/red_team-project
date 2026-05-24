from transformers import pipeline
class LLM:
    def __init__(self):
        self.model=pipeline(
            'text-generation',
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=300
        )
    def generate(self,prompt:str):
        result=self.model(prompt)[0]['generated_text']
        return result