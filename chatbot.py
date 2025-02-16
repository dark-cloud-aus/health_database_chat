from openai import OpenAI
from decouple import config
from typing import Dict, List

class HealthcareGPT:
    def __init__(self, data_summary: Dict, vector_store=None):
        self.openai_api_key = config('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.openai_api_key)
        self.data_summary = data_summary
        self.vector_store = vector_store
        
    def ask_question(self, question: str) -> str:
        """Ask a question about the healthcare dataset"""
        try:
            # Get relevant records using vector search
            relevant_records = []
            if self.vector_store:
                relevant_records = self.vector_store.query_similar(question, top_k=5)
            
            # Construct context for the AI
            context = f"""
            You are analyzing a real hospital dataset with the following files:

            1. icustays.csv - Contains information about ICU stays
            2. chartevents.csv - Contains patient charting data
            3. inputevents.csv - Contains data about medications and other inputs
            4. datetimeevents.csv - Contains timestamped events
            5. ingredientevents.csv - Contains medication ingredient information
            6. d_items.csv - Dictionary/reference table for items
            7. caregiver.csv - Information about healthcare providers

            The columns available in these files are: {', '.join(self.data_summary['columns'])}

            Relevant records found:
            {relevant_records if relevant_records else 'No specific records found'}

            Please analyze the data structure and provide specific insights about: {question}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a healthcare data analyst expert specializing in hospital ICU data analysis. Provide specific insights based on the available data."},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error processing question: {str(e)}" 
