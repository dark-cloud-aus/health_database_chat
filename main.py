# CodeBlue MedHack 2025 Challenge Patient Data Analyser written by David Gilmore

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.analyzer import DataAnalyzer
from src.chatbot import HealthcareGPT
from src.vector_store import VectorStore
from pathlib import Path
import logging
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize components
        logger.info("Starting application...")
        data_path = Path.home() / "Desktop/med/data"
        
        logger.info("Initializing data loader...")
        loader = DataLoader(data_path)
        
        # Load data
        logger.info("Loading data...")
        df = loader.load_data()
        
        # Process data
        logger.info("Processing data...")
        processor = DataProcessor(df)
        summary_stats = processor.get_summary_stats()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore()
        
        # Convert dataframes to records and store in Pinecone
        logger.info("Loading data into vector store...")
        chunk_size = 100  # Reduced chunk size
        for file_name in ['icustays.csv', 'chartevents.csv', 'inputevents.csv']:
            logger.info(f"Processing {file_name}...")
            try:
                for chunk in pd.read_csv(data_path / file_name, chunksize=chunk_size):
                    records = chunk.to_dict('records')
                    vector_store.upsert_records(records)
                    time.sleep(1)  # Add delay between chunks
                    logger.info(f"Processed {len(records)} records from {file_name}")
            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")
                continue
        
        logger.info("Data loading complete!")
        
        # Initialize analyzer and chatbot with vector store
        logger.info("Initializing analyzer and chatbot...")
        analyzer = DataAnalyzer(df)
        chatbot = HealthcareGPT(summary_stats, vector_store)  # Pass vector_store to chatbot
        
        logger.info("Setup complete! Ready for questions.")
        print("\n=== CodeBlue Healthcare Data Analysis System ===")
        print("Type your questions about the healthcare data or 'quit' to exit")
        print("================================================\n")
        
        # Interactive loop
        while True:
            question = input("\nTeam CodeBlue Enter your question about the healthcare data (or 'quit' to exit): ")
            
            if question.lower() == 'quit':
                break
                
            # Get AI response
            response = chatbot.ask_question(question)
            print("\nAI Analysis:", response)
            
            # Ask if user wants to see visualizations
            viz_response = input("\nWould you like to see any visualizations? (yes/no): ")
            if viz_response.lower() == 'yes':
                column = input("Which column would you like to visualize?: ")
                chart_type = input("What type of chart? (histogram/box): ")
                fig = analyzer.generate_visualization(column, chart_type)
                fig.show()
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
