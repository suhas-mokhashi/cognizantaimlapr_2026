#batch inferences using reviews.csv and transformer model
import pandas as pd
from transformers import pipeline

def batch_inference(reviews_csv):
    # Load the reviews from the CSV file
    reviews_df = pd.read_csv(reviews_csv)
    
    # Initialize the sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", 
                                  model="distilbert-base-uncased-finetuned-sst-2-english" )
    
    # Perform batch inference on the reviews
    results = sentiment_pipeline(reviews_df['review_text'].tolist())
    
    # Add the results to the DataFrame
    reviews_df['sentiment'] = [result['label'] for result in results]
    #confidence scores
    reviews_df['confidence'] = [result['score'] for result in results]
    return reviews_df