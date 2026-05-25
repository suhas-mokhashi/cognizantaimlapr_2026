#realtime inference using fastapi

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

app=FastAPI()

class Request(BaseModel):
    review_text:str=Field(...,description="enter text")   

@app.post("/predict_sentiment")
def analyse_text(request:Request):
    sentiment_pipeline=pipeline("sentiment-analysis")
    result=sentiment_pipeline(request.review_text) [0]
    return {"sentiment":result[0]['label'],
            "confidence_score":round(result[0]['score'],4)}