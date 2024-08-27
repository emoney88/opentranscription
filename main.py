from dotenv import load_dotenv
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from transformers import pipeline
import torch
import torchaudio
import whisper
import html
import time

# Load environment variables from .env file
load_dotenv()

# Access the variables using os.getenv
whisper_model_name = os.getenv("WHISPER_MODEL", "large-v2")
torch_device = os.getenv("TORCH_DEVICE", "cuda")

# Load Whisper model with environment settings
whisper_model = whisper.load_model(whisper_model_name).to(torch_device)

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Load NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize FastAPI app
app = FastAPI()

# Function to perform transcription
def transcribe_audio(file_path):
    start_time = time.time()
    result = whisper_model.transcribe(file_path)
    end_time = time.time()
    processing_time = end_time - start_time
    transcription = result['text']
    return transcription, processing_time

# Function to analyze sentiment
def analyze_sentiment(text):
    start_time = time.time()
    result = sentiment_pipeline(text)[0]
    end_time = time.time()
    processing_time = end_time - start_time
    sentiment = result['label']
    score = result['score']
    return sentiment, score, processing_time

# Function to perform NER
def perform_ner(text):
    start_time = time.time()
    result = ner_pipeline(text)
    end_time = time.time()
    processing_time = end_time - start_time
    entities = [entity['word'] for entity in result if entity['entity'] == 'B-PER']
    return entities, processing_time

# Function to generate HTML response
def generate_html(transcription, sentiment, score, entities, segments, times):
    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transcription Results</title>
    </head>
    <body>
        <h1>Transcription Results</h1>
        <h2>Transcription</h2>
        <p>{html.escape(transcription)}</p>
        <h2>Sentiment</h2>
        <p>Label: {html.escape(sentiment)}</p>
        <p>Score: {score:.2f}</p>
        <h2>Named Entities</h2>
        <p>Entities: {", ".join(html.escape(entity) for entity in entities)}</p>
        <h2>Segments</h2>
        <table border="1">
            <thead>
                <tr>
                    <th>Start</th>
                    <th>End</th>
                    <th>Text</th>
                </tr>
            </thead>
            <tbody>
        """
    for segment in segments:
        html_content += f"""
                <tr>
                    <td>{segment['start']:.2f}</td>
                    <td>{segment['end']:.2f}</td>
                    <td>{html.escape(segment['text'])}</td>
                </tr>
        """
    html_content += f"""
            </tbody>
        </table>
        <h2>Processing Times</h2>
        <p>Transcription Time: {times['transcription_time']:.2f} seconds</p>
        <p>Sentiment Analysis Time: {times['sentiment_time']:.2f} seconds</p>
        <p>NER Time: {times['ner_time']:.2f} seconds</p>
    </body>
    </html>
    """
    return html_content

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Transcription
        transcription, transcription_time = transcribe_audio(file_path)

        # Sentiment Analysis
        sentiment, score, sentiment_time = analyze_sentiment(transcription)

        # Named Entity Recognition
        entities, ner_time = perform_ner(transcription)

        # Compile the processing times
        times = {
            "transcription_time": transcription_time,
            "sentiment_time": sentiment_time,
            "ner_time": ner_time
        }

        # Generate HTML response
        html_response = generate_html(transcription, sentiment, score, entities, [], times)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
    finally:
        os.remove(file_path)

    return html_response

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
