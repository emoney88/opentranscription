# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from transformers import pipeline
import torch
import torchaudio
import whisper
import os
from io import BytesIO
import html

# Initialize FastAPI app
app = FastAPI()

# Load Whisper model for transcription
whisper_model = whisper.load_model("large-v2")

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Function to perform transcription
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['text'], result['segments']

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Function to perform NER
def perform_ner(text):
    result = ner_pipeline(text)
    entities = [entity['word'] for entity in result if entity['entity'] == 'B-PER']
    return entities

# Function to generate HTML response
def generate_html(transcription, sentiment, score, entities, segments):
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
    html_content += """
            </tbody>
        </table>
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Perform transcription
        transcription, segments = transcribe_audio(file_path)
        # Perform sentiment analysis
        sentiment, score = analyze_sentiment(transcription)
        # Perform named entity recognition
        entities = perform_ner(transcription)

        # Generate HTML response
        html_response = generate_html(transcription, sentiment, score, entities, segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
    finally:
        os.remove(file_path)

    return html_response

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
