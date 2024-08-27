markdown

# OpenTranscription

OpenTranscription is a FastAPI application that performs transcription, sentiment analysis, and named entity recognition (NER) on audio files using the Whisper, Hugging Face Transformers, and PyTorch libraries.

## Features

- **Transcription**: Converts audio files into text using the Whisper model.
- **Sentiment Analysis**: Analyzes the sentiment of the transcribed text using a pre-trained model.
- **Named Entity Recognition (NER)**: Identifies named entities such as people in the transcribed text.
- **Environment Configuration**: Uses `.env` files to configure model paths and processing settings.

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended for faster processing)
- Conda (optional but recommended for managing environments)

### Installation

1. **Clone the repository:**

   git clone https://github.com/emoney88/opentranscription.git
   cd opentranscription

    Create and activate a Conda environment:

    bash

conda create -n opentranscription python=3.8
conda activate opentranscription

Install the required packages:

bash

pip install -r requirements.txt

Configure your .env file:

Create a .env file in the project root directory and add the following content:

plaintext

    WHISPER_MODEL=large-v2
    TORCH_DEVICE=cuda
    WHISPER_MODEL: The Whisper model to use (e.g., large-v2).
    TORCH_DEVICE: The device for processing, typically cuda for GPU or cpu.

Usage

    Start the FastAPI server:

    python main.py

    Send a POST request to /transcribe using an API client like Postman:
        Endpoint: http://127.0.0.1:8000/transcribe
        Method: POST
        Form Data:
            file: Select an audio file to upload.

