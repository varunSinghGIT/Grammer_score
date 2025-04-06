# Spoken Language Grammar Scoring Model

## Project Overview
The objective of this competition is to develop a Grammar Scoring Engine for spoken data samples. You are provided with an audio dataset where each file is between 45 to 60 seconds long. The ground truth labels are MOS Likert Grammar Scores for each audio instance (see rubric below). Your task is to build a model that takes an audio file as input and outputs a continuous score ranging from 0 to 5.

Your submission will be assessed based on your ability to preprocess the audio data, select an appropriate methodology to solve the problem, and evaluate its performance using relevant metrics.

Training: The training dataset consists of 444 samples.

Testing (Evaluation): The testing dataset consists of 195 samples.

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Features
- Transcribes spoken audio using OpenAI's Whisper ASR model
- Extracts acoustic features (MFCCs) from audio recordings
- Detects grammar errors using LanguageTool
- Predicts grammar quality scores using a neural network
- Processes both training and test datasets

## Dataset
The project uses the SHL (Spoken Language Health) dataset with the following structure:
- `audios_train/`: Directory containing training audio WAV files
- `audios_test/`: Directory containing test audio WAV files  
- `train.csv`: Training data with filenames and corresponding grammar scores
- `sample_submission.csv`: Template for test predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/spoken-language-grammar-scoring.git
cd spoken-language-grammar-scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Librosa
- Whisper
- Language-tool-python
- NumPy
- Pandas
- scikit-learn
- tqdm

## Project Structure
```
spoken-language-grammar-scoring/
├── model/
│   ├── grammar_score_model.h5    # Trained model
│   └── scaler.pkl                # Feature scaler
├── data/                         # Dataset paths (configured in code)
├── src/
│   ├── main.py                   # Main script
│   ├── feature_extraction.py     # Feature extraction functions
│   ├── model.py                  # Model definition
│   └── utils.py                  # Utility functions
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Methodology

### 1. Feature Extraction
The system extracts two types of features:
- **Audio Features**: 13 MFCCs (Mel-Frequency Cepstral Coefficients) mean values from the audio
- **Grammar Features**: 
  - Count of grammar errors detected
  - Count of unique types of grammar errors

### 2. Speech Recognition
OpenAI's Whisper model (small variant) is used to transcribe the audio files into text.

### 3. Grammar Error Detection
LanguageTool performs grammar checking on the transcribed text, identifying various types of errors.

### 4. Model Architecture
A neural network with the following architecture:
- Input layer: Combined audio features and grammar features
- Hidden layers:
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.3)
  - Dense layer (64 units, ReLU activation)
- Output layer: Single unit for regression (grammar score)

### 5. Training
- Loss function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Optimizer: Adam
- Early stopping with patience of 3 epochs
- Batch size: 16
- Train/validation split: 80%/20%

## Usage

### Training a Model
```python
# The main script handles training
python src/main.py --mode train
```

### Making Predictions
```python
# Generate predictions for test data
python src/main.py --mode predict
```

### Using the Model
```python
# Example code for using the trained model
from tensorflow.keras.models import load_model
import joblib
import librosa
import numpy as np
import whisper
import language_tool_python

# Load model and scaler
model = load_model('model/grammar_score_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Load audio processing tools
asr_model = whisper.load_model("small")
tool = language_tool_python.LanguageTool('en-US')

# Process a new audio file
def predict_grammar_score(audio_path):
    # Extract MFCC features
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Transcribe and get grammar errors
    result = asr_model.transcribe(audio_path)
    text = result['text']
    matches = tool.check(text)
    grammar_errors = len(matches)
    unique_errors = len(set(match.ruleId for match in matches))
    
    # Combine features
    features = np.concatenate([mfcc_mean, [grammar_errors, unique_errors]])
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict score
    score = model.predict(features_scaled)[0][0]
    
    # Clip to valid range
    return max(0, min(score, 5))
```

## Results
The model achieves:
- Mean Absolute Error on validation data: [Fill in your best validation MAE]
- Effective grammar scoring for a variety of speech samples
- Predictions constrained to the 0-5 score range

## Future Improvements
- Experiment with larger Whisper models for improved transcription
- Incorporate additional linguistic features beyond grammar error counts
- Implement data augmentation for audio to improve model robustness
- Add support for languages beyond English
- Explore more complex model architectures (e.g., LSTM for temporal features)

## License
[Choose your license]

## Acknowledgments
- OpenAI for the Whisper ASR model
- LanguageTool for grammar checking capabilities
- The SHL dataset creators
