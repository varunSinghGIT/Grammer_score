# Grammer_score
Grammar Score Prediction from Audio
This project predicts grammar scores from audio recordings using a combination of acoustic features and automated grammar checking. It leverages speech-to-text transcription, grammar error detection, and deep learning to estimate grammatical proficiency.

Features
Audio Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients) extraction using Librosa.

Speech Recognition: Whisper ASR model for transcribing audio to text.

Grammar Analysis: LanguageTool integration to detect grammatical errors.

Neural Network: Keras-based model combining acoustic and grammar features.

Scalable Pipeline: Full workflow from raw audio to predicted scores.

Installation
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/grammar-score-prediction.git
cd grammar-score-prediction
Install dependencies:

bash
Copy
pip install -r requirements.txt
requirements.txt (example):

Copy
tensorflow>=2.10
librosa==0.10.0
pandas==1.5.3
whisper-openai==1.1.10
language-tool-python==2.7.1
scikit-learn==1.2.2
tqdm==4.65.0
Usage
Data Preparation
Place your dataset in the following structure:

Copy
dataset/
├── audios_train/   # Training audio files (.wav)
├── audios_test/    # Test audio files (.wav)
├── train.csv       # Training metadata with 'filename' and 'grammar_score'
└── sample_submission.csv  # Submission template
Training
python
Copy
# Update paths in the script if needed
AUDIO_DIR = 'path/to/audios_train'
train_csv = pd.read_csv('path/to/train.csv')

# Run the full pipeline:
python train.py
Prediction
python
Copy
# After training, predict on test data:
TEST_AUDIO_DIR = 'path/to/audios_test'
python predict.py
Model Architecture
A neural network combining acoustic and grammatical features:

python
Copy
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 15)]              0         
                                                                 
 dense (Dense)               (None, 128)               2048      
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dense_2 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 10,369
Trainable params: 10,369
Non-trainable params: 0
Key Components
Feature Extraction:

13 MFCC coefficients + temporal statistics

Grammar error counts from LanguageTool

Unique grammar rule violations

Preprocessing:

Standard scaling of features

Train/validation split (80/20)

Training:

Early stopping with 3-epoch patience

Adam optimizer, MSE loss

Batch size: 16, Max epochs: 20

Output Files
grammar_score_model.h5: Trained Keras model

scaler.pkl: Feature scaler for new data

sample_submission.csv: Predictions on test set

Evaluation
Metrics tracked during training:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Example training output:

Copy
Epoch 1/20
125/125 [==============================] - 2s 8ms/step - loss: 2.4567 - mae: 1.2345 - val_loss: 1.8901 - val_mae: 1.1023
...
License
MIT License. See LICENSE for details.

Acknowledgements
SHL Dataset (hypothetical example)

OpenAI Whisper for speech recognition

LanguageTool for grammar checking
