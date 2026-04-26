# IMDB Sentiment Analysis using RNN, LSTM, and BiLSTM

A comprehensive deep learning project for sentiment analysis on IMDB movie reviews using various Recurrent Neural Network architectures. This project compares SimpleRNN, LSTM, and Bidirectional LSTM models to classify movie reviews as positive or negative.

## 🎯 Project Overview

This project implements sentiment classification models that predict whether a movie review is positive or negative using the IMDB dataset. The models utilize different RNN architectures with embedding layers to process sequential text data and achieve high accuracy in sentiment classification.

**Key Features:**
- Sentiment classification (Positive/Negative)
- Comparison of three RNN architectures: SimpleRNN, LSTM, and BiLSTM
- Deep learning with TensorFlow/Keras
- Embedding layers for text processing
- Interactive Streamlit web application for real-time predictions
- Pre-trained models for quick deployment
- Data preprocessing with padding and early stopping
- Model evaluation with accuracy, loss, confusion matrix, and classification reports

**Models Implemented:**
- **SimpleRNN**: Basic recurrent neural network for sequence processing
- **LSTM**: Long Short-Term Memory network for better handling of long sequences
- **BiLSTM**: Bidirectional LSTM for capturing context from both directions

**Final Model Chosen:** LSTM (`lstm_model.h5`) - Selected for its balanced performance in handling long sequences without significant overfitting.

## 🚀 Live Demo

**Streamlit App:** https://imdb-sentiment-analysis-using-rnn.streamlit.app/

## 📋 Requirements

- Python 3.11+
- TensorFlow 2.15.0
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

See `requirements.txt` for complete dependencies and `runtime.txt` for Python version specification.

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "IMDB Sentiment Analysis using RNN"
```

### 2. Create a Virtual Environment (Optional but Recommended)

Using conda:
```bash
conda create -n imdb_env python=3.11
conda activate imdb_env
```

Or using venv:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
IMDB Sentiment Analysis using RNN/
├── imdb_model.ipynb              # Main training notebook with all three models
├── embedding.ipynb               # Embedding layer exploration and analysis
├── prediction.ipynb              # Model prediction and inference examples
├── main.py                       # Streamlit web application for predictions
├── lstm_model.h5                 # Pre-trained LSTM model (final model)
├── bilstm_model.h5               # Pre-trained BiLSTM model
├── simple_rnn_model.h5           # Pre-trained SimpleRNN model
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for deployment
└── README.md                     # This file
```

## 🛠️ Usage

### Option 1: Use the Pre-trained Model (Recommended)

Run the Streamlit app with the pre-trained LSTM model:

```bash
streamlit run main.py
```

Then open your browser to `http://localhost:8501` and enter a movie review to get sentiment predictions.

### Option 2: Train Your Own Models

Open and run the training notebook:

```bash
jupyter notebook imdb_model.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model building for SimpleRNN, LSTM, and BiLSTM
- Training with early stopping
- Model evaluation and comparison
- Saving trained models

### Option 3: Explore Embeddings

```bash
jupyter notebook embedding.ipynb
```

### Option 4: Make Predictions

```bash
jupyter notebook prediction.ipynb
```

## 📊 Model Performance

Based on the evaluation metrics:

| Model      | Accuracy | Loss   | Notes |
|------------|----------|--------|-------|
| SimpleRNN | ~79%    | ~0.44 | Basic RNN, prone to vanishing gradients |
| LSTM      | ~86%    | ~0.35 | Better sequence handling |
| BiLSTM    | ~87%    | ~0.32 | Best performance, slight overfitting |

**Final Choice:** LSTM model (`lstm_model.h5`) provides excellent accuracy with better generalization and is recommended for production use.

## 🔧 Model Architecture

### LSTM Model (Final)
```
Embedding Layer (10000, 128) -> LSTM (128) -> Dense (1, sigmoid)
```

### BiLSTM Model
```
Embedding Layer (10000, 128) -> Bidirectional LSTM (64, dropout=0.2) -> Dense (1, sigmoid)
```

### SimpleRNN Model
```
Embedding Layer (10000, 128) -> SimpleRNN (128, relu) -> Dense (1, sigmoid)
```

## 📈 Data Preprocessing

- **Dataset:** IMDB movie reviews (25,000 training, 25,000 testing)
- **Vocabulary Size:** 10,000 most frequent words
- **Sequence Length:** Padded/truncated to 500 words
- **Labels:** Binary (0: Negative, 1: Positive)

## 🎯 Prediction Example

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import numpy as np

# Load the model
model = load_model('lstm_model.h5')

# Prepare input text
word_index = imdb.get_word_index()
def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 is <UNK>
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

# Make prediction
review = "This movie was fantastic!"
encoded_review = encode_review(review)
prediction = model.predict(encoded_review)[0][0]
sentiment = "Positive" if prediction > 0.5 else "Negative"
print(f"Sentiment: {sentiment} (Confidence: {prediction:.2f})")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IMDB dataset from Keras
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework
- Scikit-learn for evaluation metrics

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note:** The BiLSTM model shows slight overfitting but provides the best overall performance. For production use, consider additional regularization techniques or data augmentation to mitigate overfitting.

Follow the cells sequentially to:
1. Load the IMDB dataset
2. Preprocess and pad sequences
3. Build the SimpleRNN model
4. Train with early stopping
5. Evaluate on test data
6. Save the trained model

### Option 3: Explore Embeddings

```bash
jupyter notebook embedding.ipynb
```

This notebook explores word embeddings and their usage in the model.

## 🧠 Model Architecture

```
Layer (Type)              Output Shape           Param #
==========================================
Embedding Layer           (None, 500, 32)        320,000
SimpleRNN Layer           (None, 100)            13,300
Dense Layer (ReLU)        (None, 50)             5,050
Dropout                   (None, 50)             0
Output Layer (Sigmoid)    (None, 1)              51
==========================================
Total Parameters: 338,401
Trainable Parameters: 338,401
```

**Architecture Details:**
- **Embedding Layer:** Converts word indices to 32-dimensional vectors
- **SimpleRNN Layer:** 100 hidden units for sequential processing
- **Dense Layer:** 50 units with ReLU activation
- **Output Layer:** Sigmoid activation for binary classification
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy

## 📊 Dataset

**IMDB Dataset:**
- 25,000 training reviews
- 25,000 test reviews
- Binary labels: 0 (negative), 1 (positive)
- Max vocabulary: 10,000 most common words
- Sequence length: Padded to 500 tokens

## 📈 Model Performance

After training with early stopping:
- **Training Accuracy:** ~88-92%
- **Test Accuracy:** ~86-90%
- **Validation Loss:** Stable convergence

*Note: Exact metrics depend on the random seed and training parameters*

## 🎮 Interactive Prediction

The Streamlit app (`main.py`) provides:
- Text input for movie reviews
- Real-time sentiment prediction
- Confidence score display
- User-friendly interface

Example:
```
Input: "This movie was absolutely amazing! Great plot and excellent acting."
Output: Positive Sentiment (Confidence: 0.94)
```

## 🔧 Configuration

Edit these parameters in the notebooks to customize training:

```python
max_features = 10000      # Vocabulary size
max_len = 500             # Sequence length
embedding_dim = 32        # Embedding dimension
hidden_units = 100        # RNN hidden units
batch_size = 32           # Training batch size
epochs = 20               # Maximum epochs
validation_split = 0.2    # Training/validation split
```

## 📚 Key Notebooks

### rnn_model.ipynb
Main training pipeline:
- Data loading and exploration
- Preprocessing and padding
- Model building and training
- Evaluation and visualization
- Model saving

### embedding.ipynb
Deep dive into embeddings:
- Word vector visualization
- Embedding layer exploration
- Vector similarity analysis

### prediction.ipynb
Making predictions:
- Loading pre-trained model
- Single and batch predictions
- Result interpretation

## 🚢 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect your repository
4. Set main file path to `main.py`
5. Add `STREAMLIT_SERVER_PORT` environment variable if needed
6. Deploy!

### Deploy to Other Platforms

The project includes `runtime.txt` for specifying Python version on platforms like Heroku.

## 🐛 Troubleshooting

**TensorFlow Import Error:**
```bash
pip install tensorflow==2.15.0
```

**CUDA/GPU Issues:**
Use CPU version:
```bash
pip install tensorflow-cpu==2.15.0
```

**Out of Memory:**
- Reduce `batch_size` in training
- Reduce `max_len` sequence length
- Use GPU if available

## 📝 Example Usage

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load model
model = load_model('simple_rnn_model.h5')

# Prepare review (simplified example)
review_indices = [10, 45, 100, 56, 20, 5]  # Encoded review
padded_review = sequence.pad_sequences([review_indices], maxlen=500)

# Predict
prediction = model.predict(padded_review)
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
```

## 📚 Resources

- [TensorFlow/Keras Documentation](https://tensorflow.org/)
- [IMDB Dataset Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
- [RNN and LSTM Tutorial](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as a deep learning machine learning project exploring sentiment analysis with RNNs.

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

## ⭐ Star This Project

If you found this helpful, please consider giving it a star! ⭐

---

**Last Updated:** April 2026
**Python Version:** 3.11+
**TensorFlow Version:** 2.15.0
