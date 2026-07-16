# PC Building Helper Chatbot

A chatbot that answers questions about building a PC using NLP techniques

## Features
- Answers FAQs about PC components
- Uses TF-IDF and cosine similarity to find the best answer
- Simple chat UI with Streamlit

## Technologies Used
- Python
- Streamlit
- NLTK
- scikit-learn

## Installation

1. Clone the repository
git clone https://github.com/MrwanAmr185/PC-Building-Chatbot

2. Install dependencies
pip install streamlit nltk scikit-learn

3. Run the app
streamlit run code.py

## How it works
1. User types a question
2. The question is preprocessed using NLTK
3. TF-IDF vectorizer converts the question to a vector
4. Cosine similarity finds the closest FAQ
5. The best matching answer is returned
