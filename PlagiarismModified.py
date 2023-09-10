import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import pickle
import base64

# Set page configuration
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="✍️",
    layout="wide",
)

# Custom CSS to improve app appearance
st.markdown(
    """
    <style>
    .stFileUploader .stButton button {
        background-color: #008CBA;
        color: white;  /* Set text color to white */
        border-radius: 5px;  /* Add rounded corners */
    }
    .stFileUploader .stButton button:hover {
        background-color: #00678E;
    }
    .stDataFrame td {
        text-align: center;
    }
    .stProgress > div > div > div {
        background-color: #008CBA;  /* Change progress bar color */
        border-radius: 5px;  /* Add rounded corners to progress bar */
    }
    .stProgress > div > div > div > div {
        background-color: white;  /* Change the background color within progress bar */
    }
    .progress-text {
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Download NLTK resources (only need to do this once)
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

# Function to calculate similarity
def similarity(doc1, doc2):
    return cosine_similarity(doc1.reshape(1, -1), doc2.reshape(1, -1))[0][0]

# Load the pre-trained TF-IDF vectorizer and existing files from pickle files
with open('tfidf_vectorizer.pickle', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open('existing_files.pickle', 'rb') as files_file:
    existing_files = pickle.load(files_file)

# Streamlit app
st.title("Plagiarism Checker")

# Upload a new file
uploaded_file = st.file_uploader("Upload a new file (in .txt format):", type=["txt"])

if uploaded_file is not None:
    # Read the uploaded file as binary data
    uploaded_content = uploaded_file.read()

    # Decode the binary content to text (assuming UTF-8 encoding, adjust if needed)
    uploaded_text = uploaded_content.decode('utf-8')

    # Preprocess the text
    preprocessed_new_file_content = preprocess_text(uploaded_text)

    # Continue with the rest of your code as before
    new_vector = tfidf_vectorizer.transform([preprocessed_new_file_content]).toarray()

    # Calculate similarity with existing files
    similarity_scores = []
    results_data = []

    for file_name, text_vector in existing_files:
        sim_score = similarity(new_vector, text_vector)
        similarity_scores.append(sim_score)
        file_pair = sorted((uploaded_file.name, file_name))
        score = (file_pair[0], file_pair[1], sim_score)
        results_data.append(score)

    avg_similarity = np.mean(similarity_scores)

    # Create a DataFrame to display results
    results_df = pd.DataFrame(results_data, columns=['File1', 'File2', 'Similarity Score'])

    # Display the average plagiarism score as a progress bar
    st.subheader("Results")
    st.progress(avg_similarity)

    # Display the percentage text alongside the progress bar using HTML
    st.markdown(f"<div class='progress-text'>Average Plagiarism Score: {avg_similarity:.2%}</div>", unsafe_allow_html=True)

    # Display the similarity score and results
    st.subheader("Similarity with Existing Files")
    st.dataframe(results_df)
