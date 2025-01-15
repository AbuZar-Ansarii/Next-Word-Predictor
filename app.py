import streamlit as st
import pickle
import time
import numpy as np

# Load the model and other necessary objects
model = pickle.load(open("next_word.pkl", "rb"))
pad_sequences_function = pickle.load(open("pad_sequence.pkl", "rb"))  # Renamed variable to clarify it's a function
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

st.title("WORD PREDICTOR")
input_text = st.text_input("Enter something here")

if st.button("Click"):
    predicted_words = []

    # Predict the next 5 words
    for _ in range(10):
        # Tokenize and pad the input text
        tokenized_text = tokenizer.texts_to_sequences([input_text])  # Convert to tokenized sequences
        padded_text = pad_sequences_function(tokenized_text, maxlen=58, padding='pre')  # Ensure proper padding

        # Predict the next word
        pred_y = model.predict(padded_text)
        pos = np.argmax(pred_y)  # Get the index of the predicted word

        # Find the word corresponding to the predicted index
        word = next((word for word, index in tokenizer.word_index.items() if index == pos), None)

        if word:  # If a word is found, append it to the input text
            predicted_words.append(word)
            input_text = input_text + " " + word
            time.sleep(1)  # Reduced sleep to make it more responsive
            st.text(f"Predicted text: {input_text}")  # Print the predicted word
        else:
            st.error("Prediction index not found in tokenizer.word_index.")
            break

    if predicted_words:
        st.sidebar.write("Predicted words: " , "  ".join(predicted_words))
