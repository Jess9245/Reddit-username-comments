import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

# Load the model and tokenizer
model_path = './fine-tuned-bert'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Assuming label_encoder is already defined and trained from previous steps
label_encoder = LabelEncoder()
label_encoder.fit(["Others", "Veterinarian", "Medical Doctor"])  # Ensure the order matches your labels

def classify_comment(comment):
    # Preprocess the input comment
    inputs = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # Decode the predictions to label names
    predicted_label = label_encoder.inverse_transform(predictions.cpu().numpy())[0]
    return predicted_label

# Function to process comments
def process_comments(comment):
    # Split the comment by pipe and remove duplicates while preserving order
    comment_list = comment.split('|')
    unique_comments = list(OrderedDict.fromkeys(comment.strip() for comment in comment_list))
    joined_comments = ' '.join(unique_comments)
    return joined_comments 

# Streamlit app
st.title('Reddit Comments Classification App')
st.write("Upload a CSV file containing a column named 'comments' to classify each comment as 'Veterinarian', 'Medical Doctor', or 'Others'.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    if 'comments' not in df.columns:
        st.error("CSV file must contain a 'comments' column.")
    else:
        # Process comments
        df['processed_comments'] = df['comments'].apply(process_comments)

        # Perform classification on each processed comment
        df['prediction'] = df['processed_comments'].apply(classify_comment)

        # Display the dataframe with predictions
        st.write(df)

        # Provide an option to download the results as a CSV file
        st.download_button(
            label="Download Predictions as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )

