# Reddit Comment Classification App

This project is a Streamlit web application that classifies reddit comments into three categories: "Veterinarian", "Medical Doctor", or "Others". The classification is performed using a fine-tuned BERT model.

## Features

- Upload a CSV file containing a column named 'comments'
- Preprocess comments by removing duplicates and preserving order
- Classify each comment into one of three categories
- Download the classified results as a CSV file

## Setup Instructions

### Prerequisites

- Python 3

### Clone the Repository

1. Open your terminal or command prompt.
2. Clone the repository using the following command:

    ```bash
    git clone <your-repo-url>
    ```

3. Navigate to the project directory:

    ```bash
    cd <your-repo-name>
    ```

### Set Up the Virtual Environment

1. Create a virtual environment:

    ```bash
    python3 -m venv myenv
    ```

2. Activate the virtual environment:

    - On macOS/Linux:

        ```bash
        source myenv/bin/activate
        ```

    - On Windows:

        ```bash
        myenv\Scripts\activate
        ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Run the Streamlit App

1. Ensure the virtual environment is activated.
2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Upload a CSV file containing a column named 'comments'.
2. The app will preprocess and classify each comment.
3. View the classified results in the app.
4. Download the classified results as a CSV file.

## Error handling on page
1. In a case where streamlit app throws a 'Cannot import BertTokenizer' error, simply refresh page to get required frontend. 