# Phishing Detection Lab

This application allows for the analysis of URLs to detect phishing attempts using various machine learning models. It supports both single URL analysis and batch testing against a live phishing feed.

## Setup

1.  Ensure you have Python 3.8+ installed.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you are running this from the root project directory, use `pip install -r App/requirements.txt`.*

## Running the Application

To start the application, run the following command from the project root:

```bash
python3 -m streamlit run App/app.py
```

## Features

*   **Single URL Analysis**: Input a URL to receive a classification (Legitimate vs. Phishing) and a confidence score.
*   **Batch API Testing**: Fetch live phishing URLs from OpenPhish and benchmark multiple models to compare their detection rates.
*   **Model Support**: Automatically detects and loads models trained and saved in the `experiments/` directory.
*   **GPU Acceleration**: Automatically detects and utilizes GPU (CUDA/MPS) if available for supported models.

## Directory Structure

*   `app.py`: Main application entry point.
*   `utils.py`: Utility functions for model loading, experiment scanning, and system checks.
*   `feature_extractor.py`: Feature engineering logic for URL processing.
*   `requirements.txt`: List of Python dependencies.
