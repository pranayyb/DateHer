# Chatbot and Compatibility Scoring API

This project provides a FastAPI-based web service that offers various functionalities related to chat interactions, compatibility scoring, and emotion prediction. It uses TensorFlow for machine learning, NLTK for text processing, and integrates with external APIs to enhance its features.

## Features

- **Chat Interaction**: Engages users in conversation, asks questions, and provides responses based on predefined intents. First it will ask some questions to set the parameters and then you can ask the chatbot questions related to relationship and stuffs.
- **Compatibility Scoring**: Computes compatibility scores between users based on their interaction history and profile attributes.
- **Matchmaking**: Finds potential matches based on user preferences and profile data.
- **Emotion Prediction**: Predicts the emotional tone of a given text using a pre-trained model.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- TensorFlow
- NLTK
- Pandas
- Scikit-learn
- HTTPX
- Requests
- Pickle

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/pranayyb/DateHer.git
    cd https://github.com/pranayyb/DateHer.git
    ```

2. **Create a Virtual Environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK Data**:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Configuration

- Ensure that `intents.json`, `chatbot_model.h5`, `tokenizer.pickle`, and `emotion.pkl` are present in the project directory.
- Configure the external API URLs used in the `/calc` and `/match` endpoints to match your environment.

## Usage

1. **Run the API Server**:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 9000
    ```

2. **Endpoints**:

    - **`/chat`**: Handles chat interactions.
      - **Method**: `POST`
      - **Request Body**:
        ```json
        {
          "user_id": "string",
          "message": "string"
        }
        ```
      - **Response**:
        ```json
        {
          "response": "string",
          "current_question_index": 0,
          "conversation": ["string"],
          "user_info": {
            "conversation": ["string"],
            "current_question_index": 0,
            "interests": ["string"],
            "values": ["string"],
            "style": "string",
            "traits": ["string"],
            "state": "string",
            "commitment": ["string"],
            "resolution": ["string"]
          }
        }
        ```

    - **`/calc`**: Calculates compatibility score based on chat history.
      - **Method**: `POST`
      - **Request Body**:
        ```json
        {
          "data": [
            {
              "text": "string",
              "senderId": "string"
            }
          ]
        }
        ```
      - **Response**:
        ```json
        {
          "comp_score": "string"
        }
        ```

    - **`/match`**: Finds potential matches based on user preferences.
      - **Method**: `POST`
      - **Request Body**:
        ```json
        {
          "user_id": "string"
        }
        ```
      - **Response**:
        ```json
        [
          {
            "id": "string",
            "age": "number",
            "interests": "string",
            "similarity": "number"
          }
        ]
        ```

    - **`/tonepredict`**: Predicts the emotion of a given text.
      - **Method**: `POST`
      - **Request Body**:
        ```json
        {
          "text": "string"
        }
        ```
      - **Response**:
        ```json
        {
          "emotion": "string"
        }
        ```

## Notes

- Ensure your machine learning model files and pickled objects are correctly placed and named!.
- Adjust CORS settings as needed for your deployment environment.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss your ideas first.


