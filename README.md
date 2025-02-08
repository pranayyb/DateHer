# DateHer - Your Ultimate Dating Companion

DateHer is an AI-powered dating platform that enhances matchmaking through personalized chatbot interactions, compatibility scoring, and emotional tone analysis. Our intelligent system ensures users connect with like-minded individuals based on shared interests and communication styles.

## Features

- **Smart Chatbot Interaction**: Users engage with a chatbot upon signing up, answering a set of predefined questions to determine their personality, interests, and values. This initial chat helps assign an interest-based score.
- **Matchmaking Algorithm**: Users are matched with others who have similar scores, ensuring they share common interests and values.
- **Message Tone Analysis**: The platform predicts the emotional tone of messages exchanged in chats, helping users better understand conversations.
- **Compatibility Scoring**: As two users continue chatting, the system calculates a dynamic compatibility score based on their interaction history.


## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/pranayyb/DateHer.git
   cd DateHer
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

## Configuration - Set the Environment Variables

- ASTRA_DB_API_ENDPOINT
- ASTRA_DB_APPLICATION_TOKEN
- GROQ_API_KEY

## Usage

1. **Run the API Server**:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 9000
   ```

2. **Endpoints**:

   - **`/chat`**: Handles chat interactions and assigns an initial interest score.

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
         },
         "chat_history": ["string"]
       }
       ```

   - **`/match`**: Finds potential matches based on user interest scores.

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
           "similarity_score": "number"
         }
       ]
       ```

   - **`/calc`**: Calculates compatibility score after users engage in multiple chats.

     - **Method**: `POST`
     - **Request Body**:
       ```json
       {
         "user_id_1": "string",
         "user_id_2": "string"
       }
       ```
     - **Response**:
       ```json
       {
         "compatibility_score": "number"
       }
       ```

   - **`/tonepredict`**: Predicts the emotional tone of a given text.
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

- Ensure your machine learning models and pickled objects are correctly placed and named.
- Adjust CORS settings as needed for your deployment environment.

## Author

- **Pranay Buradkar**

## License

This project is licensed under the MIT License.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue to discuss your ideas first.
