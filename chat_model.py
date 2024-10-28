import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings

# from :class:`~langchain_huggingface import HuggingFaceEmbeddings`
from groq import Groq
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from astrapy import DataAPIClient
from astrapy.constants import VectorMetric

client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))

db = client.get_database_by_api_endpoint(
    "https://4e3c413f-b346-4583-adec-ccafcd2abb5a-us-east-2.apps.astra.datastax.com"
)

collection = db["dateher_chat"]

from langchain_core.runnables import RunnableLambda


class GroqLLM(RunnableLambda):
    def __init__(self, api_key, model_name):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def invoke(self, *args, **kwargs):
        # Check if we have retrieved texts in args
        if (
            args
            and isinstance(args[0], list)
            and all(isinstance(text, str) for text in args[0])
        ):
            # Combine retrieved texts into a single string
            combined_texts = "\n".join(args[0])
        else:
            raise ValueError("Input texts are required and must be a list of strings.")

        stop = kwargs.get("stop", None)

        # Ensure combined_texts is a string
        if combined_texts is None or not isinstance(combined_texts, str):
            raise ValueError("Input texts are required and must be a string.")

        # Create the chat completion request
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a dating chatbot and you are chatting as a real person, suggesting and answering their queries as needed. If you have no answer, then simply generate one of your own, but be relevant to it. juset use plain text no bold italics etc, if no answer seems relevant just ask them to keep queries related to dating only...stop using of course",
                },
                {
                    "role": "user",
                    "content": combined_texts
                    + "\n\nBased on the above information, answet the user query to help them",
                },
            ],
            model=self.model_name,
        )

        response = chat_completion.choices[0].message.content

        # Handle stop tokens if provided
        if stop and response:
            for token in stop:
                if token in response:
                    response = response.split(token)[0]

        return response


model_name = "mixtral-8x7b-32768"
model = GroqLLM(api_key=GROQ_API_KEY, model_name=model_name)


def get_relevant_documents(query):
    query_vector = embeddings.embed_query(query)
    results = collection.find(
        sort={"$vector": query_vector},
        limit=30,
        include_similarity=True,
    )

    relevant_documents = [doc for doc in results if doc["$similarity"] >= 0.7]

    return relevant_documents


def gen(query):
    res = get_relevant_documents(query)
    docc = []
    f = 0
    for i in res:
        if f > 10:
            break
        docc.append(i["text"])
        f = f + 1
    response = model.invoke(docc)
    return response


# response=gen("How to start a conversation?")
# print(response)


# from astrapy import DataAPIClient

# # Initialize the client
# # client = DataAPIClient()
# db = client.get_database_by_api_endpoint(
#     "https://4e3c413f-b346-4583-adec-ccafcd2abb5a-us-east-2.apps.astra.datastax.com"
# )

# collection=db['dateher_chat']
# print(f"Connected to Astra DB: {db.list_collection_names()}")
# print(collection.)

# import ast
import re
from langchain.prompts import PromptTemplate

clt = Groq(
    api_key=GROQ_API_KEY,
)
prompt_template = PromptTemplate(
    input_variables=["text", "attribute_type"],
    template="Extract a list of {attribute_type} mentioned in the following text, ensuring only positive or neutral attributes are included (ignore dislikes or negative mentions): {text}. Provide only the words in a Python list, without any additional text.",
)


def char(input_text, attribute_type):
    formatted_prompt = prompt_template.format(
        text=input_text, attribute_type=attribute_type
    )
    response = clt.chat.completions.create(
        messages=[{"role": "user", "content": formatted_prompt}],
        model="llama3-8b-8192",
    )
    output = response.choices[0].message.content
    try:
        # Remove any leading/trailing whitespace and additional text if needed
        cleaned_output = output.strip()
        # Use regex to match and extract the content within square brackets
        match = re.search(r"\[.*\]", cleaned_output)
        if match:
            # Convert the matched string to a list
            words_list = eval(
                match.group(0)
            )  # Use eval to convert the string representation to a list
            return words_list # Print the result list
        else:
            print("No valid list found in the output.")
    except Exception as e:
        print(f"Error processing output: {e}")


char("I enjoy painting and love spending time in nature. Traveling to new places excites me.","style")