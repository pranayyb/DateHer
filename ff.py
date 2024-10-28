import os
from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from dotenv import load_dotenv
load_dotenv()
# Initialize the client and get a "Database" object
client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
database = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
print(f"* Database: {database.info().name}\n")

# Create a collection. The default similarity metric is cosine.
# Choose dimensions that match your vector data.
# If you're not sure, use the vector dimension that your embeddings model produces.
collection = database.create_collection(
    "vector_test",
    dimension=5,
    metric=VectorMetric.COSINE,  # Or just 'cosine'.
    check_exists=False, # Optional.
)
print(f"* Collection: {collection.full_name}\n")

# Insert documents with embeddings into the collection.
documents = [
    {
        "text": "Chat bot integrated sneakers that talk to you",
        "$vector": [0.1, 0.15, 0.3, 0.12, 0.05],
    },
    {
        "text": "An AI quilt to help you sleep forever",
        "$vector": [0.45, 0.09, 0.01, 0.2, 0.11],
    },
    {
        "text": "A deep learning display that controls your mood",
        "$vector": [0.1, 0.05, 0.08, 0.3, 0.6],
    },
]
# insertion_result = collection.insert_many(documents)
# print(f"* Inserted {len(insertion_result.inserted_ids)} items.\n")

# Perform a similarity search.
query_vector = [0.15, 0.1, 0.1, 0.35, 0.55]
results = collection.find(
    sort={"$vector": query_vector},
    limit=10,
    include_similarity=True,
)
print("Vector search results:")
for document in results:
    print("    ", document)