import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure

# Embedding Model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Connect Weaviate
client = weaviate.connect_to_local()

# Delete old collection if exists
if client.collections.exists("MobileCollection"):
    client.collections.delete("MobileCollection")

# Create Collection
client.collections.create(
    name="MobileCollection",
    vector_config=Configure.Vectors.self_provided()
)

collection = client.collections.get(
    "MobileCollection"
)


# Chunking Function
def chunk_text(text, chunk_size=50, overlap=10):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        chunk = words[start:start + chunk_size]

        chunks.append(
            " ".join(chunk)
        )

        start += (chunk_size - overlap)

    return chunks


# Read Dataset
df = pd.read_csv(
    r"C:\Users\DELL\Downloads\archive (2)\mobiles.csv"
)

for _, row in df.iterrows():

    document = f"""
    Brand: {row['Brand']}
    Mobile Name: {row['Title']}
    Price: {row['Price']}
    Rating: {row['Rating']}
    Storage: {row['Internal Storage']}
    Camera: {row['Primary Camera']}
    Battery: {row['Battery Capacity']}
    Processor: {row['Processor Type']}
    Operating System: {row['Operating System']}
    """

    chunks = chunk_text(document)

    for chunk in chunks:

        vector = embedding_model.encode(
            chunk
        ).tolist()

        collection.data.insert(
            properties={
                "brand": str(row["Brand"]),
                "title": str(row["Title"]),
                "price": str(row["Price"]),
                "rating": str(row["Rating"]),
                "storage": str(row["Internal Storage"]),
                "camera": str(row["Primary Camera"]),
                "battery": str(row["Battery Capacity"]),
                "processor": str(row["Processor Type"]),
                "os": str(row["Operating System"]),
                "document": chunk
            },
            vector=vector
        )

print("Data inserted successfully")

client.close()