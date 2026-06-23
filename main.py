from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import weaviate
import re

app = FastAPI(
    title="Corrective RAG Mobile Recommendation System"
)

# Embedding Model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Weaviate Connection
client = weaviate.connect_to_local()

collection = client.collections.get(
    "MobileCollection"
)


class QueryRequest(BaseModel):
    query: str


def camera_score(camera_text):

    matches = re.findall(
        r"(\d+)MP",
        str(camera_text)
    )

    if matches:
        return max(
            [int(x) for x in matches]
        )

    return 0


def battery_score(battery_text):

    matches = re.findall(
        r"(\d+)",
        str(battery_text)
    )

    if matches:
        return int(matches[0])

    return 0


def get_recommendation(query):

    query_vector = embedding_model.encode(
        query
    ).tolist()

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=20
    )

    if not results.objects:
        return {
            "message": "No mobile found"
        }

    # Remove duplicate mobiles
    unique_docs = {}

    for obj in results.objects:

        title = obj.properties["title"]

        if title not in unique_docs:
            unique_docs[title] = obj

    unique_results = list(
        unique_docs.values()
    )

    # All Chunks
    all_chunks_created = {}

    for i, obj in enumerate(
        unique_results[:5]
    ):
        all_chunks_created[
            f"chunk{i+1}"
        ] = obj.properties["document"]

    # Retrieved Chunks
    retrieved_chunks = {}

    for i, obj in enumerate(
        unique_results[:5]
    ):
        retrieved_chunks[
            f"retrieved_chunk{i+1}"
        ] = obj.properties["document"]

    query_emb = embedding_model.encode(
        query,
        convert_to_tensor=True
    )

    ranked_docs = []

    for obj in unique_results:

        doc_emb = embedding_model.encode(
            obj.properties["document"],
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            query_emb,
            doc_emb
        ).item()

        ranked_docs.append(
            (obj, similarity)
        )

    # Corrective Retrieval

    if "camera" in query.lower():

        ranked_docs.sort(
            key=lambda x: (
                camera_score(
                    x[0].properties["camera"]
                ),
                x[1]
            ),
            reverse=True
        )

    elif "battery" in query.lower():

        ranked_docs.sort(
            key=lambda x: (
                battery_score(
                    x[0].properties["battery"]
                ),
                x[1]
            ),
            reverse=True
        )

    elif "rating" in query.lower():

        ranked_docs.sort(
            key=lambda x:
            float(
                x[0].properties["rating"]
            )
            if x[0].properties["rating"] != "nan"
            else 0,
            reverse=True
        )

    else:

        ranked_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

    top_3 = ranked_docs[:3]

    model_names = [
        "tinyllama",
        "qwen2.5:0.5b",
        "smollm2:360m"
    ]

    recommendations = []

    for idx, (doc, score) in enumerate(
        top_3
    ):

        recommendations.append(
            {
                "model_name": model_names[idx],
                "mobile_name": doc.properties["title"],
                "brand": doc.properties["brand"],
                "price": doc.properties["price"],
                "camera": doc.properties["camera"],
                "battery": doc.properties["battery"],
                "storage": doc.properties["storage"],
                "rating": doc.properties["rating"],
                "processor": doc.properties["processor"],
                "os": doc.properties["os"],
                "score": round(score * 100, 2)
            }
        )

    return {

        "user_query": query,

        "all_chunks_created":
            all_chunks_created,

        "retrieved_chunks":
            retrieved_chunks,

        "top_most_recommendation":
            recommendations[0],

        "total_suggestions":
            len(recommendations),

        "all_ranked_suggestions":
            recommendations
    }


@app.post("/recommend")
def recommend_mobile(
    request: QueryRequest
):
    return get_recommendation(
        request.query
    )


@app.get("/recommend")
def recommend_mobile_get(
    query: str
):
    return get_recommendation(
        query
    )