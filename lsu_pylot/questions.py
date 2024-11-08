import traceback
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List
from scipy import spatial
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc_model = "multilingual-e5-large"
pc_index_name = "p-docs"


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings"""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_context(question, df, max_len=1800):
    """
    Creates the context for a question by finding the most similar documents
    """
    q_embeddings = (
        openai.embeddings.create(input=question, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    for i, row in df.sort_values("distances", ascending=True).iterrows():
        # Add length of the text to the cur_len
        cur_len += row["n_tokens"] + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="gpt-4o-mini",
    question="What is the meaning of life?",
    max_len=1800,
    debug=False,
    max_tokens=150,
    stop_sequence=None,
):
    """Answer a question based on the most similar context"""
    context = create_context(question, df, max_len=max_len)
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know.\"
                Try to site sources to the links in the context when possible. \n\nContext: {context}\n\n--\n\nQuestion: {question}\nSource:\nAnswer:
                """,
                }
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


def pinecone_search(query):
    pc = Pinecone(api_key=pinecone_api_key)
    query_embedding = pc.inference.embed(
        model=pc_model, inputs=[query], parameters={"input_type": "query"}
    )
    index = pc.Index(pc_index_name)
    results = index.query(
        vector=query_embedding[0].values,
        namespace="article1",
        top_k=3,
        include_values=False,
        include_metadata=True,
    )
    return results


def pinecone_answer(question):
    matches = pinecone_search(question)
    match_texts = [m["metadata"]["text"] for m in matches["matches"]]
    context = "\n\n###\n\n".join(match_texts)
    try:
        llm_response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know.\"
                    \n\nContext: {context}\n\n--\n\nQuestion: {question}\nSource:\nAnswer:
                    """,
                }
            ],
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return llm_response.choices[0].message.content
    except Exception as e:
        print(traceback.format_exc())
        return ""
