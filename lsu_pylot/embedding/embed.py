import pandas as pd
import os
import tiktoken
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOMAIN = "developer.mozilla.org"


def remove_newlines(series):
    series = series.str.replace("\n", " ")
    series = series.str.replace("\\n", " ")
    series = series.str.replace("  ", " ")
    series = series.str.replace("  ", " ")
    return series


def do_process():
    texts = []
    folder_name = f"text/{DOMAIN}/"
    for f_name in os.listdir(folder_name):
        print(f"processing file {f_name}")
        with open(os.path.join(folder_name, f_name), "r", encoding="UTF-8") as f:
            text = f.read()
            filename = f_name[:-4].replace("_", "/")
            if filename.endswith(".txt") or "users/fxa/login" in filename:
                continue
            texts.append((filename, text))

    df = pd.DataFrame(texts, columns=["fname", "text"])

    df["text"] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv("processed/scraped.csv")
    print("Finished")


def create_embedding(x):
    return (
        openai.embeddings.create(input=x, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )


def do_embeddings():
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv("processed/scraped.csv", index_col=0)
    df.columns = ["title", "text"]
    print(f"Len of scraped is {len(df)}")
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    chunk_size = 3000  # max number of tokens

    text_splitter = RecursiveCharacterTextSplitter(
        length_function=len,
        chunk_size=chunk_size,
        chunk_overlap=0,
        add_start_index=False,
    )

    shortened = []

    for row in df.iterrows():
        if row[1]["text"] is None:
            continue
        if row[1]["n_tokens"] > chunk_size:
            print(f"Row has {row[1]['n_tokens']} tokens")
            chunks = text_splitter.create_documents(row[1]["text"])
            for chunk in chunks:
                shortened.append(chunk.page_content)
        else:
            shortened.append(row[1]["text"])

    df = pd.DataFrame(shortened, columns=["text"])
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    print(f"Total chunks are {len(shortened)}")

    df["embeddings"] = df.text.apply(create_embedding)
    df.to_csv("processed/embeddings.csv")


if __name__ == "__main__":
    do_embeddings()
