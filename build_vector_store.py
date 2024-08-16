from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')

examples = [
    {"input": "Show mram report", "output": """{"action" : "open", "link": "mram-collection-report"}"""},
    {"input": "Show mran collection report", "output": """{"action" : "open", "link": "mram-collection-report"}"""}
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples, persist_directory="./chroma_db")


