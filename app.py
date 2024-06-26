from langchain_community.llms import HuggingFaceHub
import os
from dotenv import load_dotenv


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
load_dotenv()
llm = HuggingFaceHub(
    repo_id="google/gemma-7b", 
    # repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5,"max_new_tokens":1000}
)
