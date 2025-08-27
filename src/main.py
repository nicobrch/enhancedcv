import requests
import traceback
import subprocess
import json

from typing import TypedDict, Dict, Any
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM
# from langchain_openai import OpenAI

llm = OllamaLLM(model="gpt-oss:20b")
# llm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="123")


class State(TypedDict, total=False):
    url: str
    resume_path: str
    job_text: str
    job_json: str
    adapted_markdown: str
    output_file: str
