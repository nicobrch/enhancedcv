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


def log(msg: str) -> None:
    print(f"[enhancedcv] {msg}", flush=True)


def scrape_job(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        log(f"scrape_job: state keys: {list(state.keys())}")
        url = state.get("url")
        if not url:
            raise ValueError("URL not found in state")

        log(f"scrape_job: fetching URL: {url}")
        resp = requests.get(url, timeout=30)
        log(f"scrape_job: HTTP status {resp.status_code}")
        html = resp.text
        log(f"scrape_job: received {len(html)} bytes")
        soup = BeautifulSoup(html, "html.parser")
        text = " ".join(soup.stripped_strings)
        state["job_text"] = text[:5000]  # limit context size
        log(f"scrape_job: extracted text length={len(state['job_text'])}")
        return state
    except Exception as e:
        log(f"scrape_job: ERROR: {str(e)}")
        log(f"scrape_job: State dump: {json.dumps(state, default=str)}")
        raise
