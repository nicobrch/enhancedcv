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


def analyze_job(state: State) -> State:
    try:
        log("analyze_job: starting LLM extraction")
        prompt = """
        Extract structured job info as JSON with fields:
        role, company, skills, requirements, description.

        Text: {job_text}
        """
        response = llm.invoke(prompt.format(job_text=state["job_text"]))
        log(f"analyze_job: got response length={len(str(response))}")
        state["job_json"] = response
        return state
    except Exception as e:
        log(f"analyze_job: ERROR: {e}")
        raise


def adapt_resume(state: State) -> State:
    try:
        log(f"adapt_resume: reading resume from {state['resume_path']}")

        with open(state["resume_path"], "r", encoding="utf-8") as f:
            resume_content = f.read()

        log(f"adapt_resume: base resume chars={len(resume_content)}")

        prompt = f"""
        You are given a resume in Markdown format with the following sections:
        {resume_content}
        
        Job description JSON:
        {state["job_json"]}

        Adapt the resume to better fit the job description, following these instructions:
        1. Maintain the original Markdown structure and formatting.
        2. Emphasize the most relevant experience and skills for the job.
        3. Keep the original language of the resume (Spanish or English).
        4. Focus on modifying the "Personal Profile", "Experience", and "Additional Information" sections.
        5. Do not add any new sections or remove existing ones.
        6. Use the job description to enhance relevant details in the resume.
        7. Return the complete adapted resume in Markdown format.
        """

        adapted_markdown = llm.invoke(prompt)
        log(
            f"adapt_resume: adapted Markdown chars={len(str(adapted_markdown))}")
        state["adapted_markdown"] = str(adapted_markdown)
        return state
    except Exception as e:
        log(f"adapt_resume: ERROR: {e}")
        raise
