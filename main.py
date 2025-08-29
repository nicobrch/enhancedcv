import sys
import json
import re
import requests
from typing import TypedDict, Any, Dict, Optional
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


class JobGraphState(TypedDict, total=False):
    url: str
    page_content: str
    summary: Dict[str, Any]


def _clean_text(text: str, max_chars: int = 12000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def scrape_page(state: JobGraphState) -> JobGraphState:
    url = state.get("url")
    if not url:
        raise ValueError("Missing 'url' in state.")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JobScraper/1.0; +https://example.com/bot)"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for tag_name in ["nav", "footer", "header"]:
        for t in soup.find_all(tag_name):
            t.decompose()

    text = soup.get_text(" ", strip=True)
    text = _clean_text(text)

    return {"page_content": text}


def summarize_content(state: JobGraphState) -> JobGraphState:
    page_content = state.get("page_content", "")
    if not page_content:
        raise ValueError("Nothing to summarize; 'page_content' is empty.")

    llm = ChatOllama(model="gpt-oss:20b", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert HR analyst. Extract the essential info from a job posting."
            ),
            (
                "human",
                "From the following page text, produce a STRICT JSON object with keys exactly:\n"
                '  "company_name": string,\n'
                '  "location": string (city/remote/hybrid),\n'
                '  "salary_range": string (if present),\n'
                '  "benefits": string[] (if present),\n'
                '  "job_title": string,\n'
                '  "experience_needed": string,\n'
                '  "skills_needed": string[],\n'
                '  "keywords": string[]\n'
                "Rules:\n"
                "- If unsure, make a reasonable guess from context.\n"
                "- Keep arrays concise (3-10 items).\n"
                "- Always include all keys, even if empty.\n"
                "- Do not add extra keys or commentary.\n\n"
                "PAGE TEXT:\n{page_content}"
            ),
        ]
    )

    chain: RunnableSequence = prompt | llm | StrOutputParser()
    raw = chain.invoke({"page_content": page_content})

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Model did not return a dict.")

    return {"summary": data}


def build_graph():
    graph = StateGraph(JobGraphState)
    graph.add_node("scrape", scrape_page)
    graph.add_node("summarize", summarize_content)
    graph.add_edge(START, "scrape")
    graph.add_edge("scrape", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


def main(argv: Optional[list[str]] = None) -> None:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python main.py <job_offer_url>")
        sys.exit(1)

    url = argv[0]
    app = build_graph()
    result = app.invoke({"url": url})

    print(json.dumps(result.get("summary", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
