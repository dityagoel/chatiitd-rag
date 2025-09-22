from __future__ import annotations

import json
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatLiteLLM

import config

# ------------------------------
# Service clients & models
# ------------------------------
_qdrant = QdrantClient(url=config.QDRANT_URL)
_embedder = SentenceTransformer(config.EMBED_MODEL)


# ------------------------------
# Internal helpers
# ------------------------------

def _format_qdrant_results(results: List[Any]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for r in results or []:
        payload = getattr(r, "payload", {}) or {}
        item = {
            "text": payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or payload.get("page_content")
            or str(payload),
            "metadata": payload,
        }
        score = getattr(r, "score", None)
        if score is not None:
            item["score"] = score
        formatted.append(item)
    return formatted


def _qdrant_search(collection: str, query: str, k: int) -> List[Dict[str, Any]]:
    if not query:
        return []
    vec = _embedder.encode(query).tolist()
    res = _qdrant.search(collection_name=collection, query_vector=vec, limit=int(k))
    return _format_qdrant_results(res)


# ------------------------------
# LangChain Tools (Rules / Courses)
# ------------------------------

def _build_tools(default_k: int) -> list:
    def _rules(q: str) -> str:
        """Return JSON list of rule passages with text and metadata."""
        return json.dumps(_qdrant_search("rules", q, default_k), ensure_ascii=False)

    def _courses(q: str) -> str:
        """Return JSON list of course passages with text and metadata."""
        return json.dumps(_qdrant_search("courses", q, default_k), ensure_ascii=False)

    rules_tool = StructuredTool.from_function(
        name="search_rules",
        func=_rules,
        description=(
            "Retrieve relevant passages from IIT Delhi academic rules, policies, regulations, UG/PG rules, "
            "grading, attendance, credits, and registration. Input: concise search query. "
            "Output: JSON list of passages with `text` and `metadata`."
        ),
    )

    courses_tool = StructuredTool.from_function(
        name="search_courses",
        func=_courses,
        description=(
            "Retrieve relevant passages about IIT Delhi courses: titles, descriptions, prerequisites, and credits. "
            "Input: concise search query. Output: JSON list of passages with `text` and `metadata`."
        ),
    )

    return [rules_tool, courses_tool]


# ------------------------------
# ReAct Agent setup
# ------------------------------

def _build_prompt() -> ChatPromptTemplate:
    system = (
        "You are ChatIITD, a ReAct agent restricted to IIT Delhi academic topics.\n"
        "Follow this policy:\n"
        "- Only answer IIT Delhi academic questions (rules, courses, curricula, regulations, academic policies).\n"
        "- If the user asks something outside this scope, reply exactly: 'Sorry, I can only serve academic requests' and stop.\n"
        "- Use the provided tools to retrieve information. If both rules and courses are relevant, call both tools.\n"
        "- If no relevant passages are found, reply exactly: 'I could not find relevant information in the provided academic documents.'\n\n"
        "Answering:\n"
        "- Base answers strictly on retrieved passages; do not speculate.\n"
        "- Be concise and clear.\n"
        "- When possible, cite rule sections or course names as they appear in the retrieved text."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("system", "Available tools:\n{tools}"),
            ("system", "Use tool names exactly as listed: {tool_names}"),
            ("human", "{input}"),
        ]
    )
    return prompt


def _build_agent(default_k: int) -> AgentExecutor:
    tools = _build_tools(default_k)

    llm = ChatLiteLLM(
        model=config.LITELLM_MODEL,
        temperature=float(getattr(config, "TEMPERATURE", 0.2)),
        max_tokens=getattr(config, "MAX_TOKENS", 512),
    )

    prompt = _build_prompt()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return executor


# ------------------------------
# Public entry point
# ------------------------------

def query_bot(user_query: str, top_k: int = 5, show_sources: bool = False) -> str:
    """Run the ReAct agent. The agent will call tools to query rules/courses as needed."""
    try:
        executor = _build_agent(default_k=top_k)
        result = executor.invoke({"input": user_query})
        output = result.get("output") if isinstance(result, dict) else str(result)
        return output or "I could not find relevant information in the provided academic documents."
    except Exception as e:
        return f"Error while processing the query: {e}"


# ------------------------------
# Simple CLI for local testing
# ------------------------------
if __name__ == "__main__":
    try:
        q = input()
        print(f"\nQ: {q}\nA: {query_bot(q)}\n")
    except KeyboardInterrupt:
        pass