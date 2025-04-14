import os
from dotenv import load_dotenv
from typing import List
from google import generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def format_context(chunks: List[str], max_chunks: int = 5) -> str:
    """Format document chunks into a readable prompt context."""
    context = ""
    for i, chunk in enumerate(chunks[:max_chunks]):
        context += f"Source {i+1}:\n{chunk.strip()}\n\n"
    return context.strip()


def generate_answer(query: str, chunks: List[str]) -> str:
    if not chunks:
        return "I couldn't find any relevant content to answer your question."

    context = format_context(chunks)

    prompt = f"""You are a helpful assistant answering questions based on uploaded documents.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating answer: {e}"

def get_llm_help(prompt: str = "") -> str:
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating expanded query: {e}"
    
def llm_query_expansion(query: str, prev_queries: List[str]) -> str:
    previous_queries_section = (
        f"Previous Queries: {', '.join(prev_queries)}\n" if prev_queries else ""
    )
    prompt = f"""You are a helpful assistant that expands user queries to be more specific and detailed. You can use previous queries as context. Avoid making the expanded query similar to previous queries. JUST GIVE THE EXPANDED QUERY WITHOUT ANY EXPLANATION.

User Query: {query}
{previous_queries_section}
Expanded Query:"""
    return get_llm_help(prompt)