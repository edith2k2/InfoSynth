import os
from dotenv import load_dotenv
from typing import List
from google import generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


def format_context(chunks: List[str], scores: List[float], max_chunks: int = 5) -> str:
    """Format document chunks into a readable prompt context."""
    context = ""
    # print('\n\nScores and len(scores)')
    # print(scores)
    # print(len(scores))
    # print("\n\n")
    # passing the relevance scores for each chunk to the LLM for better RAG
    for i, (chunk, score) in enumerate(zip(chunks[:max_chunks], scores[:max_chunks])):
        context += f"**Source {i+1} (Relevance: {score:.2f}):**\n{chunk.strip()}\n\n"
    return context.strip()


# how can we improve the RAG integration in our codebase?
# # 1. Use a more sophisticated chunking strategy to ensure that the chunks are semantically coherent.
# # 2. Implement a caching mechanism to avoid re-processing the same files multiple times.
# # 3. Use a more advanced model for generating answers, such as a transformer-based model.
# # 4. Implement a feedback loop to improve the model's performance over time based on user interactions.
def generate_answer(query: str, chunks: List[str], scores: List[float]) -> str:
    if not chunks:
        return "I couldn't find any relevant content to answer your question."

    context = format_context(chunks, scores)
    # print("\n\ncontext from generate_answer in llm.py:", context, "\n\n")

    prompt = f"""You are a helpful assistant answering questions based on uploaded documents. With each document, I am also providing you the relevance score of the document to the question.

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