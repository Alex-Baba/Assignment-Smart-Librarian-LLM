import os, re, json
from typing import Any, Dict, List
from openai import OpenAI

from chatbot.rag import search_books, index_books
from chatbot.tools import get_summary_by_title
from chatbot.extras import sanitized_or_warning, tts_say_cli, generate_cover_png

# System prompt for LLM to recommend a book and explain rationale
SYSTEM_PROMPT = """You are Smart Librarian. Recommend a single best-fit book given RAG search results.
Return a JSON object with keys:
"title" (exact recommended book title) and "why" (1-3 sentence rationale).
No extra keys.
"""

def call_llm_for_recommendation(user_query: str, rag_hits: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Calls the LLM to recommend a book based on RAG search results and user query.
    Returns a dict with 'title' and 'why' keys.
    """
    client = OpenAI()
    resp = client.responses.create(
        model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "system", "content": "RAG_CONTEXT: " + json.dumps({"results": rag_hits})}
        ]
    )
    txt = resp.output_text
    # Extract JSON object from LLM output
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return {"title": rag_hits[0]["title"], "why": "Top semantic match from your query."}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"title": rag_hits[0]["title"], "why": "Top semantic match from your query."}

def main():
    """
    Main CLI loop for Smart Librarian.
    Indexes books, then repeatedly accepts user queries, sanitizes input, performs RAG search,
    gets LLM recommendation, speaks and generates cover, and prints full summary.
    """
    n = index_books()
    print(f"Indexed/updated {n} books.\n")
    print("Smart Librarian CLI. Type your request (or 'exit'):\n")
    while True:
        user = input("You> ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        # Sanitize input for safety and respectfulness
        clean = sanitized_or_warning(user)
        if clean is None:
            print("Assistant> Let’s keep it respectful. Try another question. 🙂")
            continue
        user = clean

        # Search for relevant books using RAG
        hits = search_books(user, k=3)
        # Get LLM recommendation based on search results
        rec = call_llm_for_recommendation(user, hits)
        title = rec.get("title") or (hits[0]["title"] if hits else None)
        why = rec.get("why", "")
        print(f"\nRecommendation: {title}\nWhy: {why}\n")

        # Optionally speak recommendation and generate cover image
        if title:
            try:
                tts_say_cli(f"My pick is {title}. {why}")
            except Exception:
                pass
            try:
                path = generate_cover_png(title, why, out_path=f"cover_{title.replace(' ','_')}.png")
                print(f"Cover image saved to: {path}\n")
            except Exception:
                print("Could not generate cover image (optional).\n")

        # Print full book summary
        full = get_summary_by_title(title) if title else "No title selected."
        print("Full summary:\n" + full + "\n")

if __name__ == "__main__":
    # Entry point for CLI
    main()