#!/usr/bin/env python3
"""
CLI tool (Assignment 3): PDF Retrieval-Augmented Generation.

1. Retrieve the most similar pages (chunks) to the query using
   TF-IDF + cosine similarity (offline).
2. Build an augmented prompt from (query, chunks).
3. Send the augmented prompt to an LLM and print the prompt + the response.

Usage:
    python search_pdf.py --pdf book.pdf --query "how do for loops work"
    python search_pdf.py --pdf book.pdf --query "recursion" --top-k 3
"""
import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context pages. If the context does not contain the answer, say "
    "that the provided pages do not contain enough information. "
    "Mention the page numbers you relied on."
)


def load_pages(pdf_path):
    """Return a list of page texts (index 0 = page 1)."""
    try:
        reader = PdfReader(pdf_path)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {pdf_path}")
    except Exception as e:
        sys.exit(f"Error: could not read PDF: {e}")

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(" ".join(text.split()))  # normalize whitespace
    return pages


def search(pages, query, top_k=1):
    """Return list of (page_number, score) sorted by similarity, best first."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    page_matrix = vectorizer.fit_transform(pages)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, page_matrix).ravel()

    ranked = scores.argsort()[::-1][:top_k]
    return [(int(i) + 1, float(scores[i])) for i in ranked]


def build_augmented_prompt(query, chunks):
    """Combine the query and retrieved chunks into one augmented prompt.

    chunks: list of (page_number, text).
    """
    context = "\n\n".join(
        f"[Page {page_no}]\n{text}" for page_no, text in chunks
    )
    return (
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based only on the context above."
    )


def get_client():
    """Create an OpenAI-compatible client from environment / .env settings."""
    load_dotenv(override=True)  # values in .env win over shell variables
    base_url = os.getenv("LLM_BASE_URL") or None
    api_key = os.getenv("LLM_API_KEY")

    is_local = bool(base_url) and (
        "localhost" in base_url or "127.0.0.1" in base_url
    )
    if not api_key:
        if is_local:
            api_key = "not-needed"  # local servers such as Ollama ignore it
        else:
            sys.exit(
                "Error: LLM_API_KEY is not set. Create a .env file "
                "(copy .env.example) and add your API key."
            )
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_answer(client, model, augmented_prompt):
    """Send the augmented prompt to the LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
            temperature=0.2,
        )
    except Exception as e:
        sys.exit(f"Error: LLM request failed: {e}")
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant PDF pages and answer a query with an LLM."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of pages (chunks) given to the LLM (default: 3)")
    parser.add_argument("--max-chars", type=int, default=3000,
                        help="Max characters per page put in the prompt (0 = all)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (overrides LLM_MODEL in .env)")
    args = parser.parse_args()

    pages = load_pages(args.pdf)
    non_empty = sum(1 for p in pages if p.strip())
    if non_empty == 0:
        sys.exit("Error: no selectable text found (is this a scanned PDF?).")

    print(f"Loaded {len(pages)} pages ({non_empty} with text).\n")

    results = search(pages, args.query, args.top_k)
    results = [(p, s) for p, s in results if s > 0]
    if not results:
        sys.exit("No matching page found for this query.")

    chunks = []
    for page_no, _score in results:
        text = pages[page_no - 1]
        if args.max_chars and len(text) > args.max_chars:
            text = text[: args.max_chars] + " ..."
        chunks.append((page_no, text))

    print("Retrieved pages: " + ", ".join(
        f"{p} (score {s:.4f})" for p, s in results) + "\n")

    augmented_prompt = build_augmented_prompt(args.query, chunks)

    print("=" * 70)
    print("AUGMENTED PROMPT")
    print("=" * 70)
    print(augmented_prompt)
    print()

    client = get_client()
    model = args.model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    answer = generate_answer(client, model, augmented_prompt)

    print("=" * 70)
    print(f"GENERATED RESPONSE (model: {model})")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()