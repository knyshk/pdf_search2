# PDF Retrieval-Augmented Generation (CLI)

A command line tool that takes a PDF with selectable text and a query, retrieves the most relevant pages, builds an **augmented prompt** from `(query, chunks)`, sends it to an LLM, and prints both the augmented prompt and the generated response.

This extends the earlier page-similarity search (Assignment 1): instead of printing the best page and its similarity score, the retrieved pages are now passed to an LLM.

## How it works

1. Text is extracted from every page using `pypdf`.
2. Each page and the query are converted into TF-IDF vectors (unigrams and bigrams, English stop words removed).
3. Cosine similarity is computed between the query and every page, and the top-k pages are selected as the **chunks** (one page = one chunk).
4. An augmented prompt is built: the retrieved chunks (labelled with page numbers) followed by the question.
5. The augmented prompt is sent to an LLM through an OpenAI-compatible API.
6. The tool prints the augmented prompt and the generated response.

Retrieval runs offline. Only the final generation step needs an LLM (API key or local model).

## Requirements

- Linux with Python 3.9+
- A PDF with selectable text (not a scanned image)
- An LLM: **an API key is required** for OpenAI or Groq. Using a local Ollama model needs no key.

## Setup

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl

mkdir -p ~/pdf_search && cd ~/pdf_search
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:

```
pypdf
scikit-learn
openai
python-dotenv
```

If you already have the old virtual environment from Assignment 1, just run `pip install -r requirements.txt` again; it will install only the two new packages.

## Configure the LLM (API key)

Create your `.env` file from the template:

```bash
cp .env.example .env
nano .env
```

Pick **one** of the three options.

### Option A: OpenAI (default)

1. Create a key at https://platform.openai.com/api-keys (needs a billed account; usage for this assignment costs a fraction of a cent).
2. Put this in `.env`:

```
LLM_API_KEY=sk-your-openai-key-here
LLM_MODEL=gpt-4o-mini
```

### Option B: Groq (free tier)

1. Create a key at https://console.groq.com/keys.
2. Put this in `.env`:

```
LLM_API_KEY=gsk_your-groq-key-here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### Option C: Ollama (local, no key)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

Put this in `.env`:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

Model names change over time; if a model is rejected, check the provider's current model list and update `LLM_MODEL`.

Keep `.env` private. If you use git, add it to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

## Get a PDF

Example (the free book *Think Python*):

```bash
curl -L -o book.pdf https://greenteapress.com/thinkpython2/thinkpython2.pdf
```

Or copy your own PDF into the folder as `book.pdf`.

## Usage

```bash
python search_pdf.py --pdf book.pdf --query "what is a recursive function"
```

Options:

| Option | Description | Default |
|---|---|---|
| `--pdf` | Path to the PDF file | required |
| `--query` | Question / search query | required |
| `--top-k` | Number of pages (chunks) given to the LLM | 3 |
| `--max-chars` | Max characters per page placed in the prompt (0 = all) | 3000 |
| `--model` | LLM model name (overrides `LLM_MODEL` in `.env`) | `.env` value, else `gpt-4o-mini` |

More examples:

```bash
python search_pdf.py --pdf book.pdf --query "dictionaries and hashing" --top-k 5
python search_pdf.py --pdf book.pdf --query "file handling" --model gpt-4o-mini
python search_pdf.py --pdf book.pdf --query "what is a recursive function" > output.txt
```

## Example output

```
Loaded 244 pages (240 with text).

Retrieved pages: 85 (score 0.3412), 87 (score 0.2874), 84 (score 0.2511)

======================================================================
AUGMENTED PROMPT
======================================================================
Context:
[Page 85]
<text of page 85>

[Page 87]
<text of page 87>

[Page 84]
<text of page 84>

Question: what is a recursive function

Answer based only on the context above.

======================================================================
GENERATED RESPONSE (model: gpt-4o-mini)
======================================================================
<LLM answer, citing page numbers>
```

## Project structure

```
pdf_search/
├── search_pdf.py
├── requirements.txt
├── .env.example
├── .env            (your private keys, not committed)
├── book.pdf
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `LLM_API_KEY is not set` | Create `.env` from `.env.example` and add your key. Run the command from the folder containing `.env`. |
| `ModuleNotFoundError: openai` or `dotenv` | Activate the venv (`source venv/bin/activate`) and run `pip install -r requirements.txt`. |
| `401 Unauthorized` | The key is wrong, expired, or does not match the provider/`LLM_BASE_URL`. |
| `429` / quota errors | Your account has no credit or hit a rate limit. Use Groq or Ollama instead. |
| Model not found | Update `LLM_MODEL` to a model your provider currently offers. |
| Error shows a different model than the one in `.env` | An old value is exported in your shell (e.g. from `source .env`). Run `unset LLM_MODEL LLM_API_KEY LLM_BASE_URL` or open a new terminal. The script now lets `.env` override the shell. |
| Connection refused (Ollama) | Make sure Ollama is running (`ollama serve`) and the model was pulled. |
| No selectable text found | The PDF is scanned; this tool needs a text layer. |

## Limitations

- TF-IDF matches on words, not meaning. A query using different wording than the page may rank poorly, which then gives the LLM weak context. Sentence embeddings (e.g. `sentence-transformers`) would improve retrieval.
- Chunks are whole pages; very long pages are truncated to `--max-chars`.
- The LLM is instructed to answer only from the provided pages, but generated output can still contain mistakes, so check it against the cited pages.
- Scanned PDFs without a text layer are not supported (the tool exits with an error).
- The page text is sent to the LLM provider when using OpenAI or Groq. Use Ollama if the PDF must stay local.