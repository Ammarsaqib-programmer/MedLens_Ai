# 🩺 MediLens AI

Agentic AI assistant that explains medical reports and identifies medicines in simple English / Roman Urdu / Urdu.

**Pipeline:** Upload → Extract → Analyze → Retrieve (RAG) → Verify → Explain → Guide

## Agents
| Agent | Role |
|---|---|
| Extraction | Reads values from PDF/photo (Gemini vision, or OCR + rules offline) |
| Analysis | Normal/High/Low + severity (deterministic code, not the LLM) |
| Medical RAG | TF-IDF retrieval over a curated medical knowledge base |
| Verification | Value-in-source, plausibility, unit and flag consistency checks |
| Explanation | Grounded plain-language explanation (+ guardrail against dosing/prescribing text) |
| Guidance | Urgency level + next steps (always refers to a doctor) |
| Medicine | Pack photo → name/ingredient → match to `Medicine_research.xlsx` (DRAP-based), asks confirmation if unsure |

## Run on Streamlit Community Cloud
1. Repo files: `app.py`, `requirements.txt`, `packages.txt`, `Medicine_research.xlsx`, `README.md`
2. share.streamlit.io → New app → select repo → main file `app.py`
3. (Recommended) App → Settings → Secrets (Groq OR Gemini; Groq is used if both are set):
   ```toml
   GROQ_API_KEY = "gsk_..."          # free: console.groq.com
   # GEMINI_API_KEY = "..."          # optional: aistudio.google.com (also reads scanned PDFs)
   ```
   Without a key the app still works in offline mode (OCR/PDF text + rule-based agents).

## Disclaimer
Educational tool only. Not a medical device. It does not diagnose, prescribe or change treatment.
