# pipeline/pdf_utils.py

import os
import uuid
from io import BytesIO

import google.generativeai as genai
from PyPDF2 import PdfReader

from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
from pipeline.relation_extractor import extract_relations
from pipeline.graph_builder import build_and_store_graph


# ----------------------------------------------------
# Gemini config
# ----------------------------------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

# Default model – you can override with GEMINI_MODEL_NAME in .env
DEFAULT_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")

genai.configure(api_key=API_KEY)


# ----------------------------------------------------
# PDF → text
# ----------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract full text from PDF using PyPDF2."""
    reader = PdfReader(BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


# ----------------------------------------------------
# Summarization with Gemini
# ----------------------------------------------------
def _chunk_for_summary(text: str, max_chars: int = 24000):
    if not text:
        return []
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def summarize_with_gemini(text: str, model_name: str | None = None) -> str:
    """
    Summarize text using Gemini (with chunking for long PDFs).
    """
    if not text.strip():
        return ""

    model_name = model_name or DEFAULT_MODEL_NAME
    model = genai.GenerativeModel(model_name)

    chunks = _chunk_for_summary(text)
    partial_summaries: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        prompt = (
            "You are an expert Knowledge Graph Architect. Your task is to create a highly detailed "
            "summary of the following text specifically for Entity-Relation extraction.\n\n"
    "STRICT INSTRUCTIONS:\n"
    "1. Entity Preservation: Do NOT generalize. Keep specific names of people, organizations, dates, and technical terms.\n"
    "2. No Pronouns: Replace 'he', 'she', 'it', or 'they' with the actual names (e.g., write 'Elon Musk' instead of 'he').\n"
    "3. Action-Oriented: Write sentences in clear Subject-Action-Object format (e.g., 'Company A acquired Company B').\n"
    "4. Density: Include as many factual details as possible while condensing the word count.\n\n"
            f"Chunk {i}:\n{chunk}"
        )
        resp = model.generate_content(prompt)
        partial_summaries.append(resp.text.strip())

    if not partial_summaries:
        return ""

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = "\n\n".join(partial_summaries)
    final_prompt = (
        "Combine these partial summaries into one coherent, concise summary "
        "of 6–8 sentences:\n\n" + combined
    )
    final_resp = model.generate_content(final_prompt)
    return final_resp.text.strip()


# ----------------------------------------------------
# Main: PDF upload → summary → KG
# ----------------------------------------------------
def process_pdf_upload(uploaded_file, run_relations: bool = True):
    """
    Pipeline:
      1. Read PDF bytes.
      2. Extract FULL text.
      3. Summarize with Gemini.
      4. Save SUMMARY + FULL TEXT into data/txt/*.txt.
      5. Build entities & relations **only from the summary** (faster):
           summary → chunk_tokens → extract_graph → extract_relations → build_and_store_graph.
      6. Return summary, txt_path, total_entities, total_relations.
    """

    # 1) PDF bytes
    pdf_bytes = uploaded_file.read()

    # 2) Full text from PDF
    full_text = extract_text_from_pdf_bytes(pdf_bytes)

    # 3) Summary with Gemini
    summary = summarize_with_gemini(full_text)

    # 4) Save TXT (both summary + full text)
    os.makedirs(os.path.join("data", "txt"), exist_ok=True)
    file_stem = os.path.splitext(uploaded_file.name)[0]
    txt_filename = f"{file_stem}_{uuid.uuid4().hex[:8]}.txt"
    txt_path = os.path.join("data", "txt", txt_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY (used for KG) ===\n")
        f.write(summary)
        f.write("\n\n=== FULL TEXT (for reference) ===\n")
        f.write(full_text)

    # 5) Build KG from the SUMMARY only (fast mode)
    target_text = summary

    chunks = chunk_tokens(target_text)
    all_entities: list = []
    all_relations: list = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"pdf_summary_chunk_{uuid.uuid4().hex[:8]}"

        # entities + base relations
        graph_data = extract_graph(chunk)
        entities = graph_data.get("entities", [])
        base_relations = graph_data.get("relations", [])

        # optional refined relations
        refined_relations = []
        if run_relations:
            refined_relations = extract_relations(chunk)

        relations_chunk = base_relations + refined_relations

        all_entities.extend(entities)
        all_relations.extend(relations_chunk)

        # store in Neo4j
        build_and_store_graph(chunk_id, chunk, entities, relations_chunk)

    return {
        "summary": summary,
        "txt_path": txt_path,
        "total_entities": len(all_entities),
        "total_relations": len(all_relations),
    }
