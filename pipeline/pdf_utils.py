# pipeline/pdf_utils.py

import os
import uuid
import time
import random
import logging
from io import BytesIO

import google.generativeai as genai
from PyPDF2 import PdfReader

from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
from pipeline.relation_extractor import extract_relations_from_text as extract_relations
from pipeline.graph_builder import build_and_store_graph

log = logging.getLogger("pdf_utils")

# ----------------------------------------------------
# Gemini config
# ----------------------------------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    log.warning("GOOGLE_API_KEY not found. PDF summarization will fail.")


# ----------------------------------------------------
# Helper: Retry Logic
# ----------------------------------------------------
def _generate_with_retry(model, prompt: str, retries: int = 3):
    """
    Calls Gemini with exponential backoff for 429 errors.
    """
    for attempt in range(1, retries + 1):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Resource exhausted" in err_str:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                log.warning(f"Gemini 429 Limit hit. Retrying in {wait_time:.1f}s (Attempt {attempt}/{retries})")
                time.sleep(wait_time)
                continue
            raise e
    raise RuntimeError("Gemini 429: Resource exhausted after max retries.")


# ----------------------------------------------------
# PDF → text
# ----------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract full text from PDF using PyPDF2."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except Exception as e:
        log.error(f"Failed to extract text from PDF: {e}")
        return ""


# ----------------------------------------------------
# Summarization with Gemini
# ----------------------------------------------------
def _chunk_for_summary(text: str, max_chars: int = 24000):
    if not text:
        return []
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def summarize_with_gemini(text: str, model_name: str | None = None) -> str:
    """
    Summarize text using Gemini. Returns None or empty string if it fails.
    """
    if not API_KEY:
        log.error("Cannot summarize: No API Key.")
        return ""

    if not text.strip():
        return ""

    model_name = model_name or DEFAULT_MODEL_NAME
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        log.error(f"Failed to init Gemini model: {e}")
        return ""

    chunks = _chunk_for_summary(text)
    partial_summaries: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        prompt = (
            "You are an expert Knowledge Graph Architect. Your task is to create a highly detailed "
            "summary of the following text specifically for Entity-Relation extraction.\n\n"
            "STRICT INSTRUCTIONS:\n"
            "1. Entity Preservation: Do NOT generalize. Keep specific names of people, organizations, dates, and technical terms.\n"
            "2. No Pronouns: Replace 'he', 'she', 'it', or 'they' with the actual names.\n"
            "3. Action-Oriented: Write sentences in clear Subject-Action-Object format.\n"
            "4. Density: Include as many factual details as possible while condensing the word count.\n\n"
            f"Chunk {i}:\n{chunk}"
        )
        try:
            resp = _generate_with_retry(model, prompt)
            if resp and resp.text:
                partial_summaries.append(resp.text.strip())
        except Exception as e:
            log.error(f"Failed to summarize chunk {i}: {e}")

    if not partial_summaries:
        return ""

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = "\n\n".join(partial_summaries)
    final_prompt = (
        "Combine these partial summaries into one coherent, concise summary "
        "of 6–8 sentences:\n\n" + combined
    )
    try:
        final_resp = _generate_with_retry(model, final_prompt)
        return final_resp.text.strip()
    except Exception as e:
        log.error(f"Failed to generate final summary: {e}")
        return combined[:10000]


# ----------------------------------------------------
# Main: PDF upload → summary → KG
# ----------------------------------------------------
def process_pdf_upload(uploaded_file, run_relations: bool = True, user_id: str | None = None):
    """
    Pipeline:
      1. Read PDF bytes.
      2. Extract FULL text.
      3. Summarize with Gemini.
      4. Fallback if summary fails.
      5. Build entities & relations (with user_id for memory).
    """

    # 1) PDF bytes
    pdf_bytes = uploaded_file.read()

    # 2) Full text from PDF
    full_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not full_text:
        raise ValueError("Could not extract text from PDF (it might be empty or scanned images).")

    # 3) Summarize with Gemini
    log.info("Attempting to summarize PDF with Gemini...")
    summary = summarize_with_gemini(full_text)

    # 4) Determine Target Text (Fallback Logic)
    target_text = summary
    used_fallback = False
    
    if not summary or len(summary) < 50:
        log.warning("Gemini summarization failed or returned too little text. Falling back to FULL TEXT extraction.")
        target_text = full_text
        used_fallback = True
    else:
        log.info("Gemini summary generated successfully.")

    # 5) Save TXT
    os.makedirs(os.path.join("data", "txt"), exist_ok=True)
    file_stem = os.path.splitext(uploaded_file.name)[0]
    txt_filename = f"{file_stem}_{uuid.uuid4().hex[:8]}.txt"
    txt_path = os.path.join("data", "txt", txt_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== PROCESSING MODE ===\n")
        f.write(f"Used Fallback (Full Text): {used_fallback}\n\n")
        f.write("=== TARGET TEXT (Used for Graph) ===\n")
        f.write(target_text)
        f.write("\n\n=== ORIGINAL FULL TEXT ===\n")
        f.write(full_text)

    # 6) Build KG from the Target Text
    chunks = chunk_tokens(target_text)
    
    log.info(f"Processing {len(chunks)} chunks for graph extraction...")
    
    all_entities: list = []
    all_relations: list = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"pdf_chunk_{uuid.uuid4().hex[:8]}"

        graph_data = extract_graph(chunk)
        entities = graph_data.get("entities", [])
        base_relations = graph_data.get("relations", [])

        refined_relations = []
        if run_relations:
            refined_relations = extract_relations(chunk)

        relations_chunk = base_relations + refined_relations

        all_entities.extend(entities)
        all_relations.extend(relations_chunk)

        # CHANGE: Pass user_id and filename as source
        build_and_store_graph(
            chunk_id, 
            chunk, 
            entities, 
            relations_chunk, 
            user_id=user_id, 
            source=f"pdf:{uploaded_file.name}"
        )

    return {
        "summary": target_text if not used_fallback else "Summary failed. Used full text.",
        "txt_path": txt_path,
        "total_entities": len(all_entities),
        "total_relations": len(all_relations),
    }