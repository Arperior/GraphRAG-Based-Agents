import json
import logging
import os
import re
from tqdm import tqdm
from datasets import load_dataset
import time
import random

from config.config import load_config
from pipeline.neo4j_client import delete_user_data
from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
from pipeline.graph_builder import build_and_store_graph, build_and_store_image
from pipeline.relation_extractor import extract_relations_from_text
from pipeline.llm_client_vision import process_image_pipeline
from pipeline.retrieval import gather_evidence_for_query, synthesize_answer
from pipeline.llm_client_gemini import gemini_complete
from pipeline.enrichment_graph import suggest_and_create_links 

# === LOGGING SETUP ===
from pathlib import Path

# Ensure log file is saved in project root — not pipeline directory
log_path = Path(__file__).resolve().parent.parent / "benchmark_run.log"

# Remove any existing handler configs entirely
logging.shutdown()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure ROOT logger now, before imports generate their loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True  # 🔥 IMPORTANT: override any earlier logging configs
)

log = logging.getLogger("benchmark")

# Make sure all other loggers flow into root
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).propagate = True

log.info(f"Logging initialized. Writing to: {log_path}")

_cfg = load_config()
# --- UNIFIED JUDGE PROMPT ---

JUDGE_PROMPT = """
# Answer Evaluation Task
You are an expert evaluator for a Chart-RAG system. Your task is to grade the Model Answer against the Ground Truth Keypoints using two specific metrics: Correctness and Coverage.

## Metric 1: Correctness (Accuracy)
- **Definition:** Assesses if the answer contains any factual errors compared to the keypoints.
- **Scoring:** Binary (0.0 or 1.0).
- **Rule:** - If ALL numerical values and claims in the answer match the keypoints exactly: Score = 1.0
  - If ANY value is incorrect, Hallucinated, or contradictory: Score = 0.0
  - Ignore missing information for this metric (that is covered by Coverage). Only judge what IS present.

## Metric 2: Coverage (Completeness)
- **Definition:** Measures the ratio of required keypoints successfully included in the answer.
- **Scoring:** Continuous (0.0 to 1.0).
- **Rule:** Coverage = (Count of Matched Keypoints) / (Total Number of Ground Truth Keypoints).
  - Matches must be exact values (e.g., "58.19%" vs "58%" -> No Match).

## Input Data
{test_input}

## Output Format
Return a single JSON object containing both scores. Do not include markdown or explanations.
{
    "correctness": 0.0 or 1.0,
    "coverage": 0.0 to 1.0
}
"""
# --- CONFIGURATION ---
BENCHMARK_USER_ID = "benchmark_agent_001"
DATASET_NAME = "ymyang/chart-mrag"
OUTPUT_FILE = "chart_mrag_results.json"
TEST_LIMIT = 50

GEMINI_MAX_RETRIES = 2

def safe_gemini_complete(prompt: str, temperature: float = 0.0) -> str:
    """
    Wraps gemini_complete with exponential backoff on rate limit / quota errors.
    """
    time.sleep(20)
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            return gemini_complete(prompt=prompt, temperature=temperature)
        except Exception as e:
            msg = str(e)
            # crude but effective detection
            if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                wait = 2 ** attempt + random.uniform(0, 1)
                log.warning(f"Gemini rate-limited or quota issue: {msg}. "
                            f"Sleeping {wait:.1f}s before retry {attempt+1}/{GEMINI_MAX_RETRIES}...")
                time.sleep(wait)
                continue
            # non-rate-limit error → rethrow
            raise
    log.error("Gemini failed after maximum retries; returning empty JSON.")
    return "{}"

def setup_system():
    """
    The neo4j driver is initialized automatically when neo4j_client is imported.
    """
    # Clean any stale data from previous runs immediately
    try:
        delete_user_data(BENCHMARK_USER_ID)
    except Exception as e:
        log.warning(f"Initial cleanup failed (might be first run): {e}")

def ingest_sample(sample):
    """
    Ingest content with ENHANCED relation extraction.
    """
    
    # 1. Text Ingestion
    if sample.get('gt_text'):
        chunks = chunk_tokens(sample['gt_text'])
        for i, chunk_text in enumerate(chunks):
            # A. Basic Extraction (LLM + Spacy Fallback)
            graph_data = extract_graph(chunk_text)
            entities = graph_data.get('entities', [])
            relations = graph_data.get('relations', [])
            
            # B. Enhanced Relation Extraction (The "Improvement")
            # We run the dedicated relation extractor to find more connections
            try:
                extra_relations = extract_relations_from_text(chunk_text)
                if extra_relations:
                    relations.extend(extra_relations)
                    log.info(f"Added {len(extra_relations)} extra relations from deep extraction.")
            except Exception as e:
                log.warning(f"Extra relation extraction failed: {e}")

            cid = f"{sample['id']}_text_{i}"
            
            build_and_store_graph(
                chunk_id=cid,
                chunk_text=chunk_text,
                entities=entities,
                relations=relations,
                user_id=BENCHMARK_USER_ID,
                source="benchmark_text"
            )

    # 2. Chart Ingestion
    if sample.get('gt_chart'):
        log.info(f"Skipping chart ingestion for sample {sample['id']} (vision disabled).")

    '''
    if sample.get('gt_chart'):
        temp_path = f"temp_bench_{sample['id']}.jpg"
        try:
            img = sample['gt_chart']
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(temp_path, format="JPEG")
            
            # Vision Pipeline (unchanged, it works well)
            scene_graph = process_image_pipeline(
                temp_path, 
                user_context="Chart data for statistical analysis"
            )
            
            build_and_store_image(
                image_path=temp_path,
                scene_graph=scene_graph,
                user_id=BENCHMARK_USER_ID,
                user_context="Chart data"
            )
            
        except Exception as e:
            log.error(f"Chart ingestion failed for {sample['id']}: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    '''

def extract_retrieved_ids(evidence_list):
    """
    Parses retrieval output strings to extract chunk IDs.
    Format: "[TEXT chunk_id] ..." or "[IMAGE img_id] ..."
    """
    ids = set()
    for ev in evidence_list:
        # Regex to find content inside brackets: [TYPE id]
        match = re.search(r"\[\w+\s+([^\]]+)\]", ev)
        if match:
            ids.add(match.group(1))
    return list(ids)

def infer_ground_truth_sources(sample):
    """Infer correct content sources (text + chart) based on sample id convention."""
    gt_ids = set()

    if sample.get('gt_text'):
        chunks = chunk_tokens(sample['gt_text'])
        for i in range(len(chunks)):
            gt_ids.add(f"{sample['id']}_text_{i}")
    '''
    if sample.get('gt_chart'):
        # Matches the ID generation logic in graph_builder.py
        safe_id = "".join(c if c.isalnum() else "_" for c in f"temp_bench_{sample['id']}")
        gt_ids.add(f"img_{safe_id}")
    '''
    return gt_ids

def compute_recall(gt_ids, retrieved_ids, k):
    if not gt_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(gt_ids.intersection(top_k))
    return hits / len(gt_ids)

def evaluate_llm_judge(answer, gt_keypoints):
    """
    Runs the LLM-as-a-Judge logic.
    OPTIMIZED: Single API call for both Correctness and Coverage.
    """
    if not gt_keypoints:
        return 0.0, 0.0

    test_input = json.dumps({
        "keypoints": gt_keypoints,
        "model_answer": answer
    }, indent=2)

    prompt = JUDGE_PROMPT.replace("{test_input}", test_input)

    # Single Call (rate-limited & retried safely)
    response_text = safe_gemini_complete(prompt=prompt, temperature=0.0)
    try:
        # Robust JSON Parsing
        clean = response_text.replace("```json", "").replace("```", "").strip()
        # Fallback regex if plain json.loads fails
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            clean = match.group(0)
            
        data = json.loads(clean)
        return float(data.get("correctness", 0.0)), float(data.get("coverage", 0.0))
        
    except Exception as e:
        log.warning(f"Judge Parsing Failed: {e}. Resp: {response_text}")
        return 0.0, 0.0

def main():
    setup_system()
    
    log.info(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    if TEST_LIMIT:
        log.warning(f"Running PARTIAL benchmark on first {TEST_LIMIT} samples.")
        eval_data = dataset.select(range(TEST_LIMIT))
    else:
        log.info("Running FULL benchmark.")
        eval_data = dataset

    all_results = []

    for sample in tqdm(eval_data, desc="Benchmarking"):
        sample_id = sample["id"]
        log.info(f"==== Processing Sample {sample_id} ====")

        try:
            # A. Clean State (Functional call)
            delete_user_data(BENCHMARK_USER_ID)
            
            # B. Ingest
            ingest_sample(sample)
            
            gt_ids = infer_ground_truth_sources(sample)

            # C. Ablation 1: Enrichment OFF
            retrieval_results_no = {}
            for hop in [1, 2]:
                _, evidence = gather_evidence_for_query(
                    query=sample['query'],
                    user_id=BENCHMARK_USER_ID,
                    k_hop=hop,
                    top_entities=5
                )
                retrieved_ids = extract_retrieved_ids(evidence)
                retrieval_results_no[f"hop_{hop}"] = {
                    "Recall@5": compute_recall(gt_ids, retrieved_ids, 5),
                    "Recall@10": compute_recall(gt_ids, retrieved_ids, 10)
                }

            # D. Ablation 2: Enrichment ON
            #suggest_and_create_links(
            #    user_id=BENCHMARK_USER_ID, 
            #    confidence_threshold=0.8
            #)

            retrieval_results_yes = {}
            final_evidence = [] 
            
            for hop in [1]:
                _, evidence = gather_evidence_for_query(
                    query=sample['query'],
                    user_id=BENCHMARK_USER_ID,
                    k_hop=hop,
                    top_entities=5
                )
                retrieved_ids = extract_retrieved_ids(evidence)
                retrieval_results_yes[f"hop_{hop}"] = {
                    "Recall@5": compute_recall(gt_ids, retrieved_ids, 5),
                    "Recall@10": compute_recall(gt_ids, retrieved_ids, 10)
                }
                final_evidence = evidence

            # E. Generation & Judging
            response = synthesize_answer(
                query=sample['query'],
                evidence=final_evidence,
                user_id=BENCHMARK_USER_ID
            )
            
            # OPTIMIZED JUDGE CALL
            correctness, coverage = evaluate_llm_judge(response, sample['gt_keypoints'])

            all_results.append({
                "id": sample_id,
                "retrieval_no_enrichment": retrieval_results_no,
                "retrieval_enriched": retrieval_results_yes,
                "final_response": response,
                "gt": sample['gt_keypoints'],
                "scores": {"correctness": correctness, "coverage": coverage}
            })

            log.info(f"[{sample_id}] Correctness={correctness:.2f}, Coverage={coverage:.2f}")
            
            if len(all_results) % 5 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)

        except Exception as e:
            log.error(f"Sample {sample_id} Failed: {e}", exc_info=True)
        time.sleep(10)

    # Final Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    # Statistics
    if all_results:
        avg_corr = sum(r['scores']['correctness'] for r in all_results) / len(all_results)
        avg_cov = sum(r['scores']['coverage'] for r in all_results) / len(all_results)
        
        # Calculate Recall Improvements (Hop 2 comparison)
        recall_gain_5 = sum(r['retrieval_enriched']['hop_1']['Recall@5'] - r['retrieval_no_enrichment']['hop_1']['Recall@5'] for r in all_results) / len(all_results)
        
        log.info(f"==== BENCHMARK COMPLETE ====")
        log.info(f"Average Correctness: {avg_corr:.4f}")
        log.info(f"Average Coverage:    {avg_cov:.4f}")
        log.info(f"Avg Recall@5 Gain:   {recall_gain_5:+.4f}")
        log.info(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()