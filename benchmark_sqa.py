import json
import logging
import os
import re
import time
from tqdm import tqdm
from datasets import load_dataset

# Import your existing pipeline modules
import config
from pipeline.neo4j_client import delete_user_data
from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
# Requested: Use relation extractor for better density
from pipeline.relation_extractor import extract_relations_from_text 
from pipeline.graph_builder import build_and_store_graph, build_and_store_image
from pipeline.llm_client_vision import process_image_pipeline
from pipeline.retrieval import gather_evidence_for_query
from pipeline import gpu_manager

# Import Local LLM for Phase 2 (Answer/Judge)
from llama_cpp import Llama

# === LOGGING ===
# Clear existing handlers to prevent duplicates
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("benchmark_scienceqa_run.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("benchmark_sqa")

# === CONFIGURATION ===
BENCHMARK_USER_ID = "benchmark_sqa_agent"
DATASET_NAME = "derek-thomas/ScienceQA"
OUTPUT_FILE = "scienceqa_results.json"
TEST_LIMIT = 10  # Set to None for full run (4,241 samples)

# Point this to your downloaded Qwen/Gemma model
# This overrides the config.py default only for this script
JUDGE_MODEL_PATH = r"D:\models\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# === LOCAL JUDGE & ANSWER ENGINE ===
class LocalReasoningEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None

    def load(self):
        """Loads the heavy reasoning model (Qwen/Gemma)"""
        if self.llm: return
        
        log.info(f"Loading Reasoning Model: {self.model_path}")
        # Request GPU access (unloads LLaVA/Mistral via gpu_manager)
        gpu_manager.request_permission_to_load("reasoning_model")
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_gpu_layers=-1, # All to GPU
            verbose=False,
            logits_all=False
        )
        # Register manually so gpu_manager knows who holds the VRAM
        gpu_manager.register_model(self.llm, "reasoning_model")

    def generate(self, prompt, system_role="You are a helpful assistant."):
        self.load() # Ensure loaded
        
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]
        
        res = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.0 # Deterministic
        )
        return res["choices"][0]["message"]["content"]

# Initialize Engine (Lazy load)
reasoning_engine = LocalReasoningEngine(JUDGE_MODEL_PATH)

# === PHASE 1: INGESTION ===
def ingest_phase(dataset):
    """
    Ingests all samples into Neo4j.
    Uses: YOLO -> LLaVA -> Mistral -> Relation Extractor.
    Does NOT load the heavy Qwen model yet.
    """
    log.info("=== PHASE 1: INGESTION ===")
    
    # 1. Clear Graph ONCE at start
    config.load_config()
    try:
        delete_user_data(BENCHMARK_USER_ID)
    except: pass

    for idx, sample in enumerate(tqdm(dataset, desc="Ingesting Samples")):
        # Unique Source ID for this question
        q_source_id = f"sqa_{idx}"
        
        # A. Text Context (Hint + Lecture)
        text_context = ""
        if sample.get('hint'): text_context += f"Hint: {sample['hint']}\n"
        if sample.get('lecture'): text_context += f"Lecture: {sample['lecture']}\n"
        
        if text_context.strip():
            chunks = chunk_tokens(text_context)
            for i, chunk_text in enumerate(chunks):
                # 1. Basic Extraction (LLM + Spacy)
                graph_data = extract_graph(chunk_text)
                entities = graph_data.get('entities', [])
                relations = graph_data.get('relations', [])
                
                # 2. Enhanced Relation Extraction (Requested Feature)
                try:
                    extra_rels = extract_relations_from_text(chunk_text)
                    if extra_rels:
                        relations.extend(extra_rels)
                except Exception as e:
                    log.warning(f"Extra relation extraction failed for {q_source_id}: {e}")

                # 3. Store
                build_and_store_graph(
                    chunk_id=f"{q_source_id}_txt_{i}",
                    chunk_text=chunk_text,
                    entities=entities,
                    relations=relations,
                    user_id=BENCHMARK_USER_ID,
                    source=q_source_id # Isolate by source
                )

        # B. Image Context
        if sample.get('image'):
            try:
                img = sample['image']
                temp_path = f"temp_{q_source_id}.jpg"
                if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                img.save(temp_path, format="JPEG")
                
                # Vision Pipeline (YOLO/LLaVA)
                # This inherently creates entities and relations from the image
                scene_graph = process_image_pipeline(temp_path, user_context="Science diagram")
                
                build_and_store_image(
                    image_path=temp_path,
                    scene_graph=scene_graph,
                    user_id=BENCHMARK_USER_ID,
                    user_context=f"Image for Q:{idx}"
                )
            except Exception as e:
                log.error(f"Image ingest error {idx}: {e}")
            finally:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

    # Skipping enrichment per user request ("dont use enrichment")
    log.info("Ingestion Complete. Skipping enrichment.")

# === PHASE 2: EXAM (Retrieval & Generation) ===
def exam_phase(dataset):
    """
    Loads Qwen ONCE and processes all Q&A.
    """
    log.info("=== PHASE 2: EXAM & JUDGING ===")
    
    results = []
    correct_count = 0
    
    for idx, sample in enumerate(tqdm(dataset, desc="Answering")):
        q_source_id = f"sqa_{idx}"
        question = sample['question']
        choices = sample['choices']
        
        # 1. Retrieval
        # Retrieve evidence from the graph we just built
        _, evidence = gather_evidence_for_query(
            question, 
            user_id=BENCHMARK_USER_ID, # Scoped to our benchmark user
            k_hop=2,
            top_entities=5
        )
        
        # 2. Generation (Using Local Qwen)
        options_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        
        prompt = f"""
        Use the provided evidence to answer the science question.
        
        Evidence:
        {evidence}
        
        Question: {question}
        Options:
        {options_str}
        
        Explain your reasoning step-by-step, then state the final correct option index.
        """
        
        response = reasoning_engine.generate(prompt)
        
        # 3. Grading (Using Local Qwen)
        # Maps the free-text answer to a specific index (0-4)
        judge_prompt = f"""
        Extract the selected option index from the model's answer.
        
        Question: {question}
        Options:
        {options_str}
        
        Model Answer:
        {response}
        
        Return ONLY a JSON object: {{"selected_index": int}}
        If unclear, return -1.
        """
        
        judge_resp = reasoning_engine.generate(judge_prompt, system_role="You are a strict JSON parser.")
        
        # Parse
        pred_idx = -1
        try:
            clean = judge_resp.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                pred_idx = int(data.get("selected_index", -1))
        except:
            pass
            
        is_correct = (pred_idx == sample['answer'])
        if is_correct: correct_count += 1
        
        results.append({
            "id": idx,
            "question": question,
            "response": response,
            "pred": pred_idx,
            "gt": sample['answer'],
            "correct": is_correct
        })
        
        log.info(f"[{idx}] Correct: {is_correct} (Pred: {pred_idx}, GT: {sample['answer']})")

        # Periodic Save
        if len(results) % 5 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    return results, correct_count

def main():
    # Load Dataset
    log.info(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="test")
    if TEST_LIMIT:
        dataset = dataset.select(range(TEST_LIMIT))
    
    # Run Phase 1: Ingest (Mistral/LLaVA/RelationExtractor active)
    ingest_phase(dataset)
    
    # Run Phase 2: Exam (Qwen active)
    results, correct_total = exam_phase(dataset)
    
    # Final Report
    accuracy = correct_total / len(results) if results else 0
    print("\n" + "="*40)
    print(f"SCIENCEQA BENCHMARK (LOCAL)")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("="*40)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()