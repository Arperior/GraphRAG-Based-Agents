# pipeline/image_utils.py
import os
import uuid
import logging
from PIL import Image

from pipeline.preprocessing import chunk_tokens
from pipeline.entity_extraction import extract_graph
from pipeline.relation_extractor import extract_relations_from_text
from pipeline.graph_builder import build_and_store_graph

# NEW: Import the unified function from local client
from pipeline.llm_client_local import describe_image

log = logging.getLogger("image_utils")

def process_image_upload(uploaded_file, run_relations: bool = True, user_id: str | None = None):
    """
    1. Load Image
    2. Describe using Local LLaVA (via llm_client_local)
    3. Build Graph from description
    """
    # 1. Load Image
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")

    # 2. Summarize Locally
    log.info(f"Generating local summary for image: {uploaded_file.name}")
    
    # CALL THE CENTRALIZED CLIENT
    summary = describe_image(image)
    
    if not summary:
        raise RuntimeError(
            "Failed to generate image summary locally. "
            "Check if your LLaVA model paths in llm_client_local.py are correct."
        )

    # 3. Save Summary
    os.makedirs(os.path.join("data", "txt"), exist_ok=True)
    file_stem = os.path.splitext(uploaded_file.name)[0]
    txt_filename = f"IMG_{file_stem}_{uuid.uuid4().hex[:8]}.txt"
    txt_path = os.path.join("data", "txt", txt_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== IMAGE SOURCE (LOCAL): {uploaded_file.name} ===\n")
        f.write(summary)

    # 4. Build Graph
    chunks = chunk_tokens(summary)
    total_entities = 0
    total_relations = 0

    for chunk in chunks:
        chunk_id = f"img_chunk_{uuid.uuid4().hex[:8]}"
        
        # Extract Graph (This uses _get_text_model in llm_client_local internally)
        graph_data = extract_graph(chunk)
        entities = graph_data.get("entities", [])
        base_relations = graph_data.get("relations", [])
        
        refined_relations = []
        if run_relations:
            refined_relations = extract_relations_from_text(chunk)
            
        relations_chunk = base_relations + refined_relations
        
        build_and_store_graph(
            chunk_id, chunk, entities, relations_chunk, 
            user_id=user_id, source=f"image:{uploaded_file.name}"
        )
        total_entities += len(entities)
        total_relations += len(relations_chunk)

    return {
        "summary": summary,
        "txt_path": txt_path,
        "total_entities": total_entities,
        "total_relations": total_relations,
    }