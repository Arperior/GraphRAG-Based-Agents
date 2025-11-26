# pipeline/gpu_manager.py
import gc
import logging
import torch

log = logging.getLogger("gpu_manager")

_CURRENT_MODEL = None
_CURRENT_MODEL_TYPE = None  # "text", "vision", "yolo"

def request_permission_to_load(model_type: str):
    """
    Ensures exclusive GPU access. Unloads conflicting models.
    """
    global _CURRENT_MODEL, _CURRENT_MODEL_TYPE

    if _CURRENT_MODEL is not None:
        if _CURRENT_MODEL_TYPE != model_type:
            log.warning(f"Swapping Models: Unloading {_CURRENT_MODEL_TYPE} to load {model_type}...")

            # 1. Specialized cleanup for GGUF (llama_cpp)
            if hasattr(_CURRENT_MODEL, "close"):
                _CURRENT_MODEL.close()
            
            # 2. Delete the Python object
            del _CURRENT_MODEL
            _CURRENT_MODEL = None
            _CURRENT_MODEL_TYPE = None
            
            # 3. Force Garbage Collection
            gc.collect()
            
            # 4. Clear CUDA Cache (Crucial for YOLO <-> GGUF switching)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            
            log.info("GPU VRAM cleared.")
        else:
            # The requested model is already loaded
            pass

def register_model(model_obj, model_type: str):
    global _CURRENT_MODEL, _CURRENT_MODEL_TYPE
    _CURRENT_MODEL = model_obj
    _CURRENT_MODEL_TYPE = model_type
    log.info(f"Registered {model_type} model as active on GPU.")