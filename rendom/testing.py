from llama_cpp import Llama

model_path = r"D:\models\Mistral-7B-Instruct-v0.1-GGUF\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

print("Loading model…")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=32,   # push first 32 layers to GPU
    n_ctx=1024,
    verbose=True,
)

print("\nModel loaded. Testing inference…")
out = llm("Q: What is 2+2?\nA:")
print(out["choices"][0]["text"])
