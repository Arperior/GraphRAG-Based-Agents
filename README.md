---

# GraphRAG Based Agent for Multimodal Knowledge Retrieval 

**A College Academic Project** 🎓

This repo houses my work on combining Graph Retrieval-Augmented Generation (GraphRAG) with agentic workflows. The main goal was to see if knowledge graphs could handle complex reasoning better than standard RAG.

## 🛠️ Setup & Installation

I’ve dumped the full, step-by-step setup instructions into **[`setup.txt`](https://www.google.com/search?q=setup.txt)** because it’s pretty long. Check that file to get everything installed correctly.

**Important:**
You **will** need a **Gemini API Key** to run this. The config files assume you have one ready to go.

## 🏃‍♂️ How to Run

**1. Run the App**
To fire up the main interface:

```bash
python app.py

```

**2. Run Benchmarks**
To see how the model performs on the **Chart-MRAG** benchmark:

```bash
python benchmark.py

```

*(There is also a `benchmark_sqa.py` for ScienceQA if you want to test that instead.)*

## 📂 Key Files

* `pipeline/`: The actual logic for the GraphRAG and agents.
* `data/` & `config/`: Where the datasets and settings live.
* `logs/`: Check here if something crashes or to see benchmark results.

---

*Just a heads up: This code was written for a specific academic deadline, so it might be a bit rough around the edges.*
