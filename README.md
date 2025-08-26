# 🛠️ Learning Journey: Building a Tool-Using Edge LLM (12 Weeks)

This repo documents my **12-week self-study & build plan** for creating a **tiny, domain-tuned, tool-calling LLM** that can run locally or on edge devices.

The goal is to:

* Learn the **theory** behind transformers, fine-tuning, RAG, and tool-use.
* Build **hands-on projects** each week.
* Maintain a reproducible repo with code, notes, and evaluations.

---

## 📅 Week-by-Week Plan

### **Phase 1 – Foundations (Weeks 1–3)**

#### Week 1: Transformer & Tokenizer Basics

* **Readings**

  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (decoder focus).
  * [Hugging Face Tokenizers Docs](https://huggingface.co/docs/tokenizers).
  * SentencePiece Paper (Kudo, 2018).
* **Project**

  * Implement a baby decoder-only Transformer (2 layers).
  * Train it on tiny dataset (e.g., Shakespeare).
  * Compare BPE vs SentencePiece tokenization.
* **Definition of Done**

  * [ ] Notebook generates text from scratch.
  * [ ] Visualized attention maps.
  * [ ] Outputs differ with BPE vs SentencePiece.

---

#### Week 2: Positional Encodings & Attention Variants

* **Readings**

  * [RoPE: Rotary Position Embeddings](https://arxiv.org/abs/2104.09864).
  * [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409).
  * [Grouped-Query Attention](https://arxiv.org/abs/2305.13245).
* **Project**

  * Implement RoPE in your mini-transformer.
  * Compare absolute vs RoPE encodings.
* **Definition of Done**

  * [ ] Notebook with side-by-side outputs.
  * [ ] Plots showing how RoPE shifts attention.

---

#### Week 3: LoRA Fine-tuning

* **Readings**

  * [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
  * [PEFT Library Docs](https://huggingface.co/docs/peft).
  * [TRL SFTTrainer Docs](https://huggingface.co/docs/trl).
* **Project**

  * Fine-tune TinyLlama or Qwen-1.5B with LoRA.
  * Use a toy dataset of 200–300 dialogs.
* **Definition of Done**

  * [ ] Adapter saved + reloadable.
  * [ ] Finetuned model changes behavior vs base.

---

### **Phase 2 – Tool Use & Structured Output (Weeks 4–6)**

#### Week 4: Structured Outputs

* **Readings**

  * [llama.cpp Grammar Docs](https://github.com/ggerganov/llama.cpp).
  * [Outlines Library](https://github.com/outlines-dev/outlines).
  * [JSON Schema Official](https://json-schema.org/).
* **Project**

  * Define toy tool `get_weather(city)`.
  * Train examples with JSON-only tool calls.
  * Enforce JSON via grammar-constrained decoding.
* **Definition of Done**

  * [ ] CLI query → always returns valid JSON.
  * [ ] Pydantic validation passes 100%.

---

#### Week 5: Tool Loop Integration

* **Readings**

  * [Model Context Protocol Docs](https://modelcontextprotocol.io/).
  * [Pydantic v2 Docs](https://docs.pydantic.dev/latest/).
* **Project**

  * Build dispatcher: prompt → JSON → call → answer.
  * Add error handling for malformed JSON, tool timeout.
* **Definition of Done**

  * [ ] Full loop works with at least 1 stubbed tool.
  * [ ] Graceful fallback on errors.

---

#### Week 6: More Tools + Domain Data

* **Readings**

  * OpenAI Function Calling Blogpost.
  * [DPO Paper (2023)](https://arxiv.org/abs/2305.18290).
* **Project**

  * Add 2–3 realistic domain tools.
  * Author 200 SFT dialogs mixing tool + natural answers.
  * Fine-tune adapters with domain data.
* **Definition of Done**

  * [ ] Model decides correctly when to call tools.
  * [ ] Handles at least 3 tools consistently.

---

### **Phase 3 – RAG & Inference Engineering (Weeks 7–9)**

#### Week 7: Retrieval-Augmented Generation (RAG)

* **Readings**

  * [FAISS Docs](https://faiss.ai/).
  * [LlamaIndex Best Practices](https://docs.llamaindex.ai).
  * [LangChain RAG Cookbook](https://python.langchain.com/docs).
* **Project**

  * Build FAISS index over 20+ docs.
  * Implement retrieval + prompt wiring.
* **Definition of Done**

  * [ ] CLI answers doc questions using retrieval.
  * [ ] Retrieved snippets are cited in output.

---

#### Week 8: Sampling & Inference Controls

* **Readings**

  * [llama-cpp-python Sampling Parameters](https://github.com/abetlen/llama-cpp-python).
  * [Speculative Decoding Paper](https://arxiv.org/abs/2302.01318).
  * [vLLM PagedAttention](https://arxiv.org/abs/2309.06180).
* **Project**

  * Create playground to test temp, top\_p, top\_k.
  * Compare deterministic vs free sampling.
* **Definition of Done**

  * [ ] Notebook with multiple completions compared.
  * [ ] JSON tool calls always use `temp=0`.

---

#### Week 9: Quantization & Edge

* **Readings**

  * [llama.cpp GGUF Guide](https://github.com/ggerganov/llama.cpp).
  * [GPTQ Paper](https://arxiv.org/abs/2210.17323).
  * [AWQ Paper](https://arxiv.org/abs/2306.00978).
  * [MLC LLM Docs](https://llm.mlc.ai/).
* **Project**

  * Convert LoRA → GGUF (`q4_K_M`).
  * Run with llama.cpp, measure tokens/sec + memory.
* **Definition of Done**

  * [ ] Model runs offline with <4GB RAM.
  * [ ] Performance logged (latency, memory).

---

### **Phase 4 – Eval, Safety, Packaging (Weeks 10–12)**

#### Week 10: Evaluation Harness

* **Readings**

  * [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
  * [HF Evaluate](https://huggingface.co/docs/evaluate).
* **Project**

  * Build eval set (50 gold JSON tool calls).
  * Compute JSON validity, tool accuracy, latency.
* **Definition of Done**

  * [ ] Eval script runs in CI.
  * [ ] Thresholds set (e.g., JSON ≥ 95%).

---

#### Week 11: Safety & Guardrails

* **Readings**

  * [OWASP LLM Top 10 (2025)](https://owasp.org).
  * [NCSC Prompt Injection Guidance](https://www.ncsc.gov.uk).
  * [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework).
* **Project**

  * Add schema validation, least-privilege tool execution.
  * Add logging (prompt, JSON, tool result).
* **Definition of Done**

  * [ ] Dispatcher never executes raw user text.
  * [ ] Errors logged clearly, model recovers gracefully.

---

#### Week 12: Packaging & Beta Release

* **Readings**

  * [HF Licensing Guide](https://huggingface.co/docs/hub/licenses).
  * [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
  * [Ollama Modelfile Docs](https://github.com/ollama/ollama).
* **Project**

  * Bundle quantized model + adapters + RAG index.
  * Package as CLI or Docker API.
  * Write README + install guide.
* **Definition of Done**

  * [ ] One-line install works (`pip install` or `docker run`).
  * [ ] Demo works end-to-end offline.

---

## 🚀 Final Deliverable

By the end of Week 12:

* A **quantized 1–3B LLM** running offline.
* **LoRA adapters** tuned on domain + tools.
* **Tool-calling loop** with safe JSON outputs.
* **Local RAG** for your docs.
* **Eval harness in CI** with thresholds.
* A **packaged release** for others to run easily.

---

