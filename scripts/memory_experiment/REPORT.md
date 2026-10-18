# Memory Layer Experiment Report

## Assessment & Architecture
oMLX is a highly optimized, Apple Silicon-native LLM inference server. It provides an OpenAI-compatible API layer using FastAPI, and maintains a strict stateless design. The server handles generation and caching (KV caching across tiered storage) but does not persist chat history between sessions.

Since oMLX expects the client to pass the full conversation context via the `messages` array in the request, modifying the oMLX core or API routes to introduce stateful memory would break its architectural contract and backward compatibility.

Therefore, the smallest, least-invasive implementation for this experiment is a client-side wrapper (`MemoryClient`). This wrapper intercepts user input, manages a persistent local store (JSON), extracts a structured state, queries historical turns, and constructs a modified prompt to send to oMLX (or in this case, a mock LLM for deterministic evaluation).

## What Changed
- **`scripts/memory_experiment/client.py`**: Created a `MemoryStore` for persisting turns and structured state, a `KeywordRetriever` for naive search, and a `MemoryClient` to orchestrate state updates and prompt construction.
- **`scripts/memory_experiment/test_client.py`**: Added unit tests to verify storing turns, updating state, basic retrieval, prompt packing, and correct inclusion of instructions for absent evidence.
- **`scripts/memory_experiment/eval_cases.py`**: Defined 7 deterministic evaluation cases that test different recall and reasoning scenarios over long-context conversations.
- **`scripts/memory_experiment/evaluate.py`**: Implemented an evaluation script that runs the test cases against four different strategies (`recent_only`, `rolling_summary`, `structured_state`, `structured_retrieval`) using a heuristic mock LLM.

No core application files were modified.

## How to Run It
1. **Run Unit Tests**:
   ```bash
   PYTHONPATH=. python -m unittest scripts.memory_experiment.test_client
   ```
2. **Run the Evaluation**:
   ```bash
   python scripts/memory_experiment/evaluate.py
   ```

## Results
The evaluation measured answer accuracy and relative token usage (proxy) across 4 strategies.

**Strategy Summary:**
- `recent_only`: 5/7 correct | Avg Tokens (proxy): 55
- `rolling_summary`: 5/7 correct | Avg Tokens (proxy): 75
- `structured_state`: 7/7 correct | Avg Tokens (proxy): 202
- `structured_retrieval`: 7/7 correct | Avg Tokens (proxy): 294

**Observations:**
- **Recent Turns & Rolling Summary**: Failed on questions requiring retrieval of details from the distant past or reasoning across multiple old turns.
- **Structured State**: Successfully provided the context needed to answer questions from the distant past while consuming significantly fewer tokens than sending the full conversation history.
- **Structured Retrieval**: Maintained 100% accuracy on the test set. Providing the raw source turns helped the model handle ambiguous contradictions and missing information effectively, though it consumed the most tokens among the memory-augmented strategies.

## Limitations & Next Steps
1. **Naive Extraction & Retrieval**: The current prototype relies on a mock heuristic for state extraction and simple keyword matching for retrieval. A production implementation would require calling the LLM to extract state and using semantic search (e.g., embeddings) to retrieve relevant turns.
2. **Token Proxy**: The evaluation uses character length as a naive proxy for token count. A real tokenizer (e.g., tiktoken) should be used for accurate budget analysis.
3. **Thresholding**: No similarity thresholds were used in keyword retrieval (just `top_k`). Semantic retrieval will require calibrating a similarity threshold against a larger dataset.
4. **Next Experiment**: Integrate the `MemoryClient` with the actual oMLX API endpoints. Evaluate the performance (latency and accuracy) using real oMLX models on a larger benchmark dataset like LongMemEval.
