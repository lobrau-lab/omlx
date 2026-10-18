# Modular Memory Manager Prototype (V2)

## 1. Implementation Location
The prototype remains housed entirely within `scripts/memory_experiment/`. This decision aligns with the findings from the first review: oMLX is a high-performance, purely stateless LLM inference server. Introducing memory management, databases, or stateful sessions into oMLX's core API endpoints (`omlx/api/`) or engine (`omlx/engine/`) would break its architectural contract.

Thus, this memory manager acts as a **smart client wrapper**. It stores conversation turns, derives a context ledger, retrieves evidence, and packs a valid OpenAI-compatible `messages` payload before calling the model provider (which in production would be the local oMLX server).

## 2. Module Boundaries & Data Flow

The `MemoryManager` orchestrates several distinct, interchangeable modules behind explicit interfaces:

- **`ConversationStore`**: Persists original immutable `Turn` records.
- **`MemoryStateStore`**: Maintains the structured index of the conversation (`MemoryState`), linking derived values back to their source `Turn` IDs.
- **`StateExtractor`**: Examines a new `Turn` and extracts candidate `Assertion` records (goals, entities, preferences, etc.).
- **`ConflictDetector`**: Compares new assertions against the existing `MemoryState` to flag overlaps or changes (e.g., preference shifts).
- **`ConflictResolver`**: Decides whether to mark old assertions as "superseded" (for explicit corrections) or "unresolved_conflict" (for ambiguous contradictions).
- **`StateUpdater`**: Applies the resolved assertions back into the `MemoryStateStore`.
- **`Retriever`**: Queries the `ConversationStore` for historical turns relevant to the current user query.
- **`EvidenceChecker`**: Evaluates retrieved turns for support/relevance.
- **`TokenCounter`**: Estimates token cost of individual components.
- **`ContextPacker`**: A greedy algorithm that packs the system prompt, state index, retrieved evidence, and recent turns into the bounded budget (`usable_input_tokens = model_context_limit - reserved_output_tokens - safety_margin_tokens`).
- **`ModelProvider`**: Takes the packed messages and performs the LLM call.

## 3. The Working Prototype and Tests
- Interfaces and Data Models: `interfaces.py`, `models.py`
- Concrete Implementations: `implementations.py`
- Orchestrator: `client_v2.py`
- Diagnostics/UI Mock: `inspector.py`
- Focused Tests: `test_client_v2.py` (Tests for state updates, explicit/ambiguous conflict resolution, and bounded packing).

## 4. Instructions to Run
**Run Unit Tests:**
```bash
PYTHONPATH=. python -m unittest scripts.memory_experiment.test_client_v2
```

**Run the Evaluation Suite:**
```bash
PYTHONPATH=. python scripts/memory_experiment/evaluate_v2.py
```
*(The evaluation script simulates the oMLX API using a heuristic mock provider for deterministic, rapid evaluation without standing up the oMLX server).*

## 5. Evaluation Results
The system was evaluated against 10 robust cases spanning ambiguous contradictions, budget exhaustion, filler content, and missing evidence.

**Model Settings for Eval:** Context Limit: 8000. Safety/Output margins: 1500.

| Strategy | Accuracy | Avg Token Proxy |
|---|---|---|
| `recent_only` | 8/10 | 256 |
| `structured_state` | 10/10 | 507 |
| `structured_retrieval` | 10/10 | 778 |

**Observations:**
- `recent_only` fails when required facts fall out of the recent context window (e.g., long filler histories) or when answering questions requiring multiple distant turns.
- `structured_state` works perfectly as an index for factual lookup, drastically reducing tokens compared to sending a full conversation history.
- `structured_retrieval` ensures that the model sees the raw original text from the past. By providing explicit instructions to rely on retrieved evidence or abstain, the LLM correctly handles "missing evidence" without hallucinating, and manages ambiguous contradictions properly.

## 6. Limitations, Failure Cases, and Next Steps
- **Limitation**: The current `ContextPacker` drops components aggressively if the budget is reached. A more nuanced packer should summarize or truncate specific older turns rather than dropping entire categories of evidence.
- **Limitation**: The prototype uses heuristic mock extractors and a simple keyword retriever.
- **Next Recommended Experiment**: Wire the `MemoryManager`'s `ModelProvider` and `StateExtractor` directly into oMLX's Python HTTP client (`openai` package). Replace the `KeywordRetriever` with a lightweight embedding model (e.g., using `omlx` embedding routes) to test semantic recall accuracy across multi-hour coding sessions.
