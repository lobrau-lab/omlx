import json
import time
from typing import List, Dict
from scripts.memory_experiment.client_v2 import MemoryManager
from scripts.memory_experiment.models import Settings
from scripts.memory_experiment.implementations import (
    LocalConversationStore, LocalMemoryStateStore, MockStateExtractor,
    RuleConflictDetector, RuleConflictResolver, SimpleStateUpdater,
    KeywordRetriever, MockEvidenceChecker, GreedyContextPacker,
    HeuristicTokenCounter, ModelProvider
)
from scripts.memory_experiment.eval_cases_v2 import EVAL_CASES
from scripts.memory_experiment.inspector import MemoryInspector

class MockModelProvider(ModelProvider):
    def call(self, messages: List[Dict[str, str]], settings: Settings) -> str:
        context_text = " ".join([m["content"] for m in messages])
        last_msg = messages[-1]["content"].lower()

        if "what is my dog's name" in last_msg:
            if "rufus" in context_text.lower():
                return "Your dog's name is Rufus."
            return "I don't know your dog's name."

        if "what would it look like" in last_msg:
            if "blue" in context_text.lower() and "triangle" in context_text.lower() and "7" in context_text:
                return "It would be a blue triangle with a 7 inside it."
            return "I don't have all your preferences to answer that."

        if "what should i have to drink" in last_msg:
            if "tea" in context_text.lower():
                return "You should have tea."
            if "coffee" in context_text.lower():
                return "You should have coffee."
            return "I'm not sure what you like to drink."

        if "what my brother's name is" in last_msg:
            if "matt" in context_text.lower():
                return "Your brother's name is Matt."
            if "mark" in context_text.lower():
                return "Your brother's name is Mark."
            return "I don't know your brother's name."

        if "where do i live" in last_msg or "where do i currently live" in last_msg:
            if "new york" in context_text.lower() and "los angeles" in context_text.lower():
                return "You mentioned living in New York but looking for an apartment in Los Angeles. Could you clarify where you live right now?"
            if "new york" in context_text.lower():
                return "You live in New York."
            return "I'm not sure where you live."

        if "what color is my new car" in last_msg:
            if "absent or conflicting, say so" in context_text.lower():
                 return "You haven't told me the color of your car yet."
            return "I do not know the color of your new car."

        if "is it the hottest planet" in last_msg:
            return "No, Venus is the hottest planet, even though Mercury is closer."

        return "I am a mock response."

def run_evaluation():
    strategies = ["recent_only", "structured_state", "structured_retrieval"]

    results = {s: {"correct": 0, "total": 0, "avg_tokens": 0} for s in strategies}

    print(f"{'Case':<30} | {'Strategy':<22} | {'Result':<6} | {'Response'}")
    print("-" * 100)

    for case in EVAL_CASES:
        for strat in strategies:
            settings = Settings()
            settings.model_context_limit = 8000

            if strat == "recent_only":
                settings.memory_enabled = False
            elif strat == "structured_state":
                settings.memory_enabled = True
                settings.retrieval_strategy = "none"
            elif strat == "structured_retrieval":
                settings.memory_enabled = True
                settings.retrieval_strategy = "keyword"

            manager = MemoryManager(
                store=LocalConversationStore(),
                state_store=LocalMemoryStateStore(),
                extractor=MockStateExtractor(),
                updater=SimpleStateUpdater(),
                detector=RuleConflictDetector(),
                resolver=RuleConflictResolver(),
                retriever=KeywordRetriever(),
                evidence_checker=MockEvidenceChecker(),
                token_counter=HeuristicTokenCounter(),
                packer=GreedyContextPacker(),
                provider=MockModelProvider(),
                settings=settings
            )

            # Process history except last message
            for msg in case["history"][:-1]:
                # Bypass normal prompt building for history injection to simulate past turns quickly
                from scripts.memory_experiment.models import Turn
                t = Turn(msg["role"], msg["content"])
                manager.store.add_turn(t)
                if msg["role"] == "user" and settings.memory_enabled:
                    manager.updater.update(t, manager.state_store.get_state(), manager.extractor, manager.detector, manager.resolver)

            # Process the final question
            final_q = case["history"][-1]["content"]

            start_time = time.time()
            chat_ret = manager.process_message(final_q)
            latency = time.time() - start_time
            response = chat_ret.response

            input_tokens_proxy = sum(manager.token_counter.count(m["content"]) for m in chat_ret.diagnostics.get("packed_messages", []))

            resp_lower = response.lower()
            correct = True
            for concept in case.get("expected_answer_concepts", []):
                if concept.lower() not in resp_lower:
                    correct = False
            for concept in case.get("unexpected_answer_concepts", []):
                if concept.lower() in resp_lower:
                    correct = False
            if case.get("should_abstain"):
                if not any(x in resp_lower for x in ["do not know", "don't know", "not sure", "clarify", "haven't told me"]):
                    correct = False

            results[strat]["total"] += 1
            if correct: results[strat]["correct"] += 1
            results[strat]["avg_tokens"] += input_tokens_proxy

            pass_str = "PASS" if correct else "FAIL"
            print(f"{case['name'][:28]:<30} | {strat:<22} | {pass_str:<6} | {response[:80]}")

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for strat in strategies:
        avg_tok = results[strat]["avg_tokens"] / max(1, results[strat]["total"])
        print(f"{strat:<22}: {results[strat]['correct']}/{results[strat]['total']} correct | Avg Tokens (proxy): {avg_tok:.0f}")

if __name__ == "__main__":
    run_evaluation()
