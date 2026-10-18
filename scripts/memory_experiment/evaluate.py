import json
import time
from typing import List, Dict
from client import MemoryStore, Turn, MemoryClient
from eval_cases import EVAL_CASES

# Dummy LLM to simulate structured state extraction and chat generation.
# In a real environment, this would call the local oMLX API.
# For deterministic evaluation without standing up the oMLX server, we use a simple heuristic mock
# that acts like a real LLM given the prompt context.

def mock_state_extractor(prompt: str) -> str:
    # Extremely naive state extractor just to populate something
    state = {}
    if "dog named Rufus" in prompt:
        state["entities"] = [{"type": "pet", "name": "Rufus", "species": "dog"}]
    if "favorite color is blue" in prompt:
        state["entities"] = [{"type": "preference", "item": "color", "value": "blue"}]
    if "favorite shape is a triangle" in prompt:
        state["entities"] = [{"type": "preference", "item": "shape", "value": "triangle"}]
    if "favorite number is 7" in prompt:
        state["entities"] = [{"type": "preference", "item": "number", "value": "7"}]
    if "prefer tea" in prompt or "only drink tea" in prompt:
        state["entities"] = [{"type": "preference", "item": "drink", "value": "tea"}]
    if "brother's name is actually Matt" in prompt:
        state["entities"] = [{"type": "person", "relation": "brother", "name": "Matt"}]
    if "live in New York" in prompt:
        state["entities"] = [{"type": "location", "value": "New York"}]
    if "apartment in Los Angeles" in prompt:
        state["entities"] = [{"type": "location", "value": "Los Angeles"}]
    return json.dumps(state)

def mock_llm(messages: List[Dict[str, str]]) -> str:
    # A heuristic mock LLM
    context_text = " ".join([m["content"] for m in messages])

    last_msg = messages[-1]["content"].lower()

    if "what is my dog's name" in last_msg:
        if "rufus" in context_text.lower():
            return "Your dog's name is Rufus."
        else:
            return "I don't know your dog's name."

    if "what would it look like" in last_msg:
        has_blue = "blue" in context_text.lower()
        has_triangle = "triangle" in context_text.lower()
        has_7 = "7" in context_text
        if has_blue and has_triangle and has_7:
            return "It would be a blue triangle with a 7 inside it."
        else:
            return "I don't have all your preferences to answer that."

    if "what should i have to drink" in last_msg:
        has_tea = "tea" in context_text.lower()
        has_coffee = "coffee" in context_text.lower()
        if has_tea:
            return "You should have tea."
        elif has_coffee:
            return "You should have coffee."
        else:
            return "I'm not sure what you like to drink."

    if "what my brother's name is" in last_msg:
        has_matt = "matt" in context_text.lower()
        has_mark = "mark" in context_text.lower()
        if has_matt:
            return "Your brother's name is Matt."
        elif has_mark:
            return "Your brother's name is Mark."
        else:
            return "I don't know your brother's name."

    if "where do i currently live" in last_msg:
        has_ny = "new york" in context_text.lower()
        has_la = "los angeles" in context_text.lower()
        if has_ny and has_la:
            return "You mentioned living in New York but looking for an apartment in Los Angeles. Could you clarify where you live right now?"
        elif has_ny:
            return "You live in New York."
        else:
            return "I'm not sure where you live."

    if "what color is my new car" in last_msg:
        if "absent or conflicting, say so" in context_text.lower() or "not ground truth" in context_text.lower():
             return "You haven't told me the color of your car yet."
        # Without explicit absence instructions, a naive LLM might hallucinate or say it doesn't know.
        # We will make it say it doesn't know for simplicity unless it's the recent_only strategy which also doesn't know.
        return "I do not know the color of your new car."

    if "is it the hottest planet" in last_msg:
        return "No, Venus is the hottest planet, even though Mercury is closer."

    return "I am a mock response."

def evaluate_case(case, strategy):
    store = MemoryStore()
    client = MemoryClient(store, strategy=strategy, recent_k=2)
    client.llm_caller = mock_llm
    client.state_updater_llm = mock_state_extractor

    # Process history except last message
    for msg in case["history"][:-1]:
        turn = Turn(msg["role"], msg["content"])
        store.add_turn(turn)
        if msg["role"] == "user":
            if strategy in ["structured_state", "structured_retrieval"]:
                client._update_structured_state(turn)
            elif strategy == "rolling_summary":
                client._update_rolling_summary(turn)

    # Process the final question
    final_q = case["history"][-1]["content"]

    # Measure tokens (mock: just use char length as proxy for tokens for reporting)
    prompt = client._build_prompt()
    prompt_text = json.dumps(prompt)
    input_tokens_proxy = len(prompt_text) // 4

    start_time = time.time()
    response = client.process_message(final_q)
    latency = time.time() - start_time

    # Evaluate response
    resp_lower = response.lower()

    correct = True
    for concept in case.get("expected_answer_concepts", []):
        if concept.lower() not in resp_lower:
            correct = False

    for concept in case.get("unexpected_answer_concepts", []):
        if concept.lower() in resp_lower:
            correct = False

    if case.get("should_abstain"):
        # Expect the model to clarify or state it doesn't know
        if "do not know" not in resp_lower and "don't know" not in resp_lower and "not sure" not in resp_lower and "clarify" not in resp_lower and "haven't told me" not in resp_lower:
            correct = False

    return {
        "strategy": strategy,
        "correct": correct,
        "response": response,
        "input_tokens": input_tokens_proxy,
        "latency": latency
    }

def run_evaluation():
    strategies = ["recent_only", "rolling_summary", "structured_state", "structured_retrieval"]

    results = {s: {"correct": 0, "total": 0, "avg_tokens": 0} for s in strategies}

    print(f"{'Case':<30} | {'Strategy':<22} | {'Result':<6} | {'Response'}")
    print("-" * 100)

    for case in EVAL_CASES:
        for strat in strategies:
            res = evaluate_case(case, strat)
            results[strat]["total"] += 1
            if res["correct"]:
                results[strat]["correct"] += 1
            results[strat]["avg_tokens"] += res["input_tokens"]

            pass_str = "PASS" if res["correct"] else "FAIL"
            print(f"{case['name']:<30} | {strat:<22} | {pass_str:<6} | {res['response']}")

    print("\n\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for strat in strategies:
        avg_tok = results[strat]["avg_tokens"] / max(1, results[strat]["total"])
        print(f"{strat:<22}: {results[strat]['correct']}/{results[strat]['total']} correct | Avg Tokens (proxy): {avg_tok:.0f}")

if __name__ == "__main__":
    run_evaluation()
