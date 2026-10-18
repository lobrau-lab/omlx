import unittest
import os
import json
from scripts.memory_experiment.client import MemoryStore, Turn, StructuredState, KeywordRetriever, MemoryClient

class TestMemoryClient(unittest.TestCase):
    def setUp(self):
        self.store_path = "test_store.json"
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        self.store = MemoryStore(persist_path=self.store_path)

    def tearDown(self):
        if os.path.exists(self.store_path):
            os.remove(self.store_path)

    def test_store_turn(self):
        t1 = Turn("user", "Hello")
        self.store.add_turn(t1)

        store2 = MemoryStore(persist_path=self.store_path)
        self.assertEqual(len(store2.turns), 1)
        self.assertEqual(store2.turns[t1.turn_id].content, "Hello")

    def test_update_state(self):
        t1 = Turn("user", "My name is Jules")
        self.store.add_turn(t1)

        state_update = {
            "entities": [{"type": "person", "name": "Jules"}]
        }
        self.store.update_state(state_update, t1.turn_id)

        self.assertEqual(len(self.store.state.entities), 1)
        self.assertEqual(self.store.state.entities[0]["value"], "{'type': 'person', 'name': 'Jules'}")
        self.assertEqual(self.store.state.entities[0]["source_turn_ids"], [t1.turn_id])

    def test_keyword_retriever(self):
        t1 = Turn("user", "I love apples and bananas")
        t2 = Turn("assistant", "That's great!")
        t3 = Turn("user", "Do you remember what fruits I like?")

        self.store.add_turn(t1)
        self.store.add_turn(t2)
        self.store.add_turn(t3)

        retrieved = KeywordRetriever.retrieve("apples", self.store, top_k=1, exclude_recent_n=1)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].turn_id, t1.turn_id)

    def test_prompt_packing_recent_only(self):
        client = MemoryClient(self.store, strategy="recent_only", recent_k=2)

        self.store.add_turn(Turn("user", "old message")) # Should be excluded
        self.store.add_turn(Turn("assistant", "old response"))

        t_recent1 = Turn("user", "new message 1")
        self.store.add_turn(t_recent1)

        t_recent2 = Turn("assistant", "new response 1")
        self.store.add_turn(t_recent2)

        prompt = client._build_prompt()
        self.assertEqual(len(prompt), 3) # System + 2 recent turns
        self.assertEqual(prompt[1]["content"], "new message 1")
        self.assertEqual(prompt[2]["content"], "new response 1")

    def test_absent_evidence_behavior(self):
        # The prompt should contain instructions on absent evidence
        client = MemoryClient(self.store, strategy="structured_retrieval", recent_k=1)
        self.store.add_turn(Turn("user", "Just saying hi"))

        prompt = client._build_prompt()
        sys_prompt = prompt[0]["content"]
        self.assertIn("Make historical claims ONLY when supported by retrieved source turns", sys_prompt)
        self.assertIn("If evidence is absent or conflicting, say so or ask for clarification", sys_prompt)
        self.assertIn("Retrieved Historical Evidence:\nNone", sys_prompt) # None retrieved

if __name__ == "__main__":
    unittest.main()
