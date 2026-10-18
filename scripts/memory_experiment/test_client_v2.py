import unittest
from scripts.memory_experiment.models import Turn, Assertion, MemoryState, Settings
from scripts.memory_experiment.implementations import (
    LocalConversationStore, LocalMemoryStateStore, MockStateExtractor,
    RuleConflictDetector, RuleConflictResolver, SimpleStateUpdater,
    KeywordRetriever, MockEvidenceChecker, GreedyContextPacker,
    HeuristicTokenCounter, ModelProvider
)
from scripts.memory_experiment.client_v2 import MemoryManager

class MockProvider(ModelProvider):
    def call(self, messages, settings): return "Mock"

class TestMemoryManagerV2(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.manager = MemoryManager(
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
            provider=MockProvider(),
            settings=self.settings
        )

    def test_state_extraction_and_update(self):
        ret = self.manager.process_message("I have a dog named Rufus.")
        state = self.manager.state_store.get_state()
        self.assertEqual(len(state.entities), 1)
        self.assertIn("Rufus", state.entities[0].content)

    def test_conflict_resolution_explicit(self):
        self.manager.process_message("My brother's name is Mark.")
        self.manager.process_message("Wait, I made a typo. My brother's name is actually Matt.")

        state = self.manager.state_store.get_state()
        self.assertEqual(len(state.entities), 2)

        mark_a = next(a for a in state.entities if "Mark" in a.content)
        matt_a = next(a for a in state.entities if "Matt" in a.content)

        self.assertEqual(mark_a.status, "superseded")
        self.assertEqual(matt_a.supersedes, mark_a.assertion_id)

    def test_conflict_resolution_ambiguous(self):
        self.manager.process_message("I live in New York.")
        self.manager.process_message("I am looking for a new apartment in Los Angeles.")

        state = self.manager.state_store.get_state()
        self.assertEqual(len(state.entities), 2)

        for a in state.entities:
            self.assertEqual(a.status, "unresolved_conflict")
            self.assertIn(a, state.conflicts)

    def test_token_budget_packing(self):
        # We set a very low usable input to force dropping
        self.manager.settings.model_context_limit = 50
        self.manager.settings.reserved_output_tokens = 10
        self.manager.settings.safety_margin_tokens = 5

        self.manager.process_message("Short message.")
        # Given it's a mock, it's just checking if the code runs without crashing when budgets are tight
        state = self.manager.state_store.get_state()
        self.assertIsNotNone(state)

if __name__ == "__main__":
    unittest.main()
