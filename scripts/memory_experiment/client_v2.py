import json
from typing import List, Dict, Any
from scripts.memory_experiment.models import Turn, Settings
from scripts.memory_experiment.interfaces import (
    ConversationStore, MemoryStateStore, StateExtractor, ConflictDetector,
    ConflictResolver, StateUpdater, Retriever, EvidenceChecker, TokenCounter,
    ContextPacker, ModelProvider, ChatReturn
)

class MemoryManager:
    def __init__(self,
                 store: ConversationStore,
                 state_store: MemoryStateStore,
                 extractor: StateExtractor,
                 updater: StateUpdater,
                 detector: ConflictDetector,
                 resolver: ConflictResolver,
                 retriever: Retriever,
                 evidence_checker: EvidenceChecker,
                 token_counter: TokenCounter,
                 packer: ContextPacker,
                 provider: ModelProvider,
                 settings: Settings):
        self.store = store
        self.state_store = state_store
        self.extractor = extractor
        self.updater = updater
        self.detector = detector
        self.resolver = resolver
        self.retriever = retriever
        self.evidence_checker = evidence_checker
        self.token_counter = token_counter
        self.packer = packer
        self.provider = provider
        self.settings = settings

    def process_message(self, user_message: str) -> ChatReturn:
        # 1. Append user message
        user_turn = Turn(role="user", content=user_message)
        self.store.add_turn(user_turn)

        state = self.state_store.get_state()

        # 2. Extract and Update State (includes conflict detection & resolution)
        if self.settings.memory_enabled:
            self.updater.update(user_turn, state, self.extractor, self.detector, self.resolver)
            self.state_store.update_state(state)

        # 3. Retrieve Evidence
        retrieved_evidence = []
        if self.settings.memory_enabled and self.settings.retrieval_strategy != "none":
            # Determine query text (e.g. recent turns)
            recent_turns = self.store.get_recent_turns(self.settings.recent_turn_count)
            query = " ".join([t.content for t in recent_turns])

            raw_turns = self.retriever.retrieve(query, self.store, self.settings)
            retrieved_evidence = self.evidence_checker.check(raw_turns, user_message)

        # 4. Pack prompt
        recent_turns = self.store.get_recent_turns(self.settings.recent_turn_count)
        sys_instructions = "You are a helpful assistant."

        packed_messages = self.packer.pack(
            system_instructions=sys_instructions,
            user_message=user_message,
            state=state,
            recent_turns=recent_turns,
            retrieved_evidence=retrieved_evidence,
            settings=self.settings,
            token_counter=self.token_counter
        )

        # 5. Call Model
        response_content = self.provider.call(packed_messages, self.settings)

        # 6. Append assistant message
        assistant_turn = Turn(role="assistant", content=response_content)
        self.store.add_turn(assistant_turn)

        # 7. Extract and Update State for assistant turn
        if self.settings.memory_enabled:
            self.updater.update(assistant_turn, state, self.extractor, self.detector, self.resolver)
            self.state_store.update_state(state)

        diagnostics = {
            "retrieved_evidence": [e["turn"].to_dict() for e in retrieved_evidence],
            "state": state.to_dict(),
            "packed_messages": packed_messages
        }

        return ChatReturn(response=response_content, diagnostics=diagnostics)
