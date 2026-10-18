import abc
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from scripts.memory_experiment.models import Turn, Assertion, MemoryState, Settings

class ConversationStore(abc.ABC):
    @abc.abstractmethod
    def add_turn(self, turn: Turn): pass
    @abc.abstractmethod
    def get_recent_turns(self, n: int) -> List[Turn]: pass
    @abc.abstractmethod
    def get_all_turns(self) -> List[Turn]: pass
    @abc.abstractmethod
    def get_turn(self, turn_id: str) -> Optional[Turn]: pass

class MemoryStateStore(abc.ABC):
    @abc.abstractmethod
    def get_state(self) -> MemoryState: pass
    @abc.abstractmethod
    def update_state(self, state: MemoryState): pass

class StateExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, turn: Turn, current_state: MemoryState) -> List[Assertion]: pass

class ConflictDetector(abc.ABC):
    @abc.abstractmethod
    def detect_conflicts(self, new_assertions: List[Assertion], current_state: MemoryState) -> List[Tuple[Assertion, Assertion, str]]: pass

class ConflictResolver(abc.ABC):
    @abc.abstractmethod
    def resolve(self, conflicts: List[Tuple[Assertion, Assertion, str]], state: MemoryState): pass

class StateUpdater(abc.ABC):
    @abc.abstractmethod
    def update(self, turn: Turn, state: MemoryState, extractor: StateExtractor, detector: ConflictDetector, resolver: ConflictResolver): pass

class Retriever(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, query: str, store: ConversationStore, settings: Settings) -> List[Turn]: pass

class EvidenceChecker(abc.ABC):
    @abc.abstractmethod
    def check(self, candidate_turns: List[Turn], current_request: str) -> List[Dict[str, Any]]: pass

class TokenCounter(abc.ABC):
    @abc.abstractmethod
    def count(self, text: str) -> int: pass

class ContextPacker(abc.ABC):
    @abc.abstractmethod
    def pack(self, system_instructions: str, user_message: str, state: MemoryState, recent_turns: List[Turn], retrieved_evidence: List[Dict[str, Any]], settings: Settings, token_counter: TokenCounter) -> List[Dict[str, str]]: pass

class ModelProvider(abc.ABC):
    @abc.abstractmethod
    def call(self, messages: List[Dict[str, str]], settings: Settings) -> str: pass

class ChatReturn:
    def __init__(self, response: str, diagnostics: Dict[str, Any] = None):
        self.response = response
        self.diagnostics = diagnostics or {}
