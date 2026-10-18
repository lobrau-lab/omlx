import uuid
import time
import hashlib
from typing import List, Dict, Any, Optional

class Turn:
    def __init__(self, role: str, content: str, conversation_id: str = "default", turn_id: str = None, timestamp: float = None, content_hash: str = None):
        self.turn_id = turn_id or str(uuid.uuid4())
        self.conversation_id = conversation_id
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
        self.content_hash = content_hash or self._compute_hash(content)

    @staticmethod
    def _compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def to_dict(self):
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "content_hash": self.content_hash
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            role=data["role"],
            content=data["content"],
            conversation_id=data.get("conversation_id", "default"),
            turn_id=data.get("turn_id"),
            timestamp=data.get("timestamp"),
            content_hash=data.get("content_hash")
        )

class Assertion:
    def __init__(self, content: str, type: str, source_turn_ids: List[str], assertion_id: str = None,
                 created_at: float = None, scope: str = "conversation", status: str = "active_candidate",
                 is_inferred: bool = False, supersedes: str = None):
        self.assertion_id = assertion_id or str(uuid.uuid4())
        self.content = content
        self.type = type
        self.source_turn_ids = source_turn_ids
        self.created_at = created_at or time.time()
        self.scope = scope
        self.status = status
        self.is_inferred = is_inferred
        self.supersedes = supersedes

    def to_dict(self):
        return {
            "assertion_id": self.assertion_id,
            "content": self.content,
            "type": self.type,
            "source_turn_ids": self.source_turn_ids,
            "created_at": self.created_at,
            "scope": self.scope,
            "status": self.status,
            "is_inferred": self.is_inferred,
            "supersedes": self.supersedes
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            content=data["content"],
            type=data["type"],
            source_turn_ids=data["source_turn_ids"],
            assertion_id=data.get("assertion_id"),
            created_at=data.get("created_at"),
            scope=data.get("scope", "conversation"),
            status=data.get("status", "active_candidate"),
            is_inferred=data.get("is_inferred", False),
            supersedes=data.get("supersedes")
        )

class MemoryState:
    def __init__(self):
        self.current_goal: List[Assertion] = []
        self.current_question: Optional[Assertion] = None
        self.active_topics: List[Assertion] = []
        self.entities: List[Assertion] = []
        self.decisions: List[Assertion] = []
        self.constraints: List[Assertion] = []
        self.preferences: List[Assertion] = []
        self.open_questions: List[Assertion] = []
        self.recent_events: List[Assertion] = []
        self.conflicts: List[Assertion] = []
        self.evidence_needed: List[Assertion] = []
        self.retrieved_evidence: List[Assertion] = []
        self.uncertainties: List[Assertion] = []
        self.updated_at: Optional[float] = None

    def to_dict(self):
        def serialize_list(l): return [a.to_dict() for a in l]
        return {
            "current_goal": serialize_list(self.current_goal),
            "current_question": self.current_question.to_dict() if self.current_question else None,
            "active_topics": serialize_list(self.active_topics),
            "entities": serialize_list(self.entities),
            "decisions": serialize_list(self.decisions),
            "constraints": serialize_list(self.constraints),
            "preferences": serialize_list(self.preferences),
            "open_questions": serialize_list(self.open_questions),
            "recent_events": serialize_list(self.recent_events),
            "conflicts": serialize_list(self.conflicts),
            "evidence_needed": serialize_list(self.evidence_needed),
            "retrieved_evidence": serialize_list(self.retrieved_evidence),
            "uncertainties": serialize_list(self.uncertainties),
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: dict):
        state = cls()
        def deserialize_list(l): return [Assertion.from_dict(a) for a in l] if l else []
        state.current_goal = deserialize_list(data.get("current_goal"))
        if data.get("current_question"):
            state.current_question = Assertion.from_dict(data["current_question"])
        state.active_topics = deserialize_list(data.get("active_topics"))
        state.entities = deserialize_list(data.get("entities"))
        state.decisions = deserialize_list(data.get("decisions"))
        state.constraints = deserialize_list(data.get("constraints"))
        state.preferences = deserialize_list(data.get("preferences"))
        state.open_questions = deserialize_list(data.get("open_questions"))
        state.recent_events = deserialize_list(data.get("recent_events"))
        state.conflicts = deserialize_list(data.get("conflicts"))
        state.evidence_needed = deserialize_list(data.get("evidence_needed"))
        state.retrieved_evidence = deserialize_list(data.get("retrieved_evidence"))
        state.uncertainties = deserialize_list(data.get("uncertainties"))
        state.updated_at = data.get("updated_at")
        return state

class Settings:
    def __init__(self):
        self.model_provider = "mock"
        self.conversation_persistence_scope = "temporary"
        self.memory_enabled = True
        self.cross_conversation_retrieval = False
        self.recent_turn_count = 5
        self.model_context_limit = 8000
        self.reserved_output_tokens = 1000
        self.safety_margin_tokens = 500
        self.retrieval_strategy = "keyword"
        self.context_packing_strategy = "greedy"
        self.diagnostic_visibility = True
