import json
import time
import uuid
from typing import List, Dict, Any, Optional

class Turn:
    def __init__(self, role: str, content: str, turn_id: str = None, timestamp: float = None):
        self.turn_id = turn_id or str(uuid.uuid4())
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()

    def to_dict(self):
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["role"], data["content"], data.get("turn_id"), data.get("timestamp"))

class StructuredState:
    def __init__(self):
        self.current_goal: List[Dict[str, Any]] = []
        self.active_topics: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.constraints: List[Dict[str, Any]] = []
        self.open_questions: List[Dict[str, Any]] = []
        self.entities: List[Dict[str, Any]] = []
        self.updated_at: Optional[float] = None
        self.source_turn_ids: List[str] = []

    def to_dict(self):
        return {
            "current_goal": self.current_goal,
            "active_topics": self.active_topics,
            "decisions": self.decisions,
            "constraints": self.constraints,
            "open_questions": self.open_questions,
            "entities": self.entities,
            "updated_at": self.updated_at,
            "source_turn_ids": self.source_turn_ids
        }

    @classmethod
    def from_dict(cls, data: dict):
        state = cls()
        state.current_goal = data.get("current_goal", [])
        state.active_topics = data.get("active_topics", [])
        state.decisions = data.get("decisions", [])
        state.constraints = data.get("constraints", [])
        state.open_questions = data.get("open_questions", [])
        state.entities = data.get("entities", [])
        state.updated_at = data.get("updated_at")
        state.source_turn_ids = data.get("source_turn_ids", [])
        return state

class MemoryStore:
    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path
        self.turns: Dict[str, Turn] = {}
        self.turn_order: List[str] = []
        self.state = StructuredState()
        self.rolling_summary = ""
        self.load()

    def add_turn(self, turn: Turn):
        self.turns[turn.turn_id] = turn
        self.turn_order.append(turn.turn_id)
        self.save()

    def update_state(self, new_state_dict: dict, source_turn_id: str):
        fields = ["current_goal", "active_topics", "decisions", "constraints", "open_questions", "entities"]

        for field in fields:
            if field in new_state_dict:
                current_list = getattr(self.state, field)
                new_items = new_state_dict[field]
                if not isinstance(new_items, list):
                    new_items = [new_items]

                for item in new_items:
                    val = item if isinstance(item, str) else str(item)
                    current_list.append({
                        "value": val,
                        "source_turn_ids": [source_turn_id],
                        "timestamp": time.time()
                    })

        self.state.updated_at = time.time()
        if source_turn_id not in self.state.source_turn_ids:
            self.state.source_turn_ids.append(source_turn_id)

        self.save()

    def update_rolling_summary(self, summary: str):
        self.rolling_summary = summary
        self.save()

    def get_recent_turns(self, n: int) -> List[Turn]:
        if n <= 0:
            return []
        return [self.turns[tid] for tid in self.turn_order[-n:]]

    def get_all_turns(self) -> List[Turn]:
        return [self.turns[tid] for tid in self.turn_order]

    def get_turn(self, turn_id: str) -> Optional[Turn]:
        return self.turns.get(turn_id)

    def save(self):
        if not self.persist_path:
            return
        data = {
            "turn_order": self.turn_order,
            "turns": {tid: t.to_dict() for tid, t in self.turns.items()},
            "state": self.state.to_dict(),
            "rolling_summary": self.rolling_summary
        }
        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            self.turn_order = data.get("turn_order", [])
            self.turns = {tid: Turn.from_dict(t) for tid, t in data.get("turns", {}).items()}
            self.state = StructuredState.from_dict(data.get("state", {}))
            self.rolling_summary = data.get("rolling_summary", "")
        except FileNotFoundError:
            pass

class KeywordRetriever:
    @staticmethod
    def retrieve(query: str, store: MemoryStore, top_k: int = 3, exclude_recent_n: int = 0) -> List[Turn]:
        query_words = set(query.lower().split())

        all_turns = store.get_all_turns()
        searchable_turns = all_turns[:-exclude_recent_n] if exclude_recent_n > 0 else all_turns

        scores = []
        for turn in searchable_turns:
            content_words = set(turn.content.lower().split())
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                scores.append((overlap, turn))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scores[:top_k]]

class MemoryClient:
    def __init__(self, store: MemoryStore, strategy: str = "recent_only", recent_k: int = 5):
        self.store = store
        self.strategy = strategy
        self.recent_k = recent_k
        self.llm_caller = None # Mockable callable
        self.state_updater_llm = None # Mockable callable

    def process_message(self, user_message: str) -> str:
        user_turn = Turn("user", user_message)
        self.store.add_turn(user_turn)

        if self.strategy in ["structured_state", "structured_retrieval"]:
            self._update_structured_state(user_turn)
        elif self.strategy == "rolling_summary":
            self._update_rolling_summary(user_turn)

        prompt_messages = self._build_prompt()

        # Make the LLM call
        response_content = self.llm_caller(prompt_messages) if self.llm_caller else f"Mock response to: {user_message}"

        assistant_turn = Turn("assistant", response_content)
        self.store.add_turn(assistant_turn)

        if self.strategy in ["structured_state", "structured_retrieval"]:
            self._update_structured_state(assistant_turn)
        elif self.strategy == "rolling_summary":
            self._update_rolling_summary(assistant_turn)

        return response_content

    def _update_structured_state(self, turn: Turn):
        if not self.state_updater_llm:
            return
        extracted_state_json = self.state_updater_llm(
            f"Extract state from this turn:\n{turn.content}\nCurrent state: {json.dumps(self.store.state.to_dict())}"
        )
        try:
            parsed = json.loads(extracted_state_json)
            self.store.update_state(parsed, turn.turn_id)
        except Exception as e:
            pass

    def _update_rolling_summary(self, turn: Turn):
        if not self.state_updater_llm:
            return
        new_summary = self.state_updater_llm(
            f"Update the following summary with the new turn.\nSummary: {self.store.rolling_summary}\nNew turn: {turn.role}: {turn.content}"
        )
        self.store.update_rolling_summary(new_summary)

    def _build_prompt(self) -> List[Dict[str, str]]:
        messages = []
        system_msg = "You are a helpful assistant."

        if self.strategy == "rolling_summary":
            system_msg += f"\n\nConversation Summary so far:\n{self.store.rolling_summary}"

        elif self.strategy in ["structured_state", "structured_retrieval"]:
            state_dict = self.store.state.to_dict()
            # Filter out empty fields for cleaner prompt
            compact_state = {k: v for k, v in state_dict.items() if v}
            state_str = json.dumps(compact_state, indent=2)
            system_msg += f"\n\nStructured Memory State:\n{state_str}\n\nNote: This is an index, not ground truth. Prioritize retrieved source turns if they conflict."

        if self.strategy == "structured_retrieval":
            # Form query from recent turns
            recent = self.store.get_recent_turns(self.recent_k)
            query_text = " ".join([t.content for t in recent])

            # Retrieve evidence
            retrieved = KeywordRetriever.retrieve(query_text, self.store, top_k=3, exclude_recent_n=self.recent_k)

            evidence_text = "\n\n".join([f"[Turn ID: {t.turn_id}, {t.role}]: {t.content}" for t in retrieved]) if retrieved else "None"
            system_msg += f"\n\nRetrieved Historical Evidence:\n{evidence_text}\n\nMake historical claims ONLY when supported by retrieved source turns. If evidence is absent or conflicting, say so or ask for clarification."

        messages.append({"role": "system", "content": system_msg})

        # Add recent turns (this includes the user turn we just added before calling this)
        recent_turns = self.store.get_recent_turns(self.recent_k)
        for t in recent_turns:
            messages.append({"role": t.role, "content": t.content})

        return messages
