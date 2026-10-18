import json
from scripts.memory_experiment.models import MemoryState, Settings
from typing import Dict, Any

class MemoryInspector:
    """
    Simulates a UI component or developer panel that exposes the memory state,
    provenance, conflicts, and budget breakdowns.
    """

    @staticmethod
    def dump(state: MemoryState, diagnostics: Dict[str, Any], settings: Settings) -> str:
        output = []
        output.append("=== MEMORY INSPECTOR ===")

        # 1. State Variables (Goals, Topics, Preferences, Entities)
        output.append("\n-- Derived State (Index) --")
        state_dict = state.to_dict()
        for key in ["current_goal", "active_topics", "entities", "decisions", "constraints", "preferences", "open_questions"]:
            items = state_dict.get(key, [])
            if items:
                output.append(f"[{key.upper()}]")
                for item in items:
                    source_ids = ", ".join(item.get("source_turn_ids", []))
                    status = item.get("status")
                    supersedes = f" (supersedes {item.get('supersedes')})" if item.get("supersedes") else ""
                    output.append(f"  - {item['content']} [status: {status}]{supersedes} (sources: {source_ids})")

        # 2. Conflicts & Superseded
        if state.conflicts:
            output.append("\n-- Unresolved Conflicts --")
            for conflict in state.conflicts:
                output.append(f"  - {conflict.content} (sources: {', '.join(conflict.source_turn_ids)})")

        # 3. Budget Breakdown
        output.append("\n-- Token Budget Breakdown (Estimated) --")
        output.append(f"  Model Context Limit:    {settings.model_context_limit}")
        output.append(f"  Reserved Output:        {settings.reserved_output_tokens}")
        output.append(f"  Safety Margin:          {settings.safety_margin_tokens}")

        usable = settings.model_context_limit - settings.reserved_output_tokens - settings.safety_margin_tokens
        output.append(f"  Usable Input Tokens:    {usable}")

        packed_messages = diagnostics.get("packed_messages", [])
        # Very rough estimate assuming 4 chars per token
        total_prompt_tokens = sum(len(m["content"]) // 4 for m in packed_messages)
        output.append(f"  Actual Prompt Tokens:   {total_prompt_tokens}")

        # 4. Evidence Used
        retrieved = diagnostics.get("retrieved_evidence", [])
        output.append("\n-- Retrieved Evidence Used --")
        if retrieved:
            for ev in retrieved:
                output.append(f"  - Turn {ev['turn_id']} ({ev['role']}): {ev['content'][:50]}...")
        else:
            output.append("  None")

        output.append("========================")
        return "\n".join(output)
