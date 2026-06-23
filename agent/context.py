"""Context-management for the agent harness.

The agent accumulates a running transcript of turns (user request, reasoning, tool calls
and tool results). Tool results can be bulky (image metadata, eval blobs). This module
keeps the context within a token-ish budget by:

  1. Always preserving the system prompt and the most recent ``keep_recent`` turns verbatim.
  2. Shrinking older bulky tool results to a compact summary ({path, scores}).
  3. Folding everything still over budget into a single summary note.

The budget is measured with a cheap word-count proxy (no tokenizer dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _approx_tokens(text: str) -> int:
    # ~1.3 tokens/word is a serviceable proxy without pulling in a tokenizer.
    return int(len(text.split()) * 1.3) + 1


@dataclass
class ContextManager:
    system_prompt: str
    max_tokens: int = 3000
    keep_recent: int = 6
    turns: list[dict] = field(default_factory=list)
    _summary: str = ""

    def add(self, role: str, content: str, *, kind: str = "text") -> None:
        """Append a turn. ``kind`` marks bulky tool output eligible for shrinking."""
        self.turns.append({"role": role, "content": content, "kind": kind})
        self.compact()

    def _size(self) -> int:
        total = _approx_tokens(self.system_prompt) + _approx_tokens(self._summary)
        return total + sum(_approx_tokens(t["content"]) for t in self.turns)

    def compact(self) -> None:
        """Shrink/summarize older turns until within budget (best-effort)."""
        if self._size() <= self.max_tokens:
            return

        # Step 1: collapse bulky tool results among the older turns.
        old = self.turns[: -self.keep_recent] if len(self.turns) > self.keep_recent else []
        for t in old:
            if t["kind"] == "tool_result" and len(t["content"]) > 200:
                t["content"] = t["content"][:160] + " …[tool result truncated]"

        if self._size() <= self.max_tokens:
            return

        # Step 2: fold all-but-recent turns into a single summary note.
        recent = self.turns[-self.keep_recent :]
        folded = self.turns[: -self.keep_recent]
        if folded:
            lines = [f"{t['role']}: {t['content'][:120]}" for t in folded]
            addition = "\n".join(lines)
            self._summary = (self._summary + "\n" + addition).strip()
            # Keep the summary itself bounded.
            if _approx_tokens(self._summary) > self.max_tokens // 2:
                kept = self._summary.splitlines()[-20:]
                self._summary = "…\n" + "\n".join(kept)
            self.turns = recent

    def render(self) -> list[dict]:
        """Produce the message list to send to the LLM."""
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self._summary:
            msgs.append({"role": "system", "content": "Earlier context summary:\n" + self._summary})
        for t in self.turns:
            role = "assistant" if t["role"] == "assistant" else "user"
            msgs.append({"role": role, "content": t["content"]})
        return msgs

    def stats(self) -> dict:
        return {
            "turns": len(self.turns),
            "approx_tokens": self._size(),
            "max_tokens": self.max_tokens,
            "summarized": bool(self._summary),
        }
