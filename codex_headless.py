"""Small LangChain-compatible adapter for Codex headless mode.

The workflows only need ``ainvoke(messages)`` and an ``AIMessage`` response.
This wrapper shells out to ``codex exec`` so it uses the locally authenticated
Codex subscription rather than an API endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage


class ChatCodexHeadless:
    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        cwd: str | Path | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model if model and model != "default" else None
        self.effort = effort if effort and effort != "none" else None
        self.cwd = Path(cwd or Path(__file__).parent).resolve()
        self.timeout = timeout

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        prompt = self._format_messages(messages)
        output_path = self._temp_output_path()
        cmd = self._command(output_path)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),
                env=self._env(),
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            if output_path.exists():
                output_path.unlink()
            raise TimeoutError(f"codex exec timed out after {self.timeout:.0f}s") from e

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            if output_path.exists():
                output_path.unlink()
            detail = (stderr or stdout).strip()
            raise RuntimeError(f"codex exec failed with exit code {proc.returncode}: {detail}")

        content = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
        if output_path.exists():
            output_path.unlink()
        if not content:
            content = stdout.strip()
        return AIMessage(content=content)

    def _command(self, output_path: Path) -> list[str]:
        cmd = ["codex"]
        if self.effort:
            cmd.extend(["-c", f"model_reasoning_effort={json.dumps(self.effort)}"])
        cmd.extend(
            [
                "--ask-for-approval",
                "never",
                "exec",
                "--ephemeral",
                "--color",
                "never",
                "--sandbox",
                "read-only",
                "-C",
                str(self.cwd),
                "-o",
                str(output_path),
            ]
        )
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append("-")
        return cmd

    @staticmethod
    def _env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        return env

    @staticmethod
    def _temp_output_path() -> Path:
        fd, name = tempfile.mkstemp(prefix="codex-headless-", suffix=".txt")
        os.close(fd)
        return Path(name)

    @classmethod
    def _format_messages(cls, messages: list[Any]) -> str:
        parts = [
            "You are being called as a text-only judge from an automation workflow.",
            "Follow the latest user instruction exactly. Return only the requested final answer.",
        ]
        for message in messages:
            role = getattr(message, "type", None) or message.__class__.__name__
            content = cls._content_to_text(getattr(message, "content", message))
            parts.append(f"\n[{role}]\n{content}")
        return "\n".join(parts).strip()

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        return str(content)
