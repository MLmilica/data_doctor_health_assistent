"""Regenerate docs/assets/chat_graph.png from the current LangGraph."""

from __future__ import annotations

import sys
from pathlib import Path

from agents.graph import build_graph

DOCS_GRAPH_PNG = Path(__file__).resolve().parent.parent / "docs" / "assets" / "chat_graph.png"


def main() -> int:
    try:
        DOCS_GRAPH_PNG.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = build_graph().get_graph().draw_mermaid_png()
        DOCS_GRAPH_PNG.write_bytes(png_bytes)
    except Exception as exc:
        print(f"Failed to regenerate graph PNG: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {DOCS_GRAPH_PNG.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
