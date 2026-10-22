"""Step 0 check: can the configured model call a tool correctly?

    ollama pull qwen2.5:7b
    uv run python scripts/check_llm.py
"""

import time

from langchain_core.tools import tool

from earth_copilot.geo.stac import bbox_area_km2
from earth_copilot.llm import get_chat_model


@tool
def area_of_bbox(west: float, south: float, east: float, north: float) -> float:
    """Area in km² of a lon/lat bounding box."""
    return bbox_area_km2((west, south, east, north))


def main() -> None:
    model = get_chat_model().bind_tools([area_of_bbox])
    question = "What is the area of the box west -74.62, south 46.18, east -74.54, north 46.26?"

    started = time.perf_counter()
    reply = model.invoke(question)
    elapsed = time.perf_counter() - started

    print(f"Answered in {elapsed:.1f}s")
    if not reply.tool_calls:
        raise SystemExit(f"No tool call. Model said: {reply.content}")
    for call in reply.tool_calls:
        print(f"Tool call: {call['name']}({call['args']})")
        print(f"Result: {area_of_bbox.invoke(call['args']):.1f} km²")


if __name__ == "__main__":
    main()
