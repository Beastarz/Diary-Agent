"""Save-time long-term memory, as an extractor. A thin adapter over services/memory.py.

Differences from the confirmed remember_fact path:
- no yes/no: proposals are applied directly, as additions only (memory.as_additions);
- `person` items are dropped: people live in the relationships extractor's tables now, and two
  homes for the same fact would drift apart.
"""

from langchain_core.runnables import Runnable

from diary_agent.models.memory import MemoryItem
from diary_agent.services import memory as memory_service
from diary_agent.services.extractors.base import Extractor, ExtractorDeps, Report


def extract(runnable: Runnable, transcript: str) -> list[MemoryItem]:
    return [i for i in memory_service.extract_memories(runnable, transcript) if i.kind != "person"]


def apply(items: list[MemoryItem], deps: ExtractorDeps) -> Report:
    if deps.store is None:
        return Report("memory")
    added = memory_service.as_additions(memory_service.propose(deps.store, items, deps.day))
    memory_service.apply_proposals(deps.store, added)
    return Report("memory", changed=len(added), details=[p["memory"]["content"] for p in added])


def describe(report: Report) -> str:
    return memory_service.remembered(report.changed) if report.changed else ""


EXTRACTOR = Extractor(
    name="memory",
    build=memory_service.build_memory_extractor,
    extract=extract,
    apply=apply,
    describe=describe,
)
