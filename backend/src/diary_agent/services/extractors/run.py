"""Runs the registered extractors over a day's transcript. No LangGraph here: the graph's
run_extractors node calls run_all(), and so can a script or a future scheduler."""

from collections.abc import Mapping, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from diary_agent.services.extractors.base import Extractor, ExtractorDeps, Report
from diary_agent.services.extractors.registry import EXTRACTORS


def build_runnables(model: BaseChatModel, extractors: Sequence[Extractor] = EXTRACTORS) -> dict[str, Runnable]:
    return {e.name: e.build(model) for e in extractors}


def run_all(
    transcript: str,
    deps: ExtractorDeps,
    runnables: Mapping[str, Runnable],
    extractors: Sequence[Extractor] = EXTRACTORS,
) -> list[Report]:
    """One report per extractor that has a runnable. Never raises: one extractor failing
    (bad JSON after retries, Ollama down, a DB error) must not stop the others or the save."""
    reports = []
    for ext in extractors:
        runnable = runnables.get(ext.name)
        if runnable is None:
            continue
        try:
            reports.append(ext.apply(ext.extract(runnable, transcript), deps))
        except Exception as e:  # noqa: BLE001 - the report is the error channel
            reports.append(Report(ext.name, error=str(e)))
    return reports


def notice_from(reports: list[Report], extractors: Sequence[Extractor] = EXTRACTORS) -> str:
    by_name = {e.name: e for e in extractors}
    parts = []
    for r in reports:
        parts.append(f"(Couldn't check for {r.name}.)" if r.error else by_name[r.name].describe(r))
    return " ".join(p for p in parts if p)
