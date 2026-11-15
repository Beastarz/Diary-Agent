"""The save-time extractor runner, with fake extractors. No Ollama."""

from datetime import date

from langchain_core.runnables import RunnableLambda

from diary_agent.services.extractors.base import Extractor, ExtractorDeps, Report
from diary_agent.services.extractors.run import build_runnables, notice_from, run_all

DEPS = ExtractorDeps(conn=None, store=None, day=date(2026, 11, 15), model_name="fake")


def echo_extractor(name: str, log: list) -> Extractor:
    def apply(items, deps):
        log.extend(items)
        return Report(name, changed=len(items), details=items)

    return Extractor(
        name=name,
        build=lambda model: RunnableLambda(lambda transcript: [f"{name}:{transcript}"]),
        extract=lambda runnable, transcript: runnable.invoke(transcript),
        apply=apply,
        describe=lambda r: f"{r.name} wrote {r.changed}." if r.changed else "",
    )


def broken_extractor(name: str) -> Extractor:
    def boom(*_):
        raise ConnectionError("ollama is down")

    return Extractor(name=name, build=lambda m: RunnableLambda(boom), extract=lambda r, t: r.invoke(t),
                     apply=lambda items, deps: Report(name), describe=lambda r: "never")


def test_run_all_runs_each_extractor_in_order():
    log = []
    exts = [echo_extractor("a", log), echo_extractor("b", log)]
    reports = run_all("hi", DEPS, build_runnables(None, exts), exts)
    assert log == ["a:hi", "b:hi"]
    assert [r.name for r in reports] == ["a", "b"]
    assert notice_from(reports, exts) == "a wrote 1. b wrote 1."


def test_a_failing_extractor_does_not_stop_the_next():
    log = []
    exts = [broken_extractor("a"), echo_extractor("b", log)]
    reports = run_all("hi", DEPS, build_runnables(None, exts), exts)
    assert reports[0].error == "ollama is down" and reports[0].changed == 0
    assert log == ["b:hi"]
    assert notice_from(reports, exts) == "(Couldn't check for a.) b wrote 1."


def test_an_extractor_without_a_runnable_is_skipped():
    log = []
    exts = [echo_extractor("a", log)]
    assert run_all("hi", DEPS, {}, exts) == []
    assert log == []


def test_notice_is_empty_when_nothing_changed():
    exts = [echo_extractor("a", [])]
    assert notice_from([Report("a")], exts) == ""
