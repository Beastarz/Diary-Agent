from datetime import date

from diary_agent.models.summary import SummaryRecord
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import summary_repository as summary_repo


def make_summary(period="week", start=date(2026, 10, 19), overview="You had a busy week.") -> SummaryRecord:
    return SummaryRecord(
        period=period, start_date=start, end_date=date(2026, 10, 25), entry_count=5, mood_avg=6.4,
        themes=["work"], wins=["10k personal best"], challenges=["deadline"], mood_trend="Dipped midweek.",
        overview=overview, model="fake", prompt_version="test",
    )


def test_roundtrip_and_overwrite(tmp_path):
    conn = entry_repo.connect(tmp_path / "j.db")
    assert summary_repo.get_summary(conn, "week", date(2026, 10, 19)) is None

    summary_repo.save_summary(conn, make_summary())
    summary_repo.save_summary(conn, make_summary(overview="Rewritten."))
    loaded = summary_repo.get_summary(conn, "week", date(2026, 10, 19))
    assert loaded == make_summary(overview="Rewritten.").model_copy(update={"created_at": loaded.created_at})


def test_week_and_month_are_separate(tmp_path):
    conn = entry_repo.connect(tmp_path / "j.db")
    summary_repo.save_summary(conn, make_summary("week", date(2026, 10, 1)))
    assert summary_repo.get_summary(conn, "month", date(2026, 10, 1)) is None
