from datetime import date

from diary_agent.models.person import Mention
from diary_agent.repositories import entry_repository as entry_repo
from diary_agent.repositories import people_repository as repo

D1, D2 = date(2026, 11, 14), date(2026, 11, 15)


def conn(tmp_path):
    return entry_repo.connect(tmp_path / "journal.db")


def test_normalise():
    assert repo.normalise("  My Boss ") == "boss"
    assert repo.normalise("Lucas's") == "lucas"
    assert repo.normalise("Aunt  May") == "aunt may"


def test_normalise_drops_a_role_before_a_name():
    # The model writes "my husband Dev" one day and "Dev" the next: one person, not two (relationships eval).
    assert repo.normalise("my husband Dev") == "dev"
    assert repo.normalise("my best friend Zara") == "zara"
    assert repo.normalise("my mum") == "mum"  # a role alone stays the key
    assert repo.normalise("my aunt May") == "aunt may"  # "Aunt May" is how she's named


def test_upsert_creates_then_merges(tmp_path):
    c = conn(tmp_path)
    p, created = repo.upsert_person(c, "Lucas", "friend", ["good at math"], D1)
    assert created and p.first_seen == D1 and p.last_seen == D1
    p, created = repo.upsert_person(c, "lucas", "other", ["Good at math", "plays chess"], D2)
    assert not created
    assert p.facts == ["good at math", "plays chess"]
    assert p.relation == "friend"  # "other" never downgrades
    assert (p.first_seen, p.last_seen) == (D1, D2)
    assert p.name == "Lucas"


def test_facts_are_capped(tmp_path):
    c = conn(tmp_path)
    p, _ = repo.upsert_person(c, "Mum", "family", [f"fact {i}" for i in range(12)], D1)
    assert len(p.facts) == repo.MAX_FACTS


def test_replace_mentions_twice_leaves_one_row_per_person(tmp_path):
    c = conn(tmp_path)
    p, _ = repo.upsert_person(c, "Lucas", "friend", [], D1)
    m = Mention(entry_date=D1, person_id=p.id, what_happened="met him", evidence="met Lucas")
    repo.replace_mentions(c, D1, [m])
    repo.replace_mentions(c, D1, [m.model_copy(update={"what_happened": "met him again"})])
    [(person, mention)] = repo.mentions_on(c, D1)
    assert person.id == p.id and mention.what_happened == "met him again"
    assert [x.what_happened for x in repo.mentions_for(c, p.id)] == ["met him again"]


def test_list_get_delete(tmp_path):
    c = conn(tmp_path)
    repo.upsert_person(c, "Lucas", "friend", [], D1)
    repo.upsert_person(c, "Mum", "family", [], D2)
    assert [p.name for p in repo.list_people(c)] == ["Mum", "Lucas"]
    assert repo.get_person(c, "my mum").name == "Mum"
    assert repo.get_person(c, "nobody") is None
    assert repo.delete_person(c, "Lucas") is True
    assert repo.delete_person(c, "Lucas") is False
    assert [p.name for p in repo.list_people(c)] == ["Mum"]


def test_normalise_keeps_groups_and_bare_roles_whole():
    assert repo.normalise("Mum and Dad") == "mum and dad"
    assert repo.normalise("my best friend") == "best friend"


def test_facts_dedupe_ignores_punctuation_and_keep_the_newest(tmp_path):
    c = conn(tmp_path)
    p, _ = repo.upsert_person(c, "Lucas", "friend", ["good at math"], D1)
    p, _ = repo.upsert_person(c, "Lucas", "friend", ["Good at math."], D2)
    assert p.facts == ["good at math"]
    p, _ = repo.upsert_person(c, "Lucas", "friend", [f"fact {i}" for i in range(12)], D2)
    assert len(p.facts) == repo.MAX_FACTS and p.facts[-1] == "fact 11"


def test_remove_fact(tmp_path):
    c = conn(tmp_path)
    repo.upsert_person(c, "Lucas", "friend", ["good at math", "plays chess"], D1)
    assert repo.remove_fact(c, "Lucas", "Plays chess") is True
    assert repo.get_person(c, "Lucas").facts == ["good at math"]
    assert repo.remove_fact(c, "Lucas", "plays chess") is False
    assert repo.remove_fact(c, "nobody", "x") is False


def test_prune_day_removes_people_only_that_day_created(tmp_path):
    c = conn(tmp_path)
    old, _ = repo.upsert_person(c, "Lucas", "friend", [], D1)
    repo.replace_mentions(c, D1, [Mention(entry_date=D1, person_id=old.id, what_happened="met", evidence="")])
    repo.upsert_person(c, "Lucas", "friend", [], D2)
    new, _ = repo.upsert_person(c, "Priya", "friend", [], D2)
    repo.replace_mentions(c, D2, [])  # the re-run found nobody
    repo.prune_day(c, D2)
    assert repo.get_person(c, "Priya") is None
    assert repo.get_person(c, "Lucas").last_seen == D1  # back to the last day they were really mentioned
