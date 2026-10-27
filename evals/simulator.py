"""A simulated user: an LLM playing a person with a script, so a whole check-in can run without you.

The persona's "true day" lives only in the simulator's prompt. The agent has to draw it out, and
the saved entry is compared with the script afterwards (personas.jsonl "gold").

Known bias: the simulator is llama3.1:8b too, the same model as the agent. It's cooperative,
articulate and answers every question, which real users don't. Expect real conversations to go
worse than simulated ones, not better. Scripted lines ("inject") pin down the turns a test is
about, so the important moment doesn't depend on the simulator improvising it.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

SIM_MODEL = "llama3.1:8b"  # pinned, like judge.JUDGE_MODEL

SIM_SYSTEM = """You are role-playing a person using a journaling app at the end of their day. The app's
assistant asks you questions; you answer as this person, in the first person.

Who you are: {style}

What actually happened today (the only facts you know):
{day}

Rules:
- Answer the question you were asked, in 1-3 short casual sentences. Don't tell everything at once.
- Use ONLY the facts above. If asked about something not listed, say you don't remember or nothing
  much happened. Never invent new events, people or places.
- You are the user, not the assistant: never ask how you can help, never give advice, never
  describe the user's feelings back to them.
- Output only your reply, with no name or label in front."""


def build_sim():
    from langchain_ollama import ChatOllama

    from diary_agent.core.config import settings

    # Normal chat temperature: people don't answer the same way twice.
    return ChatOllama(model=SIM_MODEL, base_url=settings.ollama_base_url, temperature=0.7, num_predict=120)


def sim_reply(sim, persona: dict, transcript: list[tuple[str, str]]) -> str:
    """Roles flip: the agent's turns are the simulator's input (Human), its own past answers are AI."""
    messages = [SystemMessage(SIM_SYSTEM.format(style=persona["style"], day="\n".join(f"- {f}" for f in persona["day"])))]
    for role, text in transcript:
        messages.append(HumanMessage(text) if role == "assistant" else AIMessage(text))
    return sim.invoke(messages).content.strip().strip('"')
