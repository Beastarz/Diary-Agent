"""The installed save-time extractors, in the order they run. An explicit list on purpose:
no plugin discovery, so what runs is visible in one place."""

from diary_agent.services.extractors import memory
from diary_agent.services.extractors.base import Extractor

EXTRACTORS: tuple[Extractor, ...] = (memory.EXTRACTOR,)
