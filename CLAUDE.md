<coding_principles>
  <general>
    Follow PEP 8. Use type hints everywhere. Write docstrings on every module,
    class, and public function. Keep functions under 40 lines — split if longer.
    No hardcoded credentials or magic numbers; use constants or config.
    Prefer explicit over implicit. Delete dead code rather than commenting it out.
  </general>

  <architecture>
    <!-- Agent design -->
    Every agent is a plain function state -> state. No side effects outside state.
    All inter-agent data must be a Pydantic v2 model — never pass raw dicts or strings.
    Use LangGraph conditional edges for gating; never gate with if-blocks inside agents.
    VADER scores individual posts on CPU; LLM is called only for narrative clustering,
    disambiguation, risk explanation, and vendor reasoning — never for raw scoring.
    Use ratio-based spike detection (negatives/total >= NEG_RATIO), not mean-based.
    Keep search_news() signature stable so a live API swaps in by replacing one function.

    <!-- OOP -->
    Prefer composition over inheritance. Only introduce a class when it owns both
    state and behaviour — use plain functions or dataclasses otherwise.
    Avoid deep inheritance chains; two levels maximum.

    <!-- SOLID -->
    Single Responsibility: each module does exactly one thing.
      schemas.py defines contracts, llm.py handles inference, synthetic_data.py
      owns data generation, agents.py holds agent logic — never mix these.
    Open/Closed: add new entities to the watchlist or new scenarios to NEWS_CORPUS
      without touching agent logic. Add new agent nodes without modifying existing ones.
    Liskov Substitution: any callable matching state->state can slot into LangGraph
      without other changes. search_news() can be replaced by any function returning
      the same list[dict] shape.
    Interface Segregation: llm.py exposes only chat() and chat_json() — callers
      must not depend on retry or parsing internals.
    Dependency Inversion: agents depend on llm.chat_json and synthetic_data.search_news
      abstractions, never on Ollama or a specific news API directly.

    <!-- Clean code -->
    One level of abstraction per function. If a function both coordinates and
    computes, split it. Name functions after what they return, not how they work.
    No function longer than 40 lines. No file longer than 200 lines — split by
    responsibility if exceeded. No commented-out code in commits.

    <!-- Modularity -->
    Modules must be independently importable and testable. smoke.py must run
    with zero changes to agents.py or orchestrator.py.
    Config values (NEG_POST, NEG_RATIO, MIN_VOLUME, model name, Ollama host)
    live in a constants block at the top of the module that owns them — never
    scattered inline across functions.
  </architecture>

  <llm>
    Always call chat_json, never chat, when you need structured output.
    System prompts must end with an explicit instruction to return JSON only.
    Every chat_json call must specify retries=2 minimum.
    Never parse LLM output with regex — use Pydantic validation only.
    Pin entity_id and entity_name to ground truth after every adverse_agent call.
    Disambiguation must reject articles that match aliases but concern a different entity.
  </llm>

  <error_handling>
    Catch and log; never silently swallow exceptions.
    On LLM validation failure re-prompt with the error, do not raise immediately.
    All agent failures must update state with an error_log entry and continue.
    vendor_agent must skip entities whose adverse finding is category == "low".
  </error_handling>

  <testing>
    smoke.py must cover data gen, VADER spike detection, vendor graph, and news
    retrieval with zero LLM calls. The brand scenario spike must always assert True.
    Run smoke.py before committing any change to agents.py or synthetic_data.py.
  </testing>
  
</coding_principles>