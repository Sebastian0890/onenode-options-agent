"""One way to reach a language model, whoever happens to be answering.

The agent needs two opinions and needs them to be independent. That is easy to
claim and easy to fake: two calls to the same model, one of them told to be
sceptical, is theatre. What makes a second opinion real is that it comes from a
different model lineage - different people, different data, different failure
modes - so it does not inherit the first one's blind spots.

So independence here is tracked by *family*, derived from the model that
actually answered, not by vendor. Two hosts serving the same open checkpoint are
one opinion sold twice, and counting them as two would be a lie the account pays
for. When only one family is configured the verdict says ``degraded`` instead of
quietly claiming a review it did not get.

Model identifiers rot, and they rot faster than a hackathon lasts. In the two
weeks before this was written GitHub Models entered its retirement brownout,
Groq shut down the two Llama models most published code had hardcoded, and
Cerebras closed the free tier this project would otherwise have used. A pinned
model name is not protection against that - it is the thing that breaks. Every
provider therefore states its preferences as substrings, and the concrete model
is resolved against whatever ``GET /models`` reports is alive at the time of the
call. A retirement then costs one request, not one trading day.

Everything here speaks the OpenAI chat-completions wire format except Anthropic,
including Google's Gemini, so there is one HTTP path and one special case. No
vendor SDK is involved: the replies wanted are small JSON objects, and one
parser with one failure mode is worth more than four clients with four.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


class LLMUnavailable(RuntimeError):
    """No model could be reached, or none replied usably.

    Callers must treat this as "do not trade". It is never a reason to guess.
    """


@dataclass(frozen=True)
class Provider:
    """A host that will answer, if its key is present in the environment."""

    name: str
    env_var: str
    base_url: str
    prefer: tuple[str, ...]
    """Substrings, best first. Matched against the host's live model list."""
    fallback_model: str
    """Used only when the model list cannot be fetched at all."""

    @property
    def api_key(self) -> str:
        return os.environ.get(self.env_var, "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


# Order is preference order. Free tiers first is deliberate: this runs on a
# budget of zero, and a provider that bills is a provider that can stop
# mid-week when a card declines.
PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="gemini",
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        prefer=("gemini-2.5-flash", "gemini-2.5", "gemini-flash", "gemini"),
        fallback_model="gemini-2.5-flash",
    ),
    Provider(
        name="groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        prefer=("gpt-oss-120b", "gpt-oss", "llama-4", "qwen", "kimi"),
        fallback_model="openai/gpt-oss-120b",
    ),
    Provider(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        prefer=("claude-opus-5", "claude-sonnet-5", "claude"),
        fallback_model="claude-sonnet-5",
    ),
    Provider(
        name="featherless",
        env_var="FEATHERLESS_API_KEY",
        base_url="https://api.featherless.ai/v1",
        prefer=("mistral-small", "mistral", "qwen", "llama"),
        fallback_model="mistralai/Mistral-Small-24B-Instruct-2501",
    ),
    Provider(
        name="openrouter",
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        prefer=("deepseek", "qwen", "llama", "mistral"),
        fallback_model="deepseek/deepseek-chat",
    ),
    Provider(
        name="cerebras",
        env_var="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        prefer=("qwen-3-32b", "qwen", "llama-4", "llama"),
        fallback_model="qwen-3-32b",
    ),
)

PROVIDERS_BY_NAME = {provider.name: provider for provider in PROVIDERS}

# Order matters where one marker contains another: "gpt-oss" is an open-weights
# checkpoint and "gpt-4" is not, and calling them one family would let the agent
# claim an independent review it never had.
_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    ("claude", "claude"),
    ("gemini", "gemini"),
    ("gemma", "gemma"),
    ("gpt-oss", "gpt-oss"),
    ("gpt", "gpt"),
    ("llama", "llama"),
    ("qwen", "qwen"),
    ("mixtral", "mistral"),
    ("magistral", "mistral"),
    ("ministral", "mistral"),
    ("mistral", "mistral"),
    ("deepseek", "deepseek"),
    ("kimi", "kimi"),
    ("glm", "glm"),
    ("nemotron", "nemotron"),
    ("phi", "phi"),
    ("grok", "grok"),
    ("command", "command"),
)


def family_of(model: str) -> str:
    """Which lineage a model belongs to, for the purpose of independence.

    An unrecognised model gets a bucket keyed on its own name, so a checkpoint
    this table has never heard of is treated as its own family rather than
    silently matching every other unknown.
    """
    lowered = model.lower()
    for marker, family in _FAMILY_MARKERS:
        if marker in lowered:
            return family
    return f"other:{lowered.split('/')[-1]}"


@dataclass(frozen=True)
class Reply:
    """A parsed JSON answer, and the receipt for who produced it."""

    payload: dict[str, Any]
    provider: str
    model: str

    @property
    def family(self) -> str:
        return family_of(self.model)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def configured_providers() -> list[Provider]:
    """Providers with a key present, in preference order."""
    return [provider for provider in PROVIDERS if provider.configured]


def _headers(provider: Provider) -> dict[str, str]:
    if provider.name == "anthropic":
        return {
            "x-api-key": provider.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    return {
        "Authorization": f"Bearer {provider.api_key}",
        "content-type": "application/json",
    }


def list_models(provider: Provider, timeout: float = 20.0) -> list[str]:
    """Ask the host what it is actually serving right now."""
    response = httpx.get(f"{provider.base_url}/models", headers=_headers(provider), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data") or payload.get("models") or []
    return [str(entry["id"]) for entry in entries if isinstance(entry, dict) and entry.get("id")]


def _best_match(models: list[str], prefer: tuple[str, ...]) -> str | None:
    """First preference that matches, shortest id first.

    Shortest wins because hosts publish an alias next to its dated snapshots -
    ``gemini-2.5-flash`` beside ``gemini-2.5-flash-preview-09-2026`` - and the
    short one is the name that keeps working.
    """
    for want in prefer:
        matches = sorted((model for model in models if want in model.lower()), key=len)
        if matches:
            return matches[0]
    return None


_resolved: dict[str, str] = {}


def resolve_model(provider: Provider, *, timeout: float = 20.0) -> str:
    """Which model this provider should be asked for, decided once per process.

    An explicit ``ONENODE_<PROVIDER>_MODEL`` always wins: when a model has to be
    pinned for a demo, pinning it should not require a code change.
    """
    override = os.environ.get(f"ONENODE_{provider.name.upper()}_MODEL", "").strip()
    if override:
        return override
    if provider.name in _resolved:
        return _resolved[provider.name]

    try:
        live = list_models(provider, timeout=timeout)
    except Exception:  # noqa: BLE001 - an unreachable catalogue is not fatal here
        live = []

    chosen = _best_match(live, provider.prefer) or provider.fallback_model
    _resolved[provider.name] = chosen
    return chosen


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull one JSON object out of a reply that may be wrapped in prose or fences."""
    cleaned = text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMUnavailable(f"no JSON object in reply: {cleaned[:200]}")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"reply was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMUnavailable("reply JSON was not an object")
    return parsed


def _post_anthropic(
    provider: Provider, model: str, system: str, user: str, max_tokens: int, timeout: float
) -> str:
    response = httpx.post(
        f"{provider.base_url}/messages",
        headers=_headers(provider),
        json={
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    blocks = response.json().get("content", [])
    return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")


def _post_openai_compatible(
    provider: Provider, model: str, system: str, user: str, max_tokens: int, timeout: float
) -> str:
    response = httpx.post(
        f"{provider.base_url}/chat/completions",
        headers=_headers(provider),
        json={
            # Zero temperature everywhere. Two runs fifteen minutes apart should
            # differ because the market moved, not because the sampler rolled
            # differently on the same numbers.
            "temperature": 0.0,
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def complete_json(
    provider: Provider,
    *,
    system: str,
    user: str,
    max_tokens: int = 1024,
    timeout: float = 40.0,
) -> Reply:
    """Ask one provider for one JSON object, and say who answered.

    Every failure - unreachable host, refused key, prose instead of JSON -
    arrives as ``LLMUnavailable``. There is no partial success to salvage.
    """
    if not provider.configured:
        raise LLMUnavailable(f"{provider.name}: {provider.env_var} is not set")

    model = resolve_model(provider, timeout=timeout)
    post = _post_anthropic if provider.name == "anthropic" else _post_openai_compatible
    try:
        text = post(provider, model, system, user, max_tokens, timeout)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200] if exc.response is not None else ""
        raise LLMUnavailable(
            f"{provider.name}/{model}: HTTP {exc.response.status_code} {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"{provider.name}/{model}: {exc}") from exc
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"{provider.name}/{model}: unexpected response shape: {exc}") from exc

    return Reply(payload=parse_json_object(text), provider=provider.name, model=model)


def ask(
    *,
    system: str,
    user: str,
    role_env: str,
    exclude_families: tuple[str, ...] = (),
    max_tokens: int = 1024,
    timeout: float = 40.0,
) -> Reply:
    """Ask the first provider that can answer, preferring an unused family.

    ``role_env`` names an environment variable that pins this role to one
    provider - ``ONENODE_PROPOSER_PROVIDER=groq`` - which is how the two roles
    are held apart deliberately rather than by whatever order the keys are in.

    Family exclusion is a preference, not a veto: a provider whose catalogue is
    unreachable resolves to its fallback name and is still tried last. A review
    from a possibly-related model, labelled as such, beats no review at all -
    the label is what keeps it honest, and the caller decides what to do with it.
    """
    pinned = os.environ.get(role_env, "").strip().lower()
    candidates = configured_providers()
    if pinned:
        provider = PROVIDERS_BY_NAME.get(pinned)
        if provider is None:
            raise LLMUnavailable(f"{role_env}={pinned!r} is not a known provider")
        if not provider.configured:
            raise LLMUnavailable(f"{role_env}={pinned!r} but {provider.env_var} is not set")
        candidates = [provider]

    if not candidates:
        names = ", ".join(provider.env_var for provider in PROVIDERS)
        raise LLMUnavailable(f"no model provider configured - set one of: {names}")

    preferred = [
        provider
        for provider in candidates
        if family_of(resolve_model(provider, timeout=timeout)) not in exclude_families
    ]
    ordered = preferred + [provider for provider in candidates if provider not in preferred]

    failures: list[str] = []
    for provider in ordered:
        try:
            return complete_json(
                provider, system=system, user=user, max_tokens=max_tokens, timeout=timeout
            )
        except LLMUnavailable as exc:
            failures.append(str(exc))

    raise LLMUnavailable("; ".join(failures) or "no provider answered")
