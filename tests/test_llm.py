"""Tests for the model layer.

Two things are load-bearing here and neither is about getting an answer.

The first is that "independent reviewer" means something. If the family
classifier collapses two different lineages into one bucket, the agent silently
stops looking for a second opinion; if it splits one lineage into two, it claims
an independence it does not have. Both failures are invisible at runtime, so
they are pinned here.

The second is that every way a provider can fail arrives as one exception type.
The callers turn that into "stand aside" and "veto" - so an unhandled shape
leaking through is a path where a broken model reaches the broker.
"""

from __future__ import annotations

import httpx
import pytest

from onenode.agents import llm


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """The resolved-model cache lives for the process, so tests must not share it."""
    llm._resolved.clear()
    yield
    llm._resolved.clear()


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    """A developer's real keys must not decide what these tests assert."""
    for provider in llm.PROVIDERS:
        monkeypatch.delenv(provider.env_var, raising=False)
    for name in ("ONENODE_PROPOSER_PROVIDER", "ONENODE_REVIEWER_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


class TestFamilies:
    def test_open_weights_gpt_is_not_openai_gpt(self):
        """gpt-oss and gpt-4 share three characters and nothing else.

        Treating them as one family would let a gpt-oss reviewer be rejected as
        "same family" as a GPT-4 proposer, or worse, accepted as independent
        when both are the same checkpoint.
        """
        assert llm.family_of("openai/gpt-oss-120b") == "gpt-oss"
        assert llm.family_of("gpt-4o-mini") == "gpt"

    @pytest.mark.parametrize(
        ("model", "family"),
        [
            ("claude-opus-5", "claude"),
            ("gemini-2.5-flash", "gemini"),
            ("gemma-3-27b-it", "gemma"),
            ("meta-llama/llama-4-scout-17b", "llama"),
            ("qwen-3-32b", "qwen"),
            ("mistralai/Mistral-Small-24B-Instruct-2501", "mistral"),
            ("mistralai/Mixtral-8x7B", "mistral"),
            ("deepseek/deepseek-chat", "deepseek"),
            ("moonshotai/kimi-k2", "kimi"),
        ],
    )
    def test_known_lineages(self, model, family):
        assert llm.family_of(model) == family

    def test_unknown_models_get_their_own_bucket(self):
        """Two unrecognised models must not be mistaken for one family."""
        first = llm.family_of("acme/thinker-9")
        second = llm.family_of("globex/reasoner-2")
        assert first != second
        assert first.startswith("other:")


class TestModelResolution:
    def test_prefers_the_first_preference_that_matches(self):
        models = ["qwen-3-32b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        assert llm._best_match(models, ("gpt-oss-120b", "qwen")) == "openai/gpt-oss-120b"

    def test_prefers_the_short_alias_over_a_dated_snapshot(self):
        """Hosts publish an alias beside its snapshots; the alias is what lasts."""
        models = ["gemini-2.5-flash-preview-09-2026", "gemini-2.5-flash"]
        assert llm._best_match(models, ("gemini-2.5-flash",)) == "gemini-2.5-flash"

    def test_no_match_is_none_rather_than_a_wrong_guess(self):
        assert llm._best_match(["qwen-3-32b"], ("claude", "gemini")) is None

    def test_an_explicit_pin_wins_over_discovery(self, monkeypatch):
        provider = llm.PROVIDERS_BY_NAME["groq"]
        monkeypatch.setenv("ONENODE_GROQ_MODEL", "openai/gpt-oss-20b")
        monkeypatch.setattr(
            llm, "list_models", lambda *a, **k: pytest.fail("discovery ran despite a pin")
        )
        assert llm.resolve_model(provider) == "openai/gpt-oss-20b"

    def test_an_unreachable_catalogue_falls_back_instead_of_raising(self, monkeypatch):
        """A model list that will not load is not a reason to skip a trading window."""

        def _explode(*args, **kwargs):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(llm, "list_models", _explode)
        provider = llm.PROVIDERS_BY_NAME["gemini"]
        assert llm.resolve_model(provider) == provider.fallback_model

    def test_discovery_happens_once_per_process(self, monkeypatch):
        calls = []

        def _count(provider, timeout=20.0):
            calls.append(provider.name)
            return ["gemini-2.5-flash"]

        monkeypatch.setattr(llm, "list_models", _count)
        provider = llm.PROVIDERS_BY_NAME["gemini"]
        llm.resolve_model(provider)
        llm.resolve_model(provider)
        assert calls == ["gemini"]


class TestParsing:
    def test_strips_fences_and_surrounding_prose(self):
        text = 'Sure!\n```json\n{"approve": false, "reason": "too tight"}\n```\nHope that helps.'
        assert llm.parse_json_object(text) == {"approve": False, "reason": "too tight"}

    def test_prose_with_no_object_is_unavailable_not_empty(self):
        """An empty dict would read downstream as a well-formed 'no fields set'."""
        with pytest.raises(llm.LLMUnavailable):
            llm.parse_json_object("I am unable to help with that request.")

    def test_a_json_array_is_rejected(self):
        with pytest.raises(llm.LLMUnavailable):
            llm.parse_json_object("[1, 2, 3]")

    def test_broken_json_is_rejected(self):
        with pytest.raises(llm.LLMUnavailable):
            llm.parse_json_object('{"approve": tru')


class TestAsk:
    def test_no_keys_at_all_names_the_variables_to_set(self):
        with pytest.raises(llm.LLMUnavailable) as caught:
            llm.ask(system="s", user="u", role_env="ONENODE_PROPOSER_PROVIDER")
        assert "GEMINI_API_KEY" in str(caught.value)

    def test_a_pin_to_an_unknown_provider_is_an_error_not_a_silent_default(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ONENODE_PROPOSER_PROVIDER", "nosuchhost")
        with pytest.raises(llm.LLMUnavailable, match="not a known provider"):
            llm.ask(system="s", user="u", role_env="ONENODE_PROPOSER_PROVIDER")

    def test_a_pin_to_a_keyless_provider_says_which_key_is_missing(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("ONENODE_REVIEWER_PROVIDER", "groq")
        with pytest.raises(llm.LLMUnavailable, match="GROQ_API_KEY"):
            llm.ask(system="s", user="u", role_env="ONENODE_REVIEWER_PROVIDER")

    def test_an_excluded_family_is_skipped_when_another_is_available(self, monkeypatch):
        """The reviewer must reach past the proposer's own lineage first."""
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GROQ_API_KEY", "y")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])
        asked: list[str] = []

        def _fake_complete(provider, **kwargs):
            asked.append(provider.name)
            return llm.Reply(payload={"ok": True}, provider=provider.name, model="stub")

        monkeypatch.setattr(llm, "complete_json", _fake_complete)
        llm.ask(
            system="s",
            user="u",
            role_env="ONENODE_REVIEWER_PROVIDER",
            exclude_families=("gemini",),
        )
        assert asked == ["groq"]

    def test_an_excluded_family_is_still_tried_when_it_is_all_there_is(self, monkeypatch):
        """A labelled same-family review beats no review; the caller decides."""
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])
        asked: list[str] = []

        def _fake_complete(provider, **kwargs):
            asked.append(provider.name)
            return llm.Reply(payload={"ok": True}, provider=provider.name, model="gemini-2.5-flash")

        monkeypatch.setattr(llm, "complete_json", _fake_complete)
        reply = llm.ask(
            system="s",
            user="u",
            role_env="ONENODE_REVIEWER_PROVIDER",
            exclude_families=("gemini",),
        )
        assert asked == ["gemini"]
        assert reply.family == "gemini"

    def test_a_dead_provider_falls_through_to_the_next(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GROQ_API_KEY", "y")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])

        def _fake_complete(provider, **kwargs):
            if provider.name == "gemini":
                raise llm.LLMUnavailable("gemini is down")
            return llm.Reply(payload={"ok": True}, provider=provider.name, model="stub")

        monkeypatch.setattr(llm, "complete_json", _fake_complete)
        reply = llm.ask(system="s", user="u", role_env="ONENODE_PROPOSER_PROVIDER")
        assert reply.provider == "groq"

    def test_when_every_provider_fails_the_reasons_are_kept(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        monkeypatch.setenv("GROQ_API_KEY", "y")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])

        def _fake_complete(provider, **kwargs):
            raise llm.LLMUnavailable(f"{provider.name} refused")

        monkeypatch.setattr(llm, "complete_json", _fake_complete)
        with pytest.raises(llm.LLMUnavailable) as caught:
            llm.ask(system="s", user="u", role_env="ONENODE_PROPOSER_PROVIDER")
        assert "gemini refused" in str(caught.value)
        assert "groq refused" in str(caught.value)


class TestTransport:
    def test_an_http_error_becomes_unavailable_with_the_body(self, monkeypatch):
        """Whatever the host says about the refusal ends up in the journal."""
        monkeypatch.setenv("GROQ_API_KEY", "y")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])

        def _post(*args, **kwargs):
            request = httpx.Request("POST", "https://example.invalid/chat/completions")
            response = httpx.Response(429, text="rate limit exceeded", request=request)
            raise httpx.HTTPStatusError("429", request=request, response=response)

        monkeypatch.setattr(httpx, "post", _post)
        with pytest.raises(llm.LLMUnavailable, match="rate limit exceeded"):
            llm.complete_json(llm.PROVIDERS_BY_NAME["groq"], system="s", user="u")

    def test_a_reply_missing_the_expected_fields_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "y")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])

        def _post(*args, **kwargs):
            request = httpx.Request("POST", "https://example.invalid/chat/completions")
            return httpx.Response(200, json={"unexpected": "shape"}, request=request)

        monkeypatch.setattr(httpx, "post", _post)
        with pytest.raises(llm.LLMUnavailable, match="unexpected response shape"):
            llm.complete_json(llm.PROVIDERS_BY_NAME["groq"], system="s", user="u")

    def test_anthropic_is_addressed_with_its_own_header_and_shape(self, monkeypatch):
        """The one provider that is not OpenAI-compatible still has to work."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "z")
        monkeypatch.setattr(llm, "list_models", lambda *a, **k: [])
        seen: dict = {}

        def _post(url, headers=None, json=None, timeout=None):
            seen["url"] = url
            seen["headers"] = headers
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": '{"approve": true, "reason": "fine"}'}]},
                request=request,
            )

        monkeypatch.setattr(httpx, "post", _post)
        reply = llm.complete_json(llm.PROVIDERS_BY_NAME["anthropic"], system="s", user="u")
        assert seen["url"].endswith("/messages")
        assert "x-api-key" in seen["headers"]
        assert reply.payload == {"approve": True, "reason": "fine"}
