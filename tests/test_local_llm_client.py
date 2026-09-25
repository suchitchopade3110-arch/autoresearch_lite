from unittest.mock import MagicMock

from generation.patch_generator import LocalLLMClient, LLMClient


def _fake_response(text, prompt_tokens=100, completion_tokens=50, usage=True):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    body = {"choices": [{"message": {"content": text}}]}
    if usage:
        body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    resp.json.return_value = body
    return resp


GOOD_DIFF = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print(1)\n"
BAD_DIFF = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1,5 @@\n-\n+print(1)\n"


def _make_client(responses, **kwargs):
    client = LocalLLMClient(base_url="http://localhost:11434/v1", model="qwen2.5-coder:32b", **kwargs)
    client.client = MagicMock()
    client.client.post.side_effect = responses
    return client


def test_generate_diff_returns_a_valid_diff_on_first_try():
    client = _make_client([_fake_response(GOOD_DIFF)])
    diff = client.generate_diff("goal", "candidate_script.py", "\n")
    assert diff.strip() == GOOD_DIFF.strip()
    assert client.client.post.call_count == 1


def test_unparseable_diff_triggers_a_retry_with_the_previous_stderr_fed_back():
    """Same retry contract as AnthropicClient - see test_anthropic_client.py."""
    client = _make_client([_fake_response(BAD_DIFF), _fake_response(GOOD_DIFF)])
    diff = client.generate_diff("goal", "candidate_script.py", "\n")

    assert diff.strip() == GOOD_DIFF.strip()
    assert client.client.post.call_count == 2

    second_call_kwargs = client.client.post.call_args_list[1].kwargs
    second_prompt = second_call_kwargs["json"]["messages"][1]["content"]
    assert "PREVIOUS ATTEMPT FAILED TO APPLY" in second_prompt
    assert "patch" in second_prompt.lower()


def test_exhausting_retries_returns_the_last_attempt_without_raising():
    client = _make_client([_fake_response(BAD_DIFF)] * 3)
    diff = client.generate_diff("goal", "candidate_script.py", "\n")
    assert client.client.post.call_count == 3
    assert diff.strip() == BAD_DIFF.strip()


def test_last_usage_records_tokens_and_a_zero_cost_estimate():
    """
    Unlike AnthropicClient, cost is always 0.0 - local inference has no
    per-token billing, only token counts (when the server reports them).
    """
    client = _make_client([_fake_response(GOOD_DIFF, prompt_tokens=200, completion_tokens=80)])
    client.generate_diff("goal", "candidate_script.py", "\n")
    assert client.last_usage["input_tokens"] == 200
    assert client.last_usage["output_tokens"] == 80
    assert client.last_usage["estimated_cost_usd"] == 0.0


def test_last_usage_defaults_to_zero_tokens_when_the_server_omits_usage():
    """Not every OpenAI-compatible local server reports a usage block."""
    client = _make_client([_fake_response(GOOD_DIFF, usage=False)])
    client.generate_diff("goal", "candidate_script.py", "\n")
    assert client.last_usage["input_tokens"] == 0
    assert client.last_usage["output_tokens"] == 0
    assert client.last_usage["estimated_cost_usd"] == 0.0


def test_markdown_fences_are_stripped_from_the_response():
    fenced = f"```diff\n{GOOD_DIFF}```"
    client = _make_client([_fake_response(fenced)])
    diff = client.generate_diff("goal", "candidate_script.py", "\n")
    assert not diff.strip().startswith("```")
    assert diff.strip() == GOOD_DIFF.strip()


def test_two_different_prompts_produce_two_different_diffs():
    diff_a = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('a')\n"
    diff_b = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('b')\n"
    client = _make_client([_fake_response(diff_a), _fake_response(diff_b)])

    result_a = client.generate_diff("goal A", "candidate_script.py", "\n")
    result_b = client.generate_diff("goal B", "candidate_script.py", "\n")

    assert result_a.strip() != result_b.strip()
    first_prompt = client.client.post.call_args_list[0].kwargs["json"]["messages"][1]["content"]
    second_prompt = client.client.post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
    assert "goal A" in first_prompt
    assert "goal B" in second_prompt


def test_base_url_is_joined_by_hand_not_via_httpx_client_base_url():
    """
    Regression guard: httpx.Client(base_url=...) merges a leading-slash
    request path via RFC 3986 URL-join rules, which silently DROPS a
    base_url path component like "/v1" instead of appending to it. This
    client must build the full URL itself so "/v1" (or any other path
    prefix a local server uses) survives.
    """
    client = _make_client([_fake_response(GOOD_DIFF)])
    client.generate_diff("goal", "candidate_script.py", "\n")
    called_url = client.client.post.call_args.args[0]
    assert called_url == "http://localhost:11434/v1/chat/completions"


def test_local_llm_client_implements_the_llm_client_interface():
    assert issubclass(LocalLLMClient, LLMClient)


def test_max_apply_retries_is_configurable():
    """orchestrator/run.py wires generation.max_apply_retries through here - a value beyond the default 3 must actually raise the retry budget, not be silently capped."""
    client = _make_client([_fake_response(BAD_DIFF)] * 5, max_apply_retries=5)
    client.generate_diff("goal", "candidate_script.py", "\n")
    assert client.client.post.call_count == 5
