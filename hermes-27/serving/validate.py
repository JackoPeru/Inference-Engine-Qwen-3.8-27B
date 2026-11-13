"""M3 serving: OpenAI-compatible request validation + SSE formatting + tests.
Pure logic (no network); the native HTTP layer will call into this contract.
Run: python3 api_validate.py  (asserts + prints PASS)
"""
import json
MAX_TOKENS_CAP = 8192
VALID_MODELS = ["hermes-27-qwen38-nvfp4"]
def validate_chat(body: dict) -> dict:
    assert isinstance(body, dict), "body must be object"
    msgs = body.get("messages")
    assert isinstance(msgs, list) and msgs, "messages non-empty list required"
    for m in msgs:
        assert isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool"), f"bad role {m}"
        c = m.get("content")
        assert c is None or isinstance(c, str) or isinstance(c, list), "bad content"
    mt = body.get("max_tokens", 512)
    assert isinstance(mt, int) and 1 <= mt <= MAX_TOKENS_CAP, "max_tokens out of range"
    temp = body.get("temperature", 0.7)
    assert isinstance(temp, (int, float)) and 0.0 <= temp <= 2.0, "temperature out of range"
    stream = body.get("stream", False)
    assert isinstance(stream, bool), "stream must be bool"
    model = body.get("model", VALID_MODELS[0])
    assert model in VALID_MODELS, f"unknown model {model}"
    think = body.get("reasoning", {})
    assert isinstance(think, dict), "reasoning must be object"
    return {"model": model, "messages": msgs, "max_tokens": mt,
            "temperature": float(temp), "stream": stream,
            "think": bool(think.get("enabled", True))}
def sse_chunk(delta: str, done: bool = False) -> str:
    if done:
        return "data: [DONE]\n\n"
    return "data: " + json.dumps({"choices": [{"delta": {"content": delta}}]}) + "\n\n"
def run_tests():
    ok = validate_chat({"messages": [{"role": "user", "content": "ciao"}], "max_tokens": 64,
                        "temperature": 0.0, "stream": True, "model": "hermes-27-qwen38-nvfp4"})
    assert ok["max_tokens"] == 64 and ok["stream"] is True
    for bad in [{}, {"messages": []}, {"messages": [{"role": "x", "content": "y"}]},
                {"messages": [{"role": "user", "content": "y"}], "max_tokens": 0},
                {"messages": [{"role": "user", "content": "y"}], "max_tokens": 99999},
                {"messages": [{"role": "user", "content": "y"}], "temperature": 5},
                {"messages": [{"role": "user", "content": "y"}], "model": "gpt-99"}]:
        try:
            validate_chat(bad)
            raise SystemExit(f"should have rejected {bad}")
        except AssertionError:
            pass
    c = sse_chunk("ciao")
    assert c.startswith("data: ") and c.endswith("\n\n") and json.loads(c[6:])
    assert sse_chunk("", done=True) == "data: [DONE]\n\n"
    print("API VALIDATE PASS")
if __name__ == "__main__":
    run_tests()
