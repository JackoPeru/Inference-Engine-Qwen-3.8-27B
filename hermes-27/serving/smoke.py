#!/usr/bin/env python3
"""Real-server smoke checks; expects a running local hermes_server."""
import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request


def request(url, payload=None, headers=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    outgoing = dict(headers or {})
    if payload is not None and "Content-Type" not in outgoing:
        outgoing["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=outgoing, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def event_stream(url, payload, on_first=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    events, current = [], []
    with urllib.request.urlopen(req, timeout=300) as res:
        for raw in res:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line:
                current.append(line)
                continue
            data = next((part[6:] for part in current if part.startswith("data: ")), None)
            current.clear()
            if data is None:
                continue
            if data == "[DONE]":
                events.append(data)
                break
            event = json.loads(data)
            events.append(event)
            if on_first and len(events) == 1:
                on_first(event)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="hermes-27")
    parser.add_argument("--backpressure", action="store_true", help="also require one 429; run server with --active 2 --queue 1")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, body = request(base + "/health")
    assert status == 200 and json.loads(body)["ready"] is True, (status, body)
    status, body = request(base + "/v1/models")
    assert status == 200 and any(x["id"] == args.model for x in json.loads(body)["data"]), (status, body)
    status, body = request(base + "/metrics")
    assert status == 200 and b"hermes_requests_active" in body, (status, body)
    status, body = request(base + "/v1/chat/completions", {"tools": []})
    assert status == 400 and json.loads(body)["error"]["code"] == "unsupported_parameter", (status, body)
    status, body = request(base + "/v1/chat/completions", {
        "model": "__nonexistent__", "messages": [{"role": "user", "content": "test"}],
    })
    assert status == 404 and json.loads(body)["error"]["code"] == "model_not_found", (status, body)
    status, body = request(base + "/v1/chat/completions", {
        "model": args.model, "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 4294967297,
    })
    assert status == 400 and json.loads(body)["error"]["code"] == "invalid_value", (status, body)
    status, body = request(base + "/__missing_route__")
    assert status == 404 and json.loads(body)["error"]["code"] == "not_found", (status, body)

    def nonstream(prompt):
        code, content = request(base + "/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0,
        })
        result = json.loads(content)
        assert code == 200 and result["usage"]["completion_tokens"] == 1, (code, result)
        return result

    def streaming(prompt):
        events = event_stream(base + "/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "stream": True,
        })
        assert events[-1] == "[DONE]" and any(isinstance(e, dict) and e["choices"][0]["finish_reason"] in {"stop", "length"} for e in events), events
        return events

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(nonstream, "Answer with one short word: blue")
        b = pool.submit(streaming, "Answer with one short word: green")
        result_a, result_b = a.result(), b.result()
    print("concurrent non-stream + SSE: PASS", result_a["id"], result_b[0]["id"])

    # Greedy generation is deterministic. SSE content must reconstruct exactly
    # the text from a separate non-stream request with the same chat template.
    parity_prompt = [{"role": "user", "content": "Reply briefly: café 🚀 中文."}]
    parity_payload = {
        "model": args.model, "messages": parity_prompt,
        "max_tokens": 8, "temperature": 0,
    }
    status, body = request(base + "/v1/chat/completions", parity_payload)
    assert status == 200, (status, body)
    reference_text = json.loads(body)["choices"][0]["message"]["content"]
    parity_events = event_stream(
        base + "/v1/chat/completions", {**parity_payload, "stream": True}
    )
    assert parity_events[-1] == "[DONE]", parity_events
    streamed_text = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in parity_events if isinstance(event, dict) and "choices" in event
    )
    assert streamed_text == reference_text, (repr(streamed_text), repr(reference_text))
    print("SSE vs non-stream exact text parity: PASS")

    cancelled_id = []
    def cancel_after_role(event):
        cancelled_id.append(event["id"])
        code, body = request(base + "/v1/cancel", {"id": cancelled_id[0]})
        assert code == 202, (code, body)

    events = event_stream(base + "/v1/chat/completions", {
        "model": args.model, "messages": [{"role": "user", "content": "Write a long answer about the color blue."}],
        "max_tokens": 32, "temperature": 0, "stream": True,
    }, on_first=cancel_after_role)
    assert events[-1] == "[DONE]" and any(isinstance(e, dict) and e["choices"][0]["finish_reason"] == "cancelled" for e in events), events
    print("stream cancellation: PASS", cancelled_id[0])

    if args.backpressure:
        def queued(i):
            return request(base + "/v1/chat/completions", {
                "model": args.model, "messages": [{"role": "user", "content": f"Queue check request {i}: explain a blue sky."}],
                "max_tokens": 1, "temperature": 0,
            })[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(queued, range(4)))
        assert 429 in statuses and 200 in statuses, statuses
        print("bounded queue / 429 backpressure: PASS", statuses)
    print("M3 smoke: PASS")


if __name__ == "__main__":
    main()
