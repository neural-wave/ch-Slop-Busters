"""The coders against a mocked HTTP layer (never a live Ollama): request body and context guard, malformed responses
(ported from daniel-eval), the loopback guard, retries and failures, unload, and the knn, lexical and command coders."""

from __future__ import annotations

import http.server
import json
import sys
import threading
import urllib.error

import pytest

from eval.coders import ollama
from eval.coders.base import CoderError
from eval.coders.command import Command
from eval.coders.knn import Knn
from eval.coders.lexical import Lexical, label_space

DIGEST = "f" * 64


def response(text: str = '{"codes":["I10"]}', reason: str = "stop", **kw) -> dict:
    return {"done": True, "done_reason": reason, "response": text, "prompt_eval_count": 10, "eval_count": 5, "total_duration": 7, **kw}


class FakeServer:
    """Stands in for ollama.request: tags, version, ps and a queue of /api/generate answers (an exception is raised)."""

    def __init__(self, answers=(), loaded=(), vram: int = 100):
        self.answers, self.loaded, self.calls, self.vram = list(answers), list(loaded), [], vram

    def __call__(self, path, body=None, timeout=180):
        self.calls.append((path, body))
        if path == "/api/tags":
            return {"models": [{"name": ollama.DEFAULT_MODEL, "digest": DIGEST, "details": {}}]}
        if path == "/api/version":
            return {"version": "test"}
        if path == "/api/ps":
            return {"models": [{"name": n, "size": 100, "size_vram": self.vram} for n in self.loaded]}
        if body.get("keep_alive") == 0:
            self.loaded.remove(body["model"])
            return {"done": True}
        if body["model"] not in self.loaded:
            self.loaded.append(body["model"])
        ans = self.answers.pop(0) if self.answers else response()
        if isinstance(ans, BaseException):
            raise ans
        return ans

    def generated(self) -> list[dict]:
        return [b for p, b in self.calls if p == "/api/generate" and "prompt" in b]


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(ollama, "request", fake)
    return fake


def rec(eid="one", note="Essential hypertension.", dialogue="Doctor: blood pressure is high.", codes=None):
    codes = [{"code": "I10", "description": "x"}] if codes is None else codes
    return {"encounter_id": eid, "dataset": "toy", "dialogue": dialogue, "note": note, "icd10_codes": codes}


# request body, context guard


def test_body_matches_daniel_eval_and_guards_the_context():
    b = ollama.body("m", "note", ollama.SCHEMA)
    assert b == {"model": "m", "prompt": "note", "stream": False, "think": False, "format": ollama.SCHEMA,
                 "options": {"temperature": 0, "seed": 0, "num_ctx": 16384, "num_predict": 512}, "keep_alive": "10m"}  # fmt: skip
    assert ollama.body("m", "note", {}, device="cpu")["options"]["num_gpu"] == 0
    with pytest.raises(ollama.PredictionError, match="context_exceeded"):
        ollama.body("m", "a" * 10000, {}, context=2048)  # rejected, never truncated
    assert ollama.body("m", "a" * (2048 - 512 - 1024), {}, context=2048)  # exactly at the budget
    with pytest.raises(ValueError):
        ollama.body("m", "note", {}, device="gpu")


def test_gold_never_changes_prediction_input(server):
    c = ollama.LlmCoder()
    a, b = rec(codes=[{"code": "I10", "description": "Essential (primary) hypertension"}]), rec(codes=[{"code": "E11.9", "description": "Type 2 diabetes"}])
    assert c.prompt(a) == c.prompt(b)
    assert "I10" not in c.prompt(a) and "hypertension." in c.prompt(a) and "blood pressure" not in c.prompt(a)
    both = ollama.LlmCoder("llm_both", inputs="both")
    assert "DIALOGUE:\nDoctor: blood pressure is high.\n\nPHYSICIAN NOTE:\nEssential hypertension." in both.prompt(a)
    assert both.config != c.config and "inputs=both" in both.config and DIGEST[:12] in c.config


# malformed responses


@pytest.mark.parametrize(("resp", "code"), [
    (response(reason="length"), "incomplete_generation"),
    ({**response(), "done": False}, "incomplete_generation"),
    ({"done": True, "done_reason": "stop"}, "missing_response"),
    (response('prose {"codes":["I10"]}'), "invalid_json"),
    (response('{"codes":[],"codes":["I10"]}'), "invalid_json"),
    (response("[]"), "not_an_object"),
])  # fmt: skip
def test_malformed_responses_are_errors(resp, code):
    with pytest.raises(ollama.PredictionError) as e:
        ollama.parse(resp)
    assert e.value.code == code


@pytest.mark.parametrize(("obj", "code"), [
    ({"codes": "I10"}, "unexpected_code_type"), ({}, "unexpected_response_schema"), ({"codes": [{"code": "I10"}]}, "unexpected_code_type"),
    ({"codes": [], "extra": 1}, "unexpected_response_schema"), ({"codes": ["I10"] * 33}, "too_many_codes"),
])  # fmt: skip
def test_unexpected_answers_are_errors(obj, code):
    with pytest.raises(ollama.PredictionError) as e:
        ollama.codes_of(obj)
    assert e.value.code == code


def test_empty_answer_is_valid_and_invalid_codes_are_dropped_and_counted():
    assert ollama.codes_of(ollama.parse(response('{"codes":[]}'))) == ([], 0)
    assert ollama.codes_of({"codes": ["made up", "e119", "E11.9", "I10"]}) == (["E11.9", "I10"], 1)


@pytest.mark.parametrize(("exc", "code"), [
    (urllib.error.URLError(ConnectionRefusedError()), "unreachable"), (urllib.error.URLError(TimeoutError()), "timeout"), (TimeoutError(), "timeout"),
    (urllib.error.HTTPError("u", 500, "boom", {}, None), "http_500"), (ValueError("bad json"), "invalid_http_json"), (OSError(), "unreachable"),
])  # fmt: skip
def test_transport_failures_become_prediction_errors(server, exc, code):
    server.answers = [exc]
    with pytest.raises(ollama.PredictionError) as e:
        ollama.generate_json("m", "p", {})
    assert e.value.code == code


def test_generate_json_returns_the_object(server):
    server.answers = [response('{"name":"x","age":3}')]
    assert ollama.generate_json(ollama.DEFAULT_MODEL, "p", {"type": "object"}, seed=1, device="cpu") == {"name": "x", "age": 3}
    assert server.generated()[0]["options"] == {"temperature": 0, "seed": 1, "num_ctx": 16384, "num_predict": 512, "num_gpu": 0}


# loopback guard


@pytest.mark.parametrize(("url", "local"), [
    ("http://127.0.0.1:11434", True), ("http://localhost:11434", True), ("http://[::1]:11434", True), ("http://127.0.0.2:80", True),
    ("http://10.0.0.5:11434", False), ("https://ollama.example.org", False), ("http://[2001:db8::1]:11434", False),
])  # fmt: skip
def test_loopback(url, local):
    assert ollama.loopback(url) is local


def test_non_loopback_endpoint_is_recorded(server, monkeypatch):
    assert ollama.LlmCoder().endpoint is None
    monkeypatch.setattr(ollama, "URL", "http://10.0.0.5:11434")
    assert ollama.endpoint() == "http://10.0.0.5:11434" and ollama.LlmCoder().endpoint == "http://10.0.0.5:11434"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://10.255.255.1/api/version")
            self.end_headers()
            return
        body = b'{"version": "local"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_no_proxy_and_no_redirect(monkeypatch):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(ollama, "URL", f"http://127.0.0.1:{httpd.server_port}")
        for var in ("http_proxy", "HTTP_PROXY", "all_proxy"):
            monkeypatch.setenv(var, "http://10.255.255.1:3128")  # would hang or fail if it were used
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        assert ollama.request("/api/version", timeout=5) == {"version": "local"}
        with pytest.raises(urllib.error.HTTPError) as e:
            ollama.request("/redirect", timeout=5)
        assert e.value.code == 302
    finally:
        httpd.shutdown()
        httpd.server_close()


# the LLM coder


def test_predictions_retries_and_usage(server):
    server.answers = [response('{"codes":["e119","E11.9","I10"]}'), response(reason="length"), response('{"codes":["J45.909"]}')]
    c = ollama.LlmCoder(retries=1)
    out = c.predict([rec("a"), rec("b")])
    assert out["a"].codes == ["E11.9", "I10"] and out["a"].attempts == 1 and out["a"].error is None
    assert out["a"].usage == {"prompt_eval_count": 10, "eval_count": 5, "total_duration": 7}
    assert out["b"].codes == ["J45.909"] and out["b"].attempts == 2


@pytest.mark.parametrize(("vram", "where"), [(100, "gpu"), (40, "partial"), (0, "cpu")])
def test_backend_is_recorded(server, vram, where):
    server.vram = vram
    assert ollama.backend(ollama.DEFAULT_MODEL) is None  # not loaded yet
    assert ollama.LlmCoder().predict([rec("a")])["a"].backend == where
    assert ollama.backend("qwen3.5") is None  # another model


def test_failed_prediction_after_retries(server):
    server.answers = [response(reason="length")] * 2
    p = ollama.LlmCoder(retries=1).predict([rec("a")])["a"]
    assert p.codes == [] and p.error == "incomplete_generation" and p.attempts == 2


def test_three_consecutive_failures_fail_the_coder(server):
    server.answers = [response(reason="length")] * 3 + [response()]
    c = ollama.LlmCoder(retries=0)
    seen = []
    with pytest.raises(CoderError, match="3 consecutive"):
        c.predict([rec(str(i)) for i in range(5)], on_result=lambda r, p: seen.append(p.error))
    assert seen == ["incomplete_generation"] * 3


def test_a_success_resets_the_failure_count(server):
    server.answers = [response(reason="length")] * 2 + [response()] + [response(reason="length")] * 2
    out = ollama.LlmCoder(retries=0).predict([rec(str(i)) for i in range(5)])
    assert [p.error for p in out.values()] == ["incomplete_generation"] * 2 + [None] + ["incomplete_generation"] * 2


def test_context_overflow_is_an_error_without_retry_or_coder_failure(server):
    c = ollama.LlmCoder(context=2048, retries=3)
    out = c.predict([rec(str(i), note="x" * 5000) for i in range(4)])
    assert all(p.error == "context_exceeded" and p.attempts == 1 for p in out.values())
    assert server.generated() == []


def test_empty_text_is_an_empty_set_without_a_call(server):
    p = ollama.LlmCoder().predict([rec("a", note="  ")])["a"]
    assert p.codes == [] and p.error is None and server.generated() == []


def test_unreachable_server_fails_the_coder(monkeypatch):
    def down(*a, **kw):
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr(ollama, "request", down)
    with pytest.raises(CoderError, match="unreachable"):
        ollama.LlmCoder()
    assert ollama.available() is False
    ollama.unload()  # a no-op, no exception


def test_model_not_installed(server):
    with pytest.raises(CoderError, match="model_not_installed"):
        ollama.LlmCoder(model="other:1b")


def test_unload(server):
    server.loaded = [ollama.DEFAULT_MODEL, "other:1b"]
    ollama.unload("missing:7b")  # not loaded: left alone (an empty request would load it)
    assert [b for p, b in server.calls if p == "/api/generate"] == []
    ollama.unload(ollama.DEFAULT_MODEL)
    assert server.loaded == ["other:1b"]
    ollama.unload()
    assert server.loaded == []
    assert [b for p, b in server.calls if p == "/api/generate"] == [{"model": ollama.DEFAULT_MODEL, "keep_alive": 0}, {"model": "other:1b", "keep_alive": 0}]


# knn, lexical, command


TRAIN = [
    {"encounter_id": "t1", "dialogue": "", "note": "asthma wheezing inhaler", "icd10_codes": [{"code": "J45.909", "description": "Asthma"}]},
    {"encounter_id": "t2", "dialogue": "", "note": "diabetes glucose insulin", "icd10_codes": [{"code": "E11.9", "description": "Type 2 diabetes"}]},
    {"encounter_id": "t3", "dialogue": "", "note": "wheezing asthma cough", "icd10_codes": [{"code": "J45.909", "description": "Asthma"}, {"code": "R05.9", "description": "Cough"}]},
]  # fmt: skip


def test_knn_ranks_neighbour_codes_and_skips_itself():
    knn = Knn(k=2, train=TRAIN, vocab=frozenset({"asthma", "wheezing", "inhaler", "diabetes", "glucose", "insulin", "cough"}))
    assert knn.ranked and knn.inputs == "both" and knn.config.startswith("knn:k=2:clinical:train=")
    out = knn.predict([{"encounter_id": "q", "dialogue": "wheezing", "note": "asthma, inhaler"}, {**TRAIN[2]}])
    assert out["q"].codes[0] == "J45.909"
    assert out["t3"].codes[0] == "J45.909" and "R05.9" not in out["t3"].codes  # its own codes are never retrieved


def test_lexical_ranks_by_description_words():
    lex = Lexical(label_space(TRAIN))
    assert lex.codes == ["E11.9", "J45.909", "R05.9"]
    assert lex.predict([{"encounter_id": "q", "dialogue": "", "note": "type 2 diabetes"}])["q"].codes[0] == "E11.9"


def script(tmp_path, body: str) -> str:
    p = tmp_path / "coder.py"
    p.write_text("import json, sys, time\nrows = [json.loads(line) for line in open(sys.argv[1])]\n" + body)
    return f"{sys.executable} -I {p} {{input}} {{output}}"


def test_command_contract_and_missing_records(tmp_path):
    body = ("assert all(r['icd10_codes'] == [] for r in rows)\n"
            "open(sys.argv[2], 'w').write(json.dumps({'encounter_id': rows[0]['encounter_id'], 'codes': ['I10', 'E11.9']}) + '\\nnot json\\n')\n")
    cmd = Command(script(tmp_path, body))
    assert cmd.name.startswith("cmd-") and cmd.config.startswith("cmd:") and cmd.ranked
    out = cmd.predict([rec("a"), rec("b")])
    assert out["a"].codes == ["I10", "E11.9"] and out["a"].error is None
    assert out["b"].codes == [] and out["b"].error == "missing_output"


def test_command_failure_timeout_and_cannot_start(tmp_path):
    assert Command(script(tmp_path, "sys.exit(3)\n")).predict([rec("a")])["a"].error == "exit_3"
    assert Command(script(tmp_path, "time.sleep(30)\n"), timeout=0.5).predict([rec("a")])["a"].error == "timeout"
    with pytest.raises(CoderError, match="cannot start"):
        Command([str(tmp_path / "missing-binary"), "{input}", "{output}"]).predict([rec("a")])


def test_command_list_template_placeholders(tmp_path):
    cmd = Command(["{python}", "-I", str(tmp_path / "c.py"), "{input}", "{output}"], name="paper", ranked=False)
    (tmp_path / "c.py").write_text("import json, sys\nopen(sys.argv[2], 'w').write(json.dumps({'encounter_id': 'a', 'codes': 'I10'}))\n")
    assert cmd.name == "paper" and not cmd.ranked and json.loads(cmd.config[4:])[0] == "{python}"
    assert cmd.predict([rec("a")])["a"].error == "invalid_output"
