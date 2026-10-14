"""Local HTTP checks for request accounting; no model, CUDA, or timing claims."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
SPEC = importlib.util.spec_from_file_location("dosa_ttft", ROOT / "benchmark/dosa/bench_ttft.py")
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.text = self.directory / "text.txt"
        self.input_text = "An unchanged real-text input.\r\n保留中文与换行。\rFinal line.\n"
        self.text.write_bytes(self.input_text.encode("utf-8"))
        self.output = self.directory / "result.json"
        self.actions = []
        self.completion_tokens = 1
        self.cached_tokens = 0
        case = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"chunked_prefill_size": 16384}')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                case.actions.append((self.path, body))
                self.send_response(200)
                self.end_headers()
                if self.path.startswith("/flush_cache"):
                    self.wfile.write(b'{"status":"succeed"}')
                    return
                event = dict(text="", output_ids=[154827] if case.completion_tokens else [],
                             meta_info=dict(prompt_tokens=131072, cached_tokens=case.cached_tokens,
                                            num_retractions=0, completion_tokens=case.completion_tokens,
                                            finish_reason={"type": "stop", "matched": 154827}))
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_each_request_is_independent_and_empty_text_eos_counts(self):
        result = client.measure(self.url, self.text, self.output, loaded_n=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(len(result["loaded_samples_s"]), 2)
        self.assertEqual([path.split("?")[0] for path, _ in self.actions],
                         ["/flush_cache", "/generate"] * 3)
        payloads = [body for path, body in self.actions if path == "/generate"]
        self.assertEqual(len({body["rid"] for body in payloads}), 3)
        self.assertTrue(all(body["text"] == self.input_text for body in payloads))
        self.assertTrue(all(body["sampling_params"] == {"max_new_tokens": 2} for body in payloads))

    def test_zero_output_and_cache_reuse_are_preserved_without_retry(self):
        for tokens, cached in [(0, 0), (1, 64)]:
            with self.subTest(tokens=tokens, cached=cached):
                self.completion_tokens, self.cached_tokens = tokens, cached
                output = self.directory / f"failure-{tokens}-{cached}.json"
                before = len(self.actions)
                with self.assertRaises(RuntimeError):
                    client.measure(self.url, self.text, output, loaded_n=2)
                saved = json.loads(output.read_text())
                self.assertEqual(saved["status"], "failed_preserved")
                self.assertEqual(len(saved["requests"]), 1)
                self.assertEqual(len(saved["requests"][0]["events"]), 1)
                self.assertEqual(len(self.actions) - before, 2)

    def test_existing_attempt_cannot_be_overwritten_or_replayed(self):
        self.output.write_text("retained attempt")
        with self.assertRaises(FileExistsError):
            client.measure(self.url, self.text, self.output, loaded_n=2)
        self.assertEqual(self.output.read_text(), "retained attempt")
        self.assertFalse(self.actions)


if __name__ == "__main__":
    unittest.main()
