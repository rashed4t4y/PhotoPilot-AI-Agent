"""Offline Gemini contract tests. Configuration and real clients are blocked."""
import importlib
import json
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from google.genai import errors, types
from PIL import Image

from utils import gemini


def response(data):
    return SimpleNamespace(text=json.dumps(data), candidates=[], prompt_feedback=None)


class GeminiTests(unittest.TestCase):
    def setUp(self):
        stream = BytesIO()
        with Image.new("RGB", (8, 8), "green") as image:
            image.save(stream, format="PNG")
        self.image = stream.getvalue()
        self.client = Mock()
        self.call = self.client.models.generate_content
        self.call.return_value = response({"caption": "Green fills the frame."})
        # Any accidental access to real configuration or client creation fails.
        for target in ("load_dotenv", "os.getenv", "genai.Client"):
            guard = patch("utils.gemini." + target, side_effect=AssertionError("Forbidden real configuration access"))
            guard.start()
            self.addCleanup(guard.stop)
        sleep = patch("utils.gemini.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)

    def generate(self, **kwargs):
        kwargs.setdefault("fields", ("caption",))
        return gemini.generate_metadata(self.image, client=self.client, **kwargs)

    def test_import_has_no_configuration_side_effects(self):
        with patch("dotenv.load_dotenv", side_effect=AssertionError("No dotenv reads")):
            importlib.reload(gemini)

    def test_caption_uses_structured_schema_and_grounding(self):
        result = self.generate(caption_style="minimal")
        self.assertEqual(result["caption"], "Green fills the frame.")
        self.call.assert_called_once()
        kwargs = self.call.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        config = kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema["required"], ["caption"])
        self.assertIn("identities", config.system_instruction)
        self.assertIn("untrusted", config.system_instruction)
        self.assertIn("minimal", kwargs["contents"][1])
        self.assertNotIn("keywords:", kwargs["contents"][1])
        self.client.close.assert_not_called()

    def test_large_image_payload_is_bounded_without_changing_source(self):
        stream = BytesIO()
        with Image.new("RGB", (3000, 1500), "green") as image:
            image.save(stream, format="PNG")
        original = stream.getvalue()
        captured_sizes = []

        def capture(**kwargs):
            captured_sizes.append(kwargs["contents"][0].size)
            return response({"caption": "Green surface."})

        self.call.side_effect = capture
        gemini.generate_metadata(stream, fields=("caption",), client=self.client)
        self.assertEqual(captured_sizes, [(2048, 1024)])
        self.assertEqual(stream.getvalue(), original)

    def test_all_fields_single_call_and_keyword_normalization(self):
        self.call.return_value = response({
            "analysis": "A green surface fills the frame.",
            "caption": "Green fills the frame.",
            "description": "A green surface.",
            "keywords": [" green ", "GREEN", "plain   background"],
        })
        result = self.generate(fields=gemini.FIELDS, keyword_count=5)
        self.assertEqual(result["keywords"], ["green", "plain background"])
        self.assertEqual(len(result["warnings"]), 2)
        self.call.assert_called_once()

    def test_keyword_count_caps_without_extra_request(self):
        self.call.return_value = response({"keywords": ["green", "background", "plain"]})
        self.assertEqual(self.generate(fields=("keywords",), keyword_count=2)["keywords"], ["green", "background"])
        self.call.assert_called_once()

    def test_missing_malformed_or_wrong_shape_never_retried(self):
        for text in (None, "", "not JSON", "[]", "{}", '{"caption": null}',
                     '{"caption": " "}', '{"caption": 123}',
                     '{"caption":"ok","unexpected":true}',
                     json.dumps({"caption": "word " * 21})):
            with self.subTest(text=text):
                self.call.reset_mock()
                self.call.return_value = SimpleNamespace(text=text, candidates=[])
                with self.assertRaises(gemini.GeminiResponseError):
                    self.generate()
                self.call.assert_called_once()
        self.sleep.assert_not_called()

    def test_invalid_keywords(self):
        for values in ([], "green", [12], [""], ["green,blue"], ["#green"]):
            with self.subTest(values=values):
                self.call.return_value = response({"keywords": values})
                with self.assertRaises(gemini.GeminiResponseError):
                    self.generate(fields=("keywords",))

    def test_sdk_response_and_finish_reason(self):
        self.call.return_value = types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(parts=[types.Part(text='{"caption":"Green surface."}')]),
                finish_reason="STOP",
            )]
        )
        self.assertEqual(self.generate()["caption"], "Green surface.")
        self.call.return_value.candidates[0].finish_reason = types.FinishReason.MAX_TOKENS
        with self.assertRaises(gemini.GeminiResponseError):
            self.generate()

    def test_blocked_response(self):
        self.call.return_value = SimpleNamespace(
            text='{"caption":"Green."}', candidates=[],
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        )
        with self.assertRaises(gemini.GeminiResponseError):
            self.generate()

    def test_auth_and_bad_request_are_not_retried_or_leaked(self):
        for code in (400, 401, 403, 404):
            with self.subTest(code=code):
                self.call.reset_mock()
                self.call.side_effect = errors.APIError(code, {"message": "SENSITIVE PROVIDER DETAILS"})
                with self.assertRaises(gemini.GeminiError) as raised:
                    self.generate()
                self.assertNotIn("SENSITIVE", str(raised.exception))
                self.call.assert_called_once()
        self.sleep.assert_not_called()

    def test_rate_limit_retry_is_bounded(self):
        self.call.side_effect = errors.APIError(429, {"message": "quota"})
        with self.assertRaisesRegex(gemini.GeminiError, "quota"):
            self.generate(max_attempts=3)
        self.assertEqual(self.call.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1, 2])

    def test_transient_service_error_recovers(self):
        self.call.side_effect = [errors.APIError(503, {}), response({"caption": "Green."})]
        self.assertEqual(self.generate()["caption"], "Green.")
        self.assertEqual(self.call.call_count, 2)

    def test_connect_error_retries_but_read_timeout_does_not(self):
        self.call.side_effect = [httpx.ConnectError("private URL"), response({"caption": "Green."})]
        self.assertEqual(self.generate()["caption"], "Green.")
        self.call.reset_mock()
        self.call.side_effect = httpx.ReadTimeout("private URL")
        with self.assertRaisesRegex(gemini.GeminiError, "network"):
            self.generate()
        self.call.assert_called_once()

    def test_unexpected_failure_is_sanitized(self):
        self.call.side_effect = RuntimeError("sensitive raw details")
        with self.assertRaises(gemini.GeminiError) as raised:
            self.generate()
        self.assertNotIn("sensitive", str(raised.exception))
        self.call.assert_called_once()

    def test_validation_happens_before_client_creation(self):
        with self.assertRaises(ValueError):
            gemini.generate_metadata(b"bad image")
        self.call.assert_not_called()

    def test_invalid_options_never_call_api(self):
        for options in ({"fields": ()}, {"fields": ("unknown",)}, {"fields": "caption"},
                        {"fields": ("caption", "caption")}, {"keyword_count": 0},
                        {"max_attempts": 4}, {"caption_style": "invalid"},
                        {"description_length": "invalid"}, {"model": ""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.generate(**options)
        self.call.assert_not_called()

    def test_legacy_function_returns_tuple_and_closes_owned_client(self):
        description = " ".join(["visible"] * 80)
        self.call.return_value = response({"caption": "Green surface.", "description": description})
        with patch.object(gemini, "_create_client", return_value=self.client):
            self.assertEqual(gemini.generate_content(BytesIO(self.image)), ("Green surface.", description))
        self.client.close.assert_called_once()
        self.call.assert_called_once()

    def test_missing_key_uses_mock_configuration_only(self):
        with patch.object(gemini, "load_dotenv") as dotenv, patch.object(gemini.os, "getenv", return_value=None):
            with self.assertRaisesRegex(gemini.GeminiError, "not configured"):
                gemini._create_client()
            dotenv.assert_called_once()

    def test_client_timeout_and_sdk_retry_disabled_with_dummy_key(self):
        with patch.object(gemini, "load_dotenv"), patch.object(gemini.os, "getenv", return_value="dummy-test-key"), patch.object(gemini.genai, "Client") as factory:
            gemini._create_client()
            options = factory.call_args.kwargs["http_options"]
            self.assertEqual(options.retry_options.attempts, 1)
            self.assertEqual(options.timeout, 60_000)

    def test_keyword_whitespace_case_and_empty_entries_normalized(self):
        self.call.return_value = response({"keywords": [" Green  leaf ", "GREEN\tLEAF", "", "  ", "plant"]})
        result = self.generate(fields=("keywords",), keyword_count=2, keyword_retry=True)
        self.assertEqual(result["keywords"], ["Green leaf", "plant"])
        self.assertEqual(result["keyword_review"]["actual_unique_count"], 2)
        self.call.assert_called_once()

    def test_one_refinement_can_resolve_shortfall_without_regenerating_caption(self):
        self.call.side_effect = [
            response({"caption": "Green.", "keywords": ["green"]}),
            response({"keywords": ["GREEN", "leaf", "plant"]}),
        ]
        result = self.generate(fields=("caption", "keywords"), keyword_count=3, keyword_retry=True, allow_partial=True)
        self.assertEqual(result["keywords"], ["green", "leaf", "plant"])
        self.assertEqual(result["caption"], "Green.")
        self.assertEqual(result["keyword_review"]["status"], "completed")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(self.call.call_count, 2)
        self.assertEqual(self.call.call_args.kwargs["config"].response_json_schema["required"], ["keywords"])

    def test_no_supported_additions_retains_original_and_stops(self):
        self.call.side_effect = [response({"keywords": ["green"]}), response({"keywords": []})]
        result = self.generate(fields=("keywords",), keyword_count=10, keyword_retry=True)
        self.assertEqual(result["keywords"], ["green"])
        self.assertEqual(result["keyword_review"]["status"], "needs_review")
        self.assertIn("fewer distinct concepts", result["keyword_review"]["reason"])
        self.assertEqual(result["keyword_review"]["refinement_attempts"], 1)
        self.assertEqual(self.call.call_count, 2)

    def test_refinement_failures_keep_successful_results_and_do_not_retry(self):
        for failure in (errors.APIError(401, {"message": "SECRET"}), errors.APIError(429, {}),
                        errors.APIError(503, {}), httpx.ReadTimeout("SECRET"),
                        SimpleNamespace(text="bad JSON", candidates=[]),
                        SimpleNamespace(text=None, candidates=[])):
            with self.subTest(failure=type(failure).__name__):
                self.call.reset_mock()
                self.call.side_effect = [response({"keywords": ["green"]}), failure]
                result = self.generate(fields=("keywords",), keyword_count=5, keyword_retry=True)
                self.assertEqual(result["keywords"], ["green"])
                self.assertTrue(result["keyword_review"]["refinement_failed"])
                self.assertEqual(result["keyword_review"]["status"], "needs_review")
                self.assertNotIn("SECRET", str(result))
                self.assertEqual(self.call.call_count, 2)

    def test_invalid_initial_keywords_do_not_trigger_refinement(self):
        for payload in ({"keywords": []}, {"keywords": [""]}, {"keywords": None}):
            self.call.reset_mock()
            self.call.return_value = response(payload)
            with self.assertRaises(gemini.GeminiResponseError):
                self.generate(fields=("keywords",), keyword_retry=True)
            self.call.assert_called_once()

    def test_transient_budget_plus_one_refinement_is_bounded(self):
        self.call.side_effect = [errors.APIError(503, {}), response({"keywords": ["green"]}), errors.APIError(503, {})]
        result = self.generate(fields=("keywords",), keyword_count=5, keyword_retry=True)
        self.assertEqual(self.call.call_count, 3)
        self.sleep.assert_called_once_with(1)
        self.assertEqual(result["keyword_review"]["status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
