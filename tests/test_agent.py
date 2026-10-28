"""Offline workflow tests; real configuration and clients are forbidden."""
from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from pydantic import ValidationError

from utils.agent import ClarificationNeeded, TaskPlan, plan_goal, run_agent, validate_plan
from utils import gemini


def image_bytes():
    stream = BytesIO()
    with Image.new("RGB", (12, 8), "green") as image:
        image.save(stream, format="PNG")
    return stream.getvalue()


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.image = image_bytes()
        self.client = Mock()
        self.call = self.client.models.generate_content
        self.call.return_value = SimpleNamespace(text='{"caption":"Green surface."}', candidates=[])
        for target in ("load_dotenv", "genai.Client"):
            guard = patch("utils.gemini." + target, side_effect=AssertionError("No real configuration/client"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_caption_only_runs_one_tool_once(self):
        with patch("utils.agent.check_image_quality") as quality:
            result = run_agent(self.image, "Write a minimal caption for Shutterstock", client=self.client)
        self.assertEqual(list(result["tasks"]), ["caption"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["plan"]["caption_style"], "minimal")
        quality.assert_not_called()
        self.call.assert_called_once()

    def test_full_stock_package_one_gemini_request(self):
        self.call.return_value = SimpleNamespace(text=json.dumps({
            "analysis": "A green background.", "caption": "Green surface.",
            "description": " ".join(["visible"] * 80), "keywords": ["green", "background"],
        }), candidates=[])
        events = []
        result = run_agent(self.image, "Prepare this image for my Shutterstock portfolio", keyword_count=2,
                           client=self.client, progress=lambda task, state: events.append((task, state)))
        self.assertEqual(len(result["tasks"]), 7)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["stock_readiness"]["resolution_result"], "below_minimum")
        self.assertTrue(result["summary"]["execution_success"])
        self.assertEqual(result["tasks"]["caption"]["status"], "completed")
        self.assertEqual(result["outputs"]["quality"]["width"], 12)
        self.assertIn(("caption", "running"), events)
        self.assertIn(("export", "completed"), events)
        self.call.assert_called_once()

    def test_quality_only_needs_no_client_or_key(self):
        result = run_agent(self.image, "Check brightness and blur")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(list(result["outputs"]), ["quality"])
        self.call.assert_not_called()

    def test_invalid_plan_falls_back_without_arbitrary_dispatch(self):
        result = run_agent(self.image, "Write a caption", plan={"tasks": ["execute_shell"]}, client=self.client)
        self.assertEqual(list(result["tasks"]), ["caption"])
        self.assertTrue(result["warnings"])
        self.call.assert_called_once()

    def test_inconsistent_allowlisted_plan_does_not_expand_scope(self):
        result = run_agent(self.image, "Caption only", plan={"tasks": ["caption", "quality"]}, client=self.client)
        self.assertEqual(list(result["tasks"]), ["caption"])

    def test_validated_model_bypass_is_rechecked(self):
        with self.assertRaises(ValidationError):
            validate_plan(TaskPlan.model_construct(tasks=["shell"]))

    def test_duplicate_and_extra_fields_rejected(self):
        for proposal in ({"tasks": ["caption", "caption"]}, {"tasks": ["caption"], "code": "bad"}):
            with self.subTest(proposal=proposal), self.assertRaises(ValidationError):
                validate_plan(proposal)

    def test_unclear_or_unsupported_goal_does_not_call_tools(self):
        for goal in ("Help me", "Crop this image", "caption without keywords", "caption or description", "export"):
            with self.subTest(goal=goal), self.assertRaises(ClarificationNeeded):
                run_agent(self.image, goal, client=self.client)
        self.call.assert_not_called()

    def test_natural_language_preferences(self):
        plan = plan_goal("Create 12 keywords and a short description")
        self.assertEqual(plan.keyword_count, 12)
        self.assertEqual(plan.description_length, "short")
        self.assertEqual(plan.tasks, ["description", "keywords"])

    def test_invalid_keyword_count_needs_clarification(self):
        with self.assertRaises(ClarificationNeeded):
            plan_goal("Create 200 keywords")

    def test_settings_cannot_override_requested_tasks(self):
        with self.assertRaises(ClarificationNeeded):
            plan_goal("Caption only", tasks=["caption", "quality"])

    def test_invalid_image_stops_before_api(self):
        result = run_agent(b"not image", "caption", client=self.client)
        self.assertEqual(result["status"], "failed")
        self.call.assert_not_called()

    def test_partial_model_fields_preserved_without_retry(self):
        self.call.return_value = SimpleNamespace(text='{"caption":"Green surface.","description":null}', candidates=[])
        result = run_agent(self.image, "caption and description", client=self.client)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["outputs"]["caption"], "Green surface.")
        self.assertEqual(result["tasks"]["description"]["status"], "failed")
        self.call.assert_called_once()

    def test_malformed_response_keeps_local_quality(self):
        self.call.return_value = SimpleNamespace(text="invalid JSON", candidates=[])
        result = run_agent(self.image, "caption and quality", client=self.client)
        self.assertEqual(result["status"], "partial")
        self.assertIn("quality", result["outputs"])
        self.call.assert_called_once()

    def test_missing_key_report_is_clear_and_local_result_survives(self):
        with patch.object(gemini, "_create_client", side_effect=gemini.GeminiError("GEMINI_API_KEY is not configured.")):
            result = run_agent(self.image, "caption and quality")
        self.assertEqual(result["status"], "partial")
        self.assertIn("not configured", result["tasks"]["caption"]["message"])

    def test_quality_failure_keeps_caption(self):
        with patch("utils.agent.check_image_quality", side_effect=RuntimeError("private information")):
            result = run_agent(self.image, "caption and quality", client=self.client)
        self.assertEqual(result["status"], "partial")
        self.assertIn("caption", result["outputs"])
        self.assertNotIn("private information", str(result))

    def test_shortfall_is_needs_review_not_complete(self):
        self.call.return_value = SimpleNamespace(text='{"keywords":["green"]}', candidates=[])
        result = run_agent(self.image, "Create 10 keywords", client=self.client)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["tasks"]["keywords"]["status"], "needs_review")
        self.assertEqual(self.call.call_count, 2)
        self.assertIn("Requested 10; returned 1", result["tasks"]["keywords"]["message"])
        self.assertIn("keywords", result["summary"]["needs_review"])

    def test_readiness_is_local_and_threshold_is_configurable(self):
        result = run_agent(self.image, "Check stock readiness", minimum_resolution_mp=0.000096)
        self.assertEqual(result["stock_readiness"]["megapixels"], 0.000096)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(list(result["tasks"]), ["readiness"])
        self.call.assert_not_called()

    def test_failed_validation_reports_skipped_tasks_and_unavailable_readiness(self):
        result = run_agent(b"invalid", "Prepare a complete stock package", client=self.client)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["summary"]["execution_success"])
        self.assertIn("caption", result["summary"]["skipped"])
        self.assertEqual(result["stock_readiness"]["resolution_result"], "unavailable")
        self.assertTrue(result["summary"]["next_steps"])
        self.call.assert_not_called()

    def test_keyword_refinement_failure_does_not_fail_caption(self):
        self.call.side_effect = [
            SimpleNamespace(text='{"caption":"Green.","keywords":["green"]}', candidates=[]),
            RuntimeError("private details"),
        ]
        result = run_agent(self.image, "caption and 5 keywords", client=self.client)
        self.assertEqual(result["tasks"]["caption"]["status"], "completed")
        self.assertEqual(result["tasks"]["keywords"]["status"], "needs_review")
        self.assertEqual(result["summary"]["failed"], [])
        self.assertNotIn("private details", str(result))


if __name__ == "__main__":
    unittest.main()
