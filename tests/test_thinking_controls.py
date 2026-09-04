import sys
import types
import unittest

try:
    import aiohttp  # noqa: F401
except ModuleNotFoundError:
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace()
    sys.modules["aiohttp"] = aiohttp

from tools import serve_openai


DEFAULTS = {
    "enable_thinking": True,
    "reasoning_effort": "low",
    "preserve_thinking": False,
}
BASE_REQUEST = {"messages": [{"role": "user", "content": "Hello"}]}


class FakeTokenizer:
    def __init__(self):
        self.call = None

    def hf_chat_template(self, messages, **kwargs):
        self.call = (messages, kwargs)
        return "rendered"


class ThinkingControlsTest(unittest.TestCase):
    def parse(self, **overrides):
        body = dict(BASE_REQUEST)
        body.update(overrides)
        return serve_openai.parse_request(body, DEFAULTS)

    def test_server_defaults_use_low_reasoning_without_history(self):
        request, error = self.parse()

        self.assertIsNone(error)
        self.assertEqual(request["chat_template_kwargs"], DEFAULTS)
        self.assertTrue(request["enable_thinking"])
        self.assertEqual(request["sampling"]["temperature"], 1.0)
        self.assertEqual(request["sampling"]["top_p"], 0.95)
        self.assertEqual(request["sampling"]["pres_p"], 0.0)

    def test_nested_enable_thinking_false_disables_reasoning(self):
        request, error = self.parse(
            chat_template_kwargs={"enable_thinking": False}
        )

        self.assertIsNone(error)
        self.assertEqual(
            request["chat_template_kwargs"],
            {"enable_thinking": False, "preserve_thinking": False},
        )
        self.assertFalse(request["enable_thinking"])
        self.assertEqual(request["sampling"]["temperature"], 0.7)
        self.assertEqual(request["sampling"]["top_p"], 0.8)
        self.assertEqual(request["sampling"]["pres_p"], 1.5)

    def test_top_level_reasoning_effort_overrides_default(self):
        request, error = self.parse(reasoning_effort="medium")

        self.assertIsNone(error)
        self.assertEqual(
            request["chat_template_kwargs"]["reasoning_effort"], "medium"
        )
        self.assertTrue(request["enable_thinking"])

    def test_none_effort_is_an_off_alias(self):
        request, error = self.parse(reasoning_effort="none")

        self.assertIsNone(error)
        self.assertFalse(request["enable_thinking"])
        self.assertNotIn("reasoning_effort", request["chat_template_kwargs"])

    def test_conflicting_controls_are_rejected(self):
        request, error = self.parse(
            reasoning_effort="low",
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertIsNone(request)
        self.assertIn("conflicts", error)

    def test_template_controls_reach_exllamav3_tokenizer(self):
        tokenizer = FakeTokenizer()
        messages = BASE_REQUEST["messages"]
        template_kwargs = {
            "enable_thinking": False,
            "preserve_thinking": False,
        }

        rendered = serve_openai.render_chat_prompt(
            tokenizer, messages, None, template_kwargs
        )

        self.assertEqual(rendered, "rendered")
        self.assertEqual(tokenizer.call[0], messages)
        self.assertEqual(
            tokenizer.call[1],
            {
                "add_generation_prompt": True,
                "tools": None,
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
