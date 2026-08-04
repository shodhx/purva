"""Plain-assert tests for the JSON extraction/validation pipeline.

Run with: python -m purva.committee.test_parse
"""

from __future__ import annotations

from purva.committee.run_judge import extract_first_json, try_parse

CLEAN_OBJECT = """{
  "subjectivity": "objective",
  "polarity": null,
  "confidence": 0.9,
  "domain": "news",
  "narrative_voice": "third_person",
  "sentiment_target": null,
  "rationale": "Reports a fact."
}"""

SECOND_OBJECT = """{
  "subjectivity": "subjective",
  "polarity": "positive",
  "confidence": 0.7,
  "domain": "festival",
  "narrative_voice": "third_person",
  "sentiment_target": null,
  "rationale": "Celebrates a happy occasion."
}"""


def test_single_clean_object():
    parsed = try_parse(CLEAN_OBJECT)
    assert parsed is not None, "clean object should parse"
    assert parsed["subjectivity"] == "objective"
    assert parsed["polarity"] is None
    assert parsed["rationale"] == "Reports a fact."


def test_object_followed_by_pipe_and_second_object():
    text = CLEAN_OBJECT + " | " + SECOND_OBJECT
    parsed = try_parse(text)
    assert parsed is not None, "should parse the first object and ignore the rest"
    assert parsed["subjectivity"] == "objective"
    assert parsed["rationale"] == "Reports a fact."


def test_object_followed_by_note_prose():
    text = CLEAN_OBJECT + "\n\nNote: I hope this helps! Let me know if you need anything else."
    parsed = try_parse(text)
    assert parsed is not None, "should parse the object and ignore trailing prose"
    assert parsed["domain"] == "news"


def test_fenced_json():
    text = "```json\n" + CLEAN_OBJECT + "\n```"
    parsed = try_parse(text)
    assert parsed is not None, "fenced JSON should still parse"
    assert parsed["subjectivity"] == "objective"


def test_truncated_json_fails():
    truncated = '{"subjectivity": "objective", "polarity": null, "confidence": 0.9, "domain": "news"'
    assert extract_first_json(truncated) is None, "unbalanced/truncated JSON must not extract"
    assert try_parse(truncated) is None, "unbalanced/truncated JSON must fail to parse"


def test_braces_inside_rationale_string():
    text = """{
  "subjectivity": "subjective",
  "polarity": "negative",
  "confidence": 0.8,
  "domain": "commentary",
  "narrative_voice": "first_person",
  "sentiment_target": null,
  "rationale": "The speaker says '{this is odd}' sarcastically, implying disapproval."
}"""
    parsed = try_parse(text)
    assert parsed is not None, "braces inside a string literal must not break balance tracking"
    assert parsed["rationale"] == "The speaker says '{this is odd}' sarcastically, implying disapproval."


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
