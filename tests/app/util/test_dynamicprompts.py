from __future__ import annotations

import dynamicprompts.parser.parse as dynamicprompts_parse
import pytest
from dynamicprompts.generators import CombinatorialPromptGenerator
from pyparsing import ParseException

from invokeai.app.util import dynamicprompts as dynamicprompts_util
from invokeai.app.util.dynamicprompts import (
    MissingWildcardsError,
    find_missing_wildcards,
    find_missing_wildcards_in_command,
    generate_combinatorial_prompts,
)


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant() -> None:
    # Regression: `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it
    # sends the combinatorial generator into an infinite loop, so it must be reported up front.
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}") == ["random"]


def test_find_missing_wildcards_detects_unknown_wildcard_nested_in_sequence_in_variant() -> None:
    # The wildcard hangs the generator even when wrapped in other text inside the variant value.
    assert find_missing_wildcards("{a __nope__|b}") == ["nope"]


@pytest.mark.parametrize("prompt", ["a __nope__ b", "__nope__", "a photo, __my_style__"])
def test_find_missing_wildcards_ignores_wildcards_outside_variants(prompt: str) -> None:
    # A wildcard used as plain literal text generates fine (no hang), so it must not be reported.
    assert find_missing_wildcards(prompt) == []


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards() -> None:
    assert find_missing_wildcards("{__nope__|a} {__nope__|b} {__other__|c}") == ["nope", "other"]


@pytest.mark.parametrize(
    "prompt",
    [
        "{__nope__|a}",
        "{a __nope__|b}",
        "a __nope__ b",
        "{a|b|c}",
        "{__nope__|a} {__nope__|b} {__other__|c}",
    ],
)
def test_find_missing_wildcards_in_command_matches_string_form(prompt: str) -> None:
    assert find_missing_wildcards_in_command(dynamicprompts_parse.parse(prompt)) == find_missing_wildcards(prompt)


EXPANSION_PROMPTS = [
    "plain text",
    "a {b|c} d",
    "{a|an} {{cute|fierce} {cat|dog}|{large|small} bird} on {a|the} {sofa|rock}",
    "{3::big|small} {cat|dog}",
    "{2$$ and $$a|b|c}",
    "${style=oil} ${style} painting of {a cat|a dog}",
    # A bare unknown wildcard is not guarded: the generator yields it back as literal text.
    "a __nope__ {b|c}",
    "   ",
]


@pytest.mark.parametrize("prompt", EXPANSION_PROMPTS)
def test_generate_combinatorial_prompts_matches_generator(prompt: str) -> None:
    expected = CombinatorialPromptGenerator().generate(prompt, max_prompts=100)
    assert generate_combinatorial_prompts(prompt, 100) == expected


def test_generate_combinatorial_prompts_respects_max_prompts() -> None:
    prompts = generate_combinatorial_prompts("{a|b|c} {d|e}", 4)
    assert len(prompts) == 4
    assert prompts == CombinatorialPromptGenerator().generate("{a|b|c} {d|e}", max_prompts=4)


def test_generate_combinatorial_prompts_empty_prompt_yields_nothing() -> None:
    assert generate_combinatorial_prompts("", 10) == []
    assert CombinatorialPromptGenerator().generate("", max_prompts=10) == []


def test_generate_combinatorial_prompts_rejects_unknown_wildcard_in_variant() -> None:
    with pytest.raises(MissingWildcardsError, match=r"^No values found for wildcard\(s\): nope, other$") as excinfo:
        generate_combinatorial_prompts("{__nope__|a} {__nope__|b} {__other__|c}", 10)
    assert excinfo.value.wildcards == ["nope", "other"]
    # Callers that previously raised/handled a plain ValueError keep working.
    assert isinstance(excinfo.value, ValueError)


def test_generate_combinatorial_prompts_raises_parse_exception_for_malformed_prompt() -> None:
    with pytest.raises(ParseException):
        generate_combinatorial_prompts("{a|b}{", 10)


def test_generate_combinatorial_prompts_parses_prompt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # The unknown-wildcard guard and the generator used to parse the prompt independently. Parsing
    # dominates the cost of an expansion, so the prompt must be parsed exactly once, whether it is
    # accepted or rejected.
    real_parse = dynamicprompts_parse.parse
    parsed: list[str] = []

    def counting_parse(prompt: str, *args, **kwargs):
        parsed.append(prompt)
        return real_parse(prompt, *args, **kwargs)

    monkeypatch.setattr(dynamicprompts_parse, "parse", counting_parse)  # what the generator would call
    monkeypatch.setattr(dynamicprompts_util, "parse", counting_parse)  # what the guard calls

    generate_combinatorial_prompts("{a|b} {c|d}", 10)
    assert parsed == ["{a|b} {c|d}"]

    parsed.clear()
    with pytest.raises(MissingWildcardsError):
        generate_combinatorial_prompts("{__nope__|a}", 10)
    assert parsed == ["{__nope__|a}"]
