import pytest

from cod_ssl.data.camotion_attributes import (
    ATTRIBUTE_CODES,
    align_camotion_attributes,
    parse_camotion_attributes,
)

FIXTURE = """seq_a OC,SC
seq_b   MO, UE, SC
seq_c
seq_d OC,MB,SO
"""


def test_attribute_parser_retains_empty_sets_and_explicit_booleans():
    parsed = parse_camotion_attributes(FIXTURE.splitlines())
    assert tuple(parsed["seq_c"]) == ATTRIBUTE_CODES
    assert not any(parsed["seq_c"].values())
    assert parsed["seq_b"]["MO"] and parsed["seq_b"]["UE"] and parsed["seq_b"]["SC"]
    assert sum(parsed["seq_d"].values()) == 3


@pytest.mark.parametrize(
    "text,match",
    [("seq_a XX", "unknown"), ("seq_a OC\nseq_a MB", "duplicate")],
)
def test_attribute_parser_rejects_unknown_codes_and_duplicate_rows(text, match):
    with pytest.raises(ValueError, match=match):
        parse_camotion_attributes(text.splitlines())


def test_attribute_alignment_requires_exact_bijection_or_explicit_alias():
    parsed = parse_camotion_attributes(["metadata_a OC", "seq_b"])
    with pytest.raises(ValueError, match="mismatch"):
        align_camotion_attributes({"seq_a", "seq_b"}, parsed)
    aligned = align_camotion_attributes(
        {"seq_a", "seq_b"}, parsed, aliases={"metadata_a": "seq_a"}
    )
    assert aligned["seq_a"]["OC"]
    with pytest.raises(ValueError, match="extra_metadata"):
        align_camotion_attributes({"seq_a"}, parsed, aliases={"metadata_a": "seq_a"})
