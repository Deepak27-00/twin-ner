"""Tests for entity parsing and character-offset BILOU alignment."""

from __future__ import annotations

import pytest

from twinner.data import (
    Example,
    TwinNERDataset,
    assign_bilou,
    find_span,
    parse_entity_string,
)
from twinner.schema import IGNORE_INDEX, LabelScheme

NULLS = {"not present", "none"}


class TestParseEntityString:
    def test_parses_both_entities(self, entity_map):
        parsed = parse_entity_string("sensorType: motion; digitalTwinID:Cam004", entity_map)
        assert parsed == {"SENSOR": "motion", "TWIN": "Cam004"}

    def test_is_case_and_whitespace_insensitive(self, entity_map):
        parsed = parse_entity_string(
            "  SENSORTYPE : Temperature ;  digitaltwinid:  Building001 ", entity_map
        )
        assert parsed == {"SENSOR": "Temperature", "TWIN": "Building001"}

    def test_ignores_unknown_keys_and_junk(self, entity_map):
        parsed = parse_entity_string("colour: red; sensorType: smoke; garbage", entity_map)
        assert parsed == {"SENSOR": "smoke"}

    @pytest.mark.parametrize("value", ["", None, float("nan"), 42])
    def test_survives_non_string_input(self, value, entity_map):
        assert parse_entity_string(value, entity_map) == {}


class TestFindSpan:
    def test_finds_exact_span(self):
        assert find_span("Fetch motion data from Cam004.", "Cam004", NULLS) == (23, 29)

    def test_matching_ignores_case(self):
        assert find_span("Fetch motion data from Cam004.", "cam004", NULLS) == (23, 29)

    def test_prefers_word_boundary_match(self):
        # "cam001" appears inside "webcam001x" first, but the standalone
        # occurrence later in the string is the correct one.
        text = "webcam001x reports for Cam001 now"
        assert find_span(text, "cam001", NULLS) == (23, 29)

    def test_returns_none_for_null_markers(self):
        assert find_span("anything", "not present", NULLS) is None

    def test_returns_none_when_absent(self):
        assert find_span("no identifier here", "Building009", NULLS) is None


class TestAssignBilou:
    scheme = LabelScheme(["SENSOR", "TWIN"], ["A"])

    def test_single_token_entity_gets_unit_tag(self):
        offsets = [(0, 0), (0, 6), (7, 11), (0, 0)]
        special = [1, 0, 0, 1]
        tags = assign_bilou(offsets, special, {"SENSOR": (0, 6)}, self.scheme)
        assert tags == [None, "U-SENSOR", "O", None]

    def test_multi_token_entity_gets_begin_inside_last(self):
        offsets = [(0, 0), (0, 3), (3, 5), (5, 6), (0, 0)]
        special = [1, 0, 0, 0, 1]
        tags = assign_bilou(offsets, special, {"TWIN": (0, 6)}, self.scheme)
        assert tags == [None, "B-TWIN", "I-TWIN", "L-TWIN", None]

    def test_special_tokens_are_never_labelled(self):
        offsets = [(0, 0), (0, 4), (0, 0), (0, 0)]
        special = [1, 0, 1, 1]
        tags = assign_bilou(offsets, special, {}, self.scheme)
        assert tags[0] is None and tags[2] is None and tags[3] is None

    def test_span_matching_no_tokens_is_dropped_not_crashed(self):
        offsets = [(0, 0), (0, 4), (0, 0)]
        special = [1, 0, 1]
        assert assign_bilou(offsets, special, {"TWIN": (50, 60)}, self.scheme) == [
            None,
            "O",
            None,
        ]


class TestDatasetEncoding:
    """End-to-end checks against a real tokenizer."""

    def _dataset(self, tokenizer, examples):
        scheme = LabelScheme(["SENSOR", "TWIN"], ["GET_LATEST_DATA", "LIST_SENSORS"])
        return TwinNERDataset(examples, tokenizer, scheme, 32, list(NULLS)), scheme

    def test_multi_wordpiece_id_is_one_span(self, tokenizer):
        example = Example(
            "Fetch the latest motion data from Cam004.",
            "GET_LATEST_DATA",
            {"SENSOR": "motion", "TWIN": "Cam004"},
        )
        dataset, scheme = self._dataset(tokenizer, [example])
        encoded = dataset[0]
        tags = [
            scheme.id_to_tag[int(i)]
            for i in encoded["labels"].tolist()
            if int(i) != IGNORE_INDEX
        ]
        # "Cam004" tokenises to cam/##00/##4 and must come back as one B-I-L span.
        assert "U-SENSOR" in tags
        assert tags.count("B-TWIN") == 1
        assert tags.count("L-TWIN") == 1

    def test_padding_is_ignored_by_the_loss(self, tokenizer):
        example = Example("List all smoke sensors.", "LIST_SENSORS", {"SENSOR": "smoke"})
        dataset, _ = self._dataset(tokenizer, [example])
        labels = dataset[0]["labels"].tolist()
        # Trailing positions are padding and must carry IGNORE_INDEX, not "O",
        # otherwise the model is rewarded for predicting "O" on empty positions.
        assert labels[-1] == IGNORE_INDEX
        assert labels[0] == IGNORE_INDEX  # [CLS]

    def test_absent_entity_produces_no_span(self, tokenizer):
        example = Example(
            "List all smoke sensors.",
            "LIST_SENSORS",
            {"SENSOR": "smoke", "TWIN": "not present"},
        )
        dataset, scheme = self._dataset(tokenizer, [example])
        tags = [
            scheme.id_to_tag[int(i)]
            for i in dataset[0]["labels"].tolist()
            if int(i) != IGNORE_INDEX
        ]
        assert not any(tag.endswith("TWIN") for tag in tags)

    def test_sequence_lengths_match_max_length(self, tokenizer):
        example = Example("List all smoke sensors.", "LIST_SENSORS", {"SENSOR": "smoke"})
        dataset, _ = self._dataset(tokenizer, [example])
        encoded = dataset[0]
        assert encoded["input_ids"].shape[0] == 32
        assert encoded["labels"].shape[0] == 32
