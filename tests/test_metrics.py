"""Tests for BILOU span decoding and span-level scoring."""

from __future__ import annotations

from twinner.metrics import Span, decode_spans, evaluate_predictions
from twinner.schema import LabelScheme


class TestDecodeSpans:
    def test_decodes_unit_and_multi_token_spans(self):
        tags = ["O", "U-SENSOR", "O", "B-TWIN", "I-TWIN", "L-TWIN", "O"]
        assert decode_spans(tags) == [Span("SENSOR", 1, 2), Span("TWIN", 3, 6)]

    def test_all_outside_yields_nothing(self):
        assert decode_spans(["O", "O", "O"]) == []

    def test_orphan_inside_tag_still_produces_a_span(self):
        # Trained models emit malformed sequences; dropping them silently would
        # under-report recall and make the demo look broken.
        assert decode_spans(["I-TWIN", "L-TWIN"]) == [Span("TWIN", 0, 2)]

    def test_type_switch_mid_span_closes_the_previous_span(self):
        assert decode_spans(["B-TWIN", "I-SENSOR"]) == [
            Span("TWIN", 0, 1),
            Span("SENSOR", 1, 2),
        ]

    def test_unterminated_span_is_closed_at_the_end(self):
        assert decode_spans(["O", "B-SENSOR", "I-SENSOR"]) == [Span("SENSOR", 1, 3)]

    def test_adjacent_units_stay_separate(self):
        assert decode_spans(["U-SENSOR", "U-SENSOR"]) == [
            Span("SENSOR", 0, 1),
            Span("SENSOR", 1, 2),
        ]


class TestEvaluatePredictions:
    def test_perfect_prediction_scores_one(self):
        tags = [["O", "U-SENSOR"], ["B-TWIN", "L-TWIN"]]
        report = evaluate_predictions(tags, tags, ["A", "B"], ["A", "B"])
        assert report.entity_micro.f1 == 1.0
        assert report.intent_accuracy == 1.0
        assert report.exact_match == 1.0

    def test_missed_span_lowers_recall_not_precision(self):
        gold = [["U-SENSOR", "O"]]
        pred = [["O", "O"]]
        report = evaluate_predictions(gold, pred, ["A"], ["A"])
        assert report.entity_micro.recall == 0.0
        assert report.entity_micro.support == 1

    def test_boundary_error_counts_as_both_fp_and_fn(self):
        # A span with the right type but wrong extent is wrong twice over --
        # this is exactly what token-level accuracy would hide.
        gold = [["B-TWIN", "L-TWIN", "O"]]
        pred = [["B-TWIN", "I-TWIN", "L-TWIN"]]
        report = evaluate_predictions(gold, pred, ["A"], ["A"])
        assert report.entity_micro.precision == 0.0
        assert report.entity_micro.recall == 0.0

    def test_exact_match_requires_intent_and_entities(self):
        tags = [["U-SENSOR"]]
        report = evaluate_predictions(tags, tags, ["A"], ["B"])
        assert report.entity_micro.f1 == 1.0
        assert report.exact_match == 0.0

    def test_joint_score_averages_entity_f1_and_intent_accuracy(self):
        gold = [["U-SENSOR"], ["U-SENSOR"]]
        pred = [["U-SENSOR"], ["O"]]
        report = evaluate_predictions(gold, pred, ["A", "A"], ["A", "A"])
        expected = 0.5 * report.entity_micro.f1 + 0.5 * 1.0
        assert report.joint_score == expected


class TestLabelScheme:
    def test_builds_four_tags_per_entity_plus_outside(self):
        scheme = LabelScheme(["SENSOR", "TWIN"], ["A"])
        assert scheme.num_tags == 1 + 4 * 2
        assert scheme.tags[0] == "O"

    def test_round_trips_through_a_dict(self):
        scheme = LabelScheme(["TWIN"], ["B", "A"])
        restored = LabelScheme.from_dict(scheme.to_dict())
        assert restored.tags == scheme.tags
        assert restored.intents == scheme.intents

    def test_ordering_is_deterministic_regardless_of_input_order(self):
        assert (
            LabelScheme(["TWIN", "SENSOR"], []).tags
            == LabelScheme(["SENSOR", "TWIN"], []).tags
        )
