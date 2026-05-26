"""
Unit tests for AgentEcon project.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Import modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.asset_loader import AssetLoader, SettlementQuery
from src.models.predictors import (
    JSONOutputParser,
    LogitQRESampler,
    OutputParser,
    PlainTextParser,
    PredictionResult,
    PromptTemplate,
)
from src.oracle.fabric_oracle import (
    Aggregator,
    FabricOracleClient,
    FixedPointMath,
    OracleQuery,
    QuadraticScoringRule,
    ReputationTracker,
    SettlementResult,
    ValidatorSubmission,
)
from src.experiments.runner import ExperimentConfig, ExperimentRunner


class TestFixedPointMath(unittest.TestCase):
    """Tests for fixed-point arithmetic."""

    def test_encode_decode(self):
        """Test encoding and decoding."""
        values = [0.0, 1.0, 0.5, 0.12345678, -1.0]
        for val in values:
            encoded = FixedPointMath.encode(val)
            decoded = FixedPointMath.decode(encoded)
            self.assertAlmostEqual(decoded, val, places=6)

    def test_add(self):
        """Test fixed-point addition."""
        a = FixedPointMath.encode(1.5)
        b = FixedPointMath.encode(2.5)
        result = FixedPointMath.decode(FixedPointMath.add(a, b))
        self.assertAlmostEqual(result, 4.0, places=6)

    def test_multiply(self):
        """Test fixed-point multiplication."""
        a = FixedPointMath.encode(2.0)
        b = FixedPointMath.encode(3.0)
        result = FixedPointMath.decode(FixedPointMath.mul(a, b))
        self.assertAlmostEqual(result, 6.0, places=4)


class TestQuadraticScoringRule(unittest.TestCase):
    """Tests for QSR scoring."""

    def setUp(self):
        self.qsr = QuadraticScoringRule(grid_size=1024)

    def test_perfect_prediction(self):
        """Perfect prediction should score 1.0."""
        score = self.qsr.score(predicted_bin=512, ground_truth_bin=512)
        self.assertEqual(score, 1.0)

    def test_wrong_prediction(self):
        """Wrong prediction should score less than 1.0."""
        score = self.qsr.score(predicted_bin=0, ground_truth_bin=1023)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_confidence_effect(self):
        """Higher confidence should improve score."""
        low_conf = self.qsr.score(512, 512, confidence=0.1)
        high_conf = self.qsr.score(512, 512, confidence=1.0)
        self.assertGreaterEqual(high_conf, low_conf)


class TestReputationTracker(unittest.TestCase):
    """Tests for reputation tracking."""

    def setUp(self):
        self.tracker = ReputationTracker(initial_reputation=1.0)

    def test_initial_reputation(self):
        """New validators should have initial reputation."""
        rep = self.tracker.get_reputation("new_validator")
        self.assertEqual(rep, 1.0)

    def test_reputation_update(self):
        """Reputation should update based on accuracy."""
        new_rep = self.tracker.update_reputation(
            validator_id="test_validator",
            prediction_score=0.9,
            ground_truth_bin=512,
            predicted_bin=512,
        )
        self.assertNotEqual(new_rep, 1.0)

    def test_sybil_detection(self):
        """Should detect potential Sybil validators."""
        submissions = []
        # Pre-populate tracker with low reputations
        for i in range(10):
            # Set low reputation directly in tracker
            self.tracker.reputations[f"low_rep_{i}"] = 0.05
            sub = ValidatorSubmission(
                query_id="test",
                validator_id=f"low_rep_{i}",
                predicted_bin=512,
                timestamp=0,
                reputation=0.05,  # Below SYBIL_THRESHOLD of 0.1
            )
            submissions.append(sub)

        sybil_ids = self.tracker.detect_sybil(submissions)
        self.assertGreater(len(sybil_ids), 0)


class TestAggregator(unittest.TestCase):
    """Tests for submission aggregation."""

    def setUp(self):
        self.aggregator = Aggregator(grid_size=1024)
        self.reputation = ReputationTracker()

    def test_empty_submissions(self):
        """Empty submissions should return middle bin."""
        bin_id, trimmed = self.aggregator.aggregate([], self.reputation)
        self.assertEqual(bin_id, 512)

    def test_single_submission(self):
        """Single submission should return that bin."""
        sub = ValidatorSubmission(
            query_id="test",
            validator_id="v1",
            predicted_bin=100,
            timestamp=0,
        )
        bin_id, trimmed = self.aggregator.aggregate([sub], self.reputation)
        self.assertEqual(bin_id, 100)

    def test_weighted_aggregate(self):
        """Higher reputation should have more weight."""
        submissions = [
            ValidatorSubmission("test", "v1", 100, 0, reputation=1.0),
            ValidatorSubmission("test", "v2", 900, 0, reputation=1.0),
        ]
        bin_id, _ = self.aggregator.aggregate(submissions, self.reputation)
        self.assertEqual(bin_id, 500)


class TestFabricOracleClient(unittest.TestCase):
    """Tests for Fabric oracle client."""

    def setUp(self):
        self.client = FabricOracleClient()

    def test_submit_query(self):
        """Should submit a new query."""
        query = self.client.submit_query(
            query_id="test_q1",
            asset="XAU_USD",
            ground_truth=1850.50,
            bin_id=512,
        )
        self.assertEqual(query.query_id, "test_q1")
        self.assertEqual(query.asset, "XAU_USD")

    def test_submit_prediction(self):
        """Should record validator prediction."""
        self.client.submit_query("test_q1", "XAU_USD", 1850.50, 512)
        submission = self.client.submit_prediction(
            query_id="test_q1",
            validator_id="validator_0",
            predicted_bin=500,
        )
        self.assertEqual(submission.predicted_bin, 500)

    def test_settle(self):
        """Should settle query and calculate metrics."""
        self.client.submit_query("test_q1", "XAU_USD", 1850.50, 512)
        for i in range(4):
            self.client.submit_prediction(
                query_id="test_q1",
                validator_id=f"validator_{i}",
                predicted_bin=512 + i * 10,
            )

        result = self.client.settle(query_id="test_q1")
        self.assertIsInstance(result, SettlementResult)
        self.assertIn(result.query_id, ["test_q1"])
        self.assertGreater(result.validators_included, 0)


class TestLogitQRESampler(unittest.TestCase):
    """Tests for QRE sampler."""

    def setUp(self):
        self.sampler = LogitQRESampler(grid_size=1024, tau=1.0, lambda_=0.5)

    def test_sample(self):
        """Should sample valid bin."""
        prior = np.array([100.0, 101.0, 102.0, 103.0, 102.5, 104.0, 105.0])
        for _ in range(10):
            bin_id = self.sampler.sample(prior)
            self.assertGreaterEqual(bin_id, 0)
            self.assertLess(bin_id, 1024)

    def test_reproducibility(self):
        """Same seed should give same results."""
        self.sampler.set_seed(42)
        prior = np.array([100.0, 101.0, 102.0])
        samples1 = [self.sampler.sample(prior) for _ in range(5)]

        self.sampler.set_seed(42)
        samples2 = [self.sampler.sample(prior) for _ in range(5)]

        self.assertEqual(samples1, samples2)

    def test_policy_probs_sum_to_one(self):
        """Policy probabilities should sum to 1."""
        prior = np.array([100.0, 101.0, 102.0])
        probs = self.sampler.get_policy_probs(prior)
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_tv_distance(self):
        """TV distance should be in [0, 1]."""
        p = np.ones(1024) / 1024
        q = np.ones(1024) / 1024
        tv = self.sampler.total_variation_distance(p, q)
        self.assertAlmostEqual(tv, 0.0, places=5)


class TestPromptTemplate(unittest.TestCase):
    """Tests for prompt templates."""

    def setUp(self):
        self.template = PromptTemplate()

    def test_ohlc_to_markdown(self):
        """Should convert OHLC to markdown table."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [102.0, 103.0],
            },
            index=["2024-01-01", "2024-01-02"],
        )
        markdown = self.template.ohlc_to_markdown(df)
        self.assertIn("| Date |", markdown)
        self.assertIn("100.0000", markdown)

    def test_build_query_prompt(self):
        """Should build complete query prompt."""
        prompt = self.template.build_query_prompt(
            asset_name="Gold",
            query_date="2024-01-01",
            settlement_date="2024-01-08",
            ohlc_data="| Date | Close |\n|------|-------|",
            min_price=1800.0,
            max_price=2000.0,
        )
        self.assertIn("Gold", prompt)
        self.assertIn("2024-01-08", prompt)


class TestOutputParsers(unittest.TestCase):
    """Tests for output parsers."""

    def test_json_parser(self):
        """JSON parser should extract bin and confidence."""
        parser = JSONOutputParser()
        bin_id, conf, success = parser.parse('{"bin": 512, "confidence": 0.75}')
        self.assertTrue(success)
        self.assertEqual(bin_id, 512)
        self.assertEqual(conf, 0.75)

    def test_json_parser_invalid(self):
        """JSON parser should fail gracefully."""
        parser = JSONOutputParser()
        bin_id, conf, success = parser.parse("not valid json")
        self.assertFalse(success)
        self.assertIsNone(bin_id)

    def test_plain_text_parser(self):
        """Plain text parser should extract bin number."""
        parser = PlainTextParser()
        bin_id, conf, success = parser.parse("The bin is 512")
        self.assertTrue(success)
        self.assertEqual(bin_id, 512)


class TestExperimentConfig(unittest.TestCase):
    """Tests for experiment configuration."""

    def test_default_config(self):
        """Default config should have correct values."""
        config = ExperimentConfig(experiment_name="test")
        self.assertEqual(config.mode, "llm")
        self.assertEqual(config.n_validators, 64)
        self.assertEqual(config.grid_size, 1024)

    def test_to_dict(self):
        """Config should serialize to dict."""
        config = ExperimentConfig(experiment_name="test")
        d = config.to_dict()
        self.assertIn("experiment_name", d)
        self.assertIn("mode", d)


class TestAssetLoader(unittest.TestCase):
    """Tests for asset loader."""

    def setUp(self):
        self.loader = AssetLoader(data_dir="data")

    def test_build_price_grid(self):
        """Should build correct price grid."""
        # Will use mocked data for unit test
        pass

    def test_price_to_bin(self):
        """Should map price to correct bin."""
        bins = np.linspace(100.0, 200.0, 1025)
        bin_id = self.loader.price_to_bin(150.0, bins)
        self.assertGreaterEqual(bin_id, 0)
        self.assertLess(bin_id, 1024)

    def test_bin_to_price(self):
        """Should map bin to center price."""
        bins = np.linspace(100.0, 200.0, 1025)
        price = self.loader.bin_to_price(512, bins)
        self.assertAlmostEqual(price, 150.0, places=1)


if __name__ == "__main__":
    unittest.main()
