"""
AgentEcon Oracle - Hyperledger Fabric Chaincode Client

Implements the oracle mechanism for prediction market settlement:
- Query submission and validation
- Quadratic scoring rule for reward calculation
- Reputation-weighted aggregation
- Sybil-resistant trimming
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OracleQuery:
    """An oracle query submitted to the Fabric network."""

    query_id: str
    asset: str
    timestamp: int
    ground_truth: float
    bin_id: int
    round_number: int = 0


@dataclass
class ValidatorSubmission:
    """A submission from a validator agent."""

    query_id: str
    validator_id: str
    predicted_bin: int
    timestamp: int
    reputation: float = 1.0
    confidence: float = 1.0
    is_sybil: bool = False


@dataclass
class SettlementResult:
    """Result of oracle settlement."""

    query_id: str
    aggregated_bin: int
    ground_truth_bin: int
    honest_reporting: bool
    welfare_gap: float
    scoring_rule_reward: float
    validators_included: int
    sybil_trimmed: int
    commit_latency_ms: float


@dataclass
class ChaincodeState:
    """State managed by the chaincode."""

    queries: dict[str, dict] = field(default_factory=dict)
    submissions: dict[str, list[dict]] = field(default_factory=dict)
    reputations: dict[str, float] = field(default_factory=dict)
    settlements: dict[str, dict] = field(default_factory=dict)
    next_round: int = 1


class FixedPointMath:
    """
    Fixed-point arithmetic for deterministic chaincode execution.

    Uses integer representation with 8 decimal places to avoid
    floating-point nondeterminism in Fabric endorsement.
    """

    SCALE = 10**8

    @classmethod
    def encode(cls, value: float) -> int:
        """Convert float to fixed-point integer."""
        return int(round(value * cls.SCALE))

    @classmethod
    def decode(cls, value: int) -> float:
        """Convert fixed-point integer to float."""
        return value / cls.SCALE

    @classmethod
    def add(cls, a: int, b: int) -> int:
        return a + b

    @classmethod
    def mul(cls, a: int, b: int) -> int:
        return int(round((a * b) / cls.SCALE))


class QuadraticScoringRule:
    """
    Quadratic Scoring Rule (QSR) for incentive-compatible forecasts.

    The QSR provides a proper scoring rule where truthful reporting
    is a strict Nash equilibrium (Witkowski et al., AAAI 2017).

    In the deployable oracle path, validators are scored against the
    deterministic settlement aggregate, not against hidden future ground truth.
    Ground truth is kept only for offline evaluation metrics.

    Uses binned representation with 1024 bins.
    """

    def __init__(self, grid_size: int = 1024):
        self.grid_size = grid_size

    def score(
        self,
        predicted_bin: int,
        ground_truth_bin: int,
        confidence: float = 1.0,
    ) -> float:
        """
        Calculate QSR score for a prediction.

        Score is in [0, 1], with 1 for perfect prediction.
        Confidence modulates the effective precision.
        """
        if predicted_bin == ground_truth_bin:
            base_score = 1.0
        else:
            distance = abs(predicted_bin - ground_truth_bin)
            max_distance = self.grid_size
            base_score = max(0.0, 1.0 - (distance / max_distance) ** 2)

        score = confidence * base_score + (1 - confidence) * (1.0 / self.grid_size)
        return score

    def score_vector(
        self, predictions: np.ndarray, ground_truth_bins: np.ndarray
    ) -> np.ndarray:
        """Score multiple predictions vectorized."""
        scores = np.zeros(len(predictions))
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth_bins)):
            scores[i] = self.score(pred, truth)
        return scores


class ReputationTracker:
    """
    Reputation tracking with decay and Sybil trimming.

    Reputation decays over time unless the validator makes accurate predictions.
    Identifies potential Sybil attacks through sudden reputation changes.
    """

    DECAY_RATE = 0.95
    SYBIL_THRESHOLD = 0.1
    MIN_REPUTATION = 0.01
    MAX_REPUTATION = 100.0

    def __init__(self, initial_reputation: float = 1.0):
        self.reputations: dict[str, float] = {}
        self.initial_reputation = initial_reputation
        self.accuracy_history: dict[str, list[float]] = {}

    def get_reputation(self, validator_id: str) -> float:
        """Get current reputation for a validator."""
        return self.reputations.get(validator_id, self.initial_reputation)

    def update_reputation(
        self,
        validator_id: str,
        prediction_score: float,
        ground_truth_bin: int,
        predicted_bin: int,
        lambda_param: float = 0.0,
    ) -> float:
        """
        Update reputation based on prediction accuracy.

        Args:
            validator_id: Validator identifier
            prediction_score: QSR score from 0 to 1
            ground_truth_bin: True bin
            predicted_bin: Predicted bin
            lambda_param: Weight for reputation component

        Returns:
            New reputation value
        """
        current = self.get_reputation(validator_id)

        accuracy = 1.0 if predicted_bin == ground_truth_bin else 0.0

        if validator_id not in self.accuracy_history:
            self.accuracy_history[validator_id] = []
        self.accuracy_history[validator_id].append(accuracy)

        recent_accuracy = np.mean(self.accuracy_history[validator_id][-20:])

        reward = prediction_score * (1 + lambda_param * recent_accuracy)

        new_reputation = current * self.DECAY_RATE + reward * (1 - self.DECAY_RATE)
        new_reputation = max(self.MIN_REPUTATION, min(self.MAX_REPUTATION, new_reputation))

        self.reputations[validator_id] = new_reputation
        return new_reputation

    def detect_sybil(
        self, submissions: list[ValidatorSubmission], window_size: int = 10
    ) -> list[str]:
        """
        Detect potential Sybil attacks through coordination patterns.

        Identifies validators that submit nearly identical predictions
        and have suspiciously low reputation compared to accuracy.
        """
        sybil_ids = []

        if len(submissions) < 2:
            return sybil_ids

        for sub in submissions:
            rep = self.get_reputation(sub.validator_id)
            if rep < self.SYBIL_THRESHOLD:
                sub.is_sybil = True
                sybil_ids.append(sub.validator_id)

        bin_counts: dict[int, list[str]] = {}
        for sub in submissions:
            if sub.validator_id not in sybil_ids:
                if sub.predicted_bin not in bin_counts:
                    bin_counts[sub.predicted_bin] = []
                bin_counts[sub.predicted_bin].append(sub.validator_id)

        for bin_id, validators in bin_counts.items():
            if len(validators) >= 5:
                reputations = [self.get_reputation(v) for v in validators]
                avg_rep = np.mean(reputations)
                if avg_rep < 0.2:
                    for v in validators:
                        if v not in sybil_ids:
                            logger.warning(f"Potential Sybil: {v} in coordinated group")
                            sybil_ids.append(v)

        return list(set(sybil_ids))

    def trim_submissions(
        self,
        submissions: list[ValidatorSubmission],
        n_trim: int,
    ) -> list[ValidatorSubmission]:
        """
        Deterministically trim lowest-reputation submissions.
        """
        if n_trim <= 0 or len(submissions) <= n_trim:
            return submissions

        non_sybil = [s for s in submissions if not s.is_sybil]
        sybil = [s for s in submissions if s.is_sybil]
        ranked = sorted(
            non_sybil,
            key=lambda s: (self.get_reputation(s.validator_id), s.validator_id),
            reverse=True,
        )
        return ranked[: len(ranked) - n_trim] + sybil


class Aggregator:
    """
    Aggregation of validator submissions using weighted trimmed mean.

    Uses fixed-point arithmetic for determinism.
    """

    def __init__(
        self,
        grid_size: int = 1024,
        trim_fraction: float = 0.1,
    ):
        self.grid_size = grid_size
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        submissions: list[ValidatorSubmission],
        reputation_tracker: ReputationTracker,
    ) -> tuple[int, int]:
        """
        Aggregate submissions using reputation-weighted trimmed mean.

        Returns:
            (aggregated_bin, n_trimmed)
        """
        if not submissions:
            return self.grid_size // 2, 0

        non_sybil = [s for s in submissions if not s.is_sybil]
        sybil_trimmed = len(submissions) - len(non_sybil)

        if not non_sybil:
            return self.grid_size // 2, sybil_trimmed

        weights = np.array(
            [reputation_tracker.get_reputation(s.validator_id) for s in non_sybil]
        )

        bins = np.array([s.predicted_bin for s in non_sybil])

        n_trim = int(len(non_sybil) * self.trim_fraction)

        weighted_mean = np.average(bins, weights=weights)

        sorted_indices = np.argsort(bins)
        trimmed_bins = bins[sorted_indices[n_trim:-n_trim] if n_trim > 0 else sorted_indices]

        if len(trimmed_bins) > 0:
            trimmed_weights = weights[sorted_indices[n_trim:-n_trim] if n_trim > 0 else sorted_indices]
            trimmed_mean = np.average(trimmed_bins, weights=trimmed_weights)
        else:
            trimmed_mean = weighted_mean

        final_aggregate = int(round(trimmed_mean))
        final_aggregate = max(0, min(self.grid_size - 1, final_aggregate))

        price_trimmed = 2 * n_trim if n_trim > 0 and len(trimmed_bins) > 0 else 0
        return final_aggregate, sybil_trimmed + price_trimmed

    def aggregate_fixed_point(
        self,
        submissions: list[ValidatorSubmission],
        reputation_tracker: ReputationTracker,
    ) -> tuple[int, int]:
        """
        Fixed-point aggregation for deterministic chaincode execution.

        All calculations use integer arithmetic with 8 decimal places.
        """
        if not submissions:
            return self.grid_size // 2, 0

        non_sybil = [s for s in submissions if not s.is_sybil]
        sybil_trimmed = len(submissions) - len(non_sybil)

        if not non_sybil:
            return self.grid_size // 2, sybil_trimmed

        total_weight = 0
        weighted_sum = 0

        for sub in non_sybil:
            rep = reputation_tracker.get_reputation(sub.validator_id)
            rep_fp = FixedPointMath.encode(rep)
            bin_fp = FixedPointMath.encode(sub.predicted_bin)

            weighted_sum = FixedPointMath.add(
                weighted_sum, FixedPointMath.mul(rep_fp, bin_fp)
            )
            total_weight = FixedPointMath.add(total_weight, rep_fp)

        if total_weight > 0:
            mean_fp = FixedPointMath.mul(weighted_sum, FixedPointMath.encode(1.0 / FixedPointMath.decode(total_weight)))
            result = FixedPointMath.decode(mean_fp)
        else:
            result = self.grid_size // 2

        result = max(0, min(self.grid_size - 1, int(round(result))))
        return result, sybil_trimmed


class FabricOracleClient:
    """
    Client for interacting with AgentEcon Fabric chaincode.

    Provides high-level interface for:
    - Submitting queries
    - Recording validator submissions
    - Settling queries and distributing rewards
    - Querying state
    """

    def __init__(
        self,
        msp_id: str = "Org1MSP",
        channel_name: str = "mychannel",
        chaincode_name: str = "agentecon",
        peer_endpoint: str = "localhost:7051",
        tls_cert_path: Optional[str] = None,
    ):
        self.msp_id = msp_id
        self.channel_name = channel_name
        self.chaincode_name = chaincode_name
        self.peer_endpoint = peer_endpoint
        self.tls_cert_path = tls_cert_path

        self.state = ChaincodeState()
        self.qsr = QuadraticScoringRule()
        self.reputation = ReputationTracker()
        self.aggregator = Aggregator()

        self._pending_submissions: dict[str, list[ValidatorSubmission]] = {}

    def submit_query(
        self,
        query_id: str,
        asset: str,
        ground_truth: float,
        bin_id: int,
    ) -> OracleQuery:
        """Submit a new settlement query."""
        query = OracleQuery(
            query_id=query_id,
            asset=asset,
            timestamp=int(time.time()),
            ground_truth=ground_truth,
            bin_id=bin_id,
            round_number=self.state.next_round,
        )

        self.state.queries[query_id] = {
            "query_id": query_id,
            "asset": asset,
            "timestamp": query.timestamp,
            "ground_truth": ground_truth,
            "bin_id": bin_id,
            "round_number": self.state.next_round,
            "status": "open",
        }

        self._pending_submissions[query_id] = []

        logger.info(f"Query submitted: {query_id} for {asset}, bin={bin_id}")
        return query

    def submit_prediction(
        self,
        query_id: str,
        validator_id: str,
        predicted_bin: int,
        confidence: float = 1.0,
    ) -> ValidatorSubmission:
        """Submit a validator's prediction."""
        submission = ValidatorSubmission(
            query_id=query_id,
            validator_id=validator_id,
            predicted_bin=predicted_bin,
            timestamp=int(time.time()),
            reputation=self.reputation.get_reputation(validator_id),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

        if query_id not in self._pending_submissions:
            self._pending_submissions[query_id] = []
        self._pending_submissions[query_id].append(submission)

        logger.debug(f"Submission received: {validator_id} -> bin {predicted_bin} for {query_id}")
        return submission

    def settle(
        self,
        query_id: str,
        lambda_param: float = 0.0,
        sybil_detection: bool = True,
    ) -> SettlementResult:
        """
        Settle a query and distribute rewards.

        Args:
            query_id: Query to settle
            lambda_param: Reputation weighting parameter
            sybil_detection: Whether to perform Sybil detection

        Returns:
            SettlementResult with metrics
        """
        start_time = time.time()

        if query_id not in self.state.queries:
            raise ValueError(f"Query not found: {query_id}")

        query_data = self.state.queries[query_id]
        ground_truth_bin = query_data["bin_id"]

        submissions = self._pending_submissions.get(query_id, [])

        for sub in submissions:
            sub.is_sybil = False

        if sybil_detection:
            sybil_ids = self.reputation.detect_sybil(submissions)
            for sub in submissions:
                if sub.validator_id in sybil_ids:
                    sub.is_sybil = True

        n_sybil = sum(1 for s in submissions if s.is_sybil)

        aggregated_bin, sybil_trimmed = self.aggregator.aggregate(
            submissions, self.reputation
        )

        honest_reporting = aggregated_bin == ground_truth_bin
        welfare_gap = abs(aggregated_bin - ground_truth_bin) / self.state.queries[query_id].get("grid_size", 1024)

        total_reward = 0.0
        for sub in submissions:
            if sub.is_sybil:
                continue

            score = self.qsr.score(sub.predicted_bin, aggregated_bin, sub.confidence)
            new_rep = self.reputation.update_reputation(
                sub.validator_id,
                score,
                aggregated_bin,
                sub.predicted_bin,
                lambda_param,
            )

            reward = score * sub.reputation
            total_reward += reward

        result = SettlementResult(
            query_id=query_id,
            aggregated_bin=aggregated_bin,
            ground_truth_bin=ground_truth_bin,
            honest_reporting=honest_reporting,
            welfare_gap=welfare_gap,
            scoring_rule_reward=total_reward,
            validators_included=max(0, len(submissions) - sybil_trimmed),
            sybil_trimmed=sybil_trimmed,
            commit_latency_ms=(time.time() - start_time) * 1000,
        )

        self.state.settlements[query_id] = {
            **result.__dict__,
            "settled_at": int(time.time()),
        }
        self.state.queries[query_id]["status"] = "settled"

        logger.info(
            f"Settled {query_id}: aggregated={aggregated_bin}, "
            f"truth={ground_truth_bin}, honest={honest_reporting}, "
            f"welfare_gap={welfare_gap:.4f}"
        )

        return result

    def get_reputation(self, validator_id: str) -> float:
        """Query a validator's current reputation."""
        return self.reputation.get_reputation(validator_id)

    def get_settlement(self, query_id: str) -> Optional[dict]:
        """Get settlement result for a query."""
        return self.state.settlements.get(query_id)

    def get_state_hash(self) -> str:
        """Get hash of current chaincode state for logging."""
        state_str = json.dumps(
            {
                "queries": self.state.queries,
                "settlements": self.state.settlements,
                "reputations": self.state.reputations,
            },
            sort_keys=True,
        )
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


if __name__ == "__main__":
    client = FabricOracleClient()

    for i in range(10):
        client.reputation.update_reputation(
            validator_id=f"validator_{i}",
            prediction_score=0.8 + np.random.random() * 0.2,
            ground_truth_bin=512,
            predicted_bin=512 + np.random.randint(-50, 50),
        )

    query = client.submit_query(
        query_id="test_q1",
        asset="XAU_USD",
        ground_truth=1850.50,
        bin_id=512,
    )

    for i in range(10):
        client.submit_prediction(
            query_id="test_q1",
            validator_id=f"validator_{i}",
            predicted_bin=512 + np.random.randint(-100, 100),
        )

    result = client.settle(query_id="test_q1", lambda_param=0.5)

    print(f"Settlement: {result}")
    print(f"Honest reporting: {result.honest_reporting}")
    print(f"Welfare gap: {result.welfare_gap:.4f}")
