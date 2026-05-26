"""
AgentEcon Statistical Analysis

Implements statistical rigor protocol from the experiment guide:
- Bootstrap paired CI (10,000 resamples)
- Holm-Bonferroni corrected paired t-tests
- Cohen's d effect sizes
- Two-way ANOVA for model × method interactions
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 10000
ALPHA = 0.05
HOLM_CORRECTION_N = 9


@dataclass
class StatisticalResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: Optional[float] = None
    significant: bool = False
    n_obs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "ci_lower": float(self.confidence_interval[0]),
            "ci_upper": float(self.confidence_interval[1]),
            "effect_size": float(self.effect_size) if self.effect_size else None,
            "significant": self.significant,
            "n_obs": self.n_obs,
        }


class BootstrapCI:
    """
    Bootstrap confidence interval computation.

    Uses paired bootstrap resampling for comparing paired observations
    (e.g., same queries with different methods).
    """

    def __init__(self, n_bootstrap: int = N_BOOTSTRAP, alpha: float = ALPHA):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

    def paired_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        statistic=np.mean,
    ) -> tuple[float, float, float]:
        """
        Compute paired bootstrap confidence interval.

        Args:
            x: First sample (e.g., LLM honest fraction)
            y: Second sample (e.g., QRE honest fraction)
            statistic: Function to compute statistic (default: mean difference)

        Returns:
            (statistic_value, ci_lower, ci_upper)
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        observed_stat = statistic(x - y)

        bootstrap_diffs = []
        n = len(x)

        rng = np.random.RandomState(42)

        for _ in range(self.n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            boot_diff = statistic((x - y)[indices])
            bootstrap_diffs.append(boot_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))

        return float(observed_stat), float(ci_lower), float(ci_upper)

    def paired_p_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alternative: str = "two-sided",
    ) -> float:
        """
        Compute paired t-test p-value.

        Args:
            x: First sample
            y: Second sample
            alternative: 'two-sided', 'greater', or 'less'

        Returns:
            p-value
        """
        diff = x - y
        t_stat, p_value = stats.ttest_1samp(diff, 0, alternative=alternative)
        return float(p_value)


class HolmBonferroniCorrection:
    """
    Holm-Bonferroni correction for multiple hypothesis testing.

    More powerful than standard Bonferroni correction while
    maintaining strong control of family-wise error rate.
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha

    def correct(
        self, p_values: list[float], test_names: Optional[list[str]] = None
    ) -> list[StatisticalResult]:
        """
        Apply Holm-Bonferroni correction.

        Args:
            p_values: List of uncorrected p-values
            test_names: Optional list of test names

        Returns:
            List of StatisticalResult with corrected p-values
        """
        if test_names is None:
            test_names = [f"test_{i}" for i in range(len(p_values))]

        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        sorted_names = np.array(test_names)[sorted_indices]

        results = []
        still_rejecting = True
        for rank, (p_val, name) in enumerate(zip(sorted_p, sorted_names), start=1):
            adjusted_p = min(p_val * (n - rank + 1), 1.0)
            significant = still_rejecting and p_val <= self.alpha / (n - rank + 1)
            if not significant:
                still_rejecting = False

            results.append(
                StatisticalResult(
                    test_name=name,
                    statistic=float(p_val),
                    p_value=float(adjusted_p),
                    confidence_interval=(np.nan, np.nan),
                    significant=significant,
                    n_obs=n,
                )
            )

        return results


class EffectSize:
    """Effect size computations for reporting."""

    @staticmethod
    def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Cohen's d for effect size.

        Cohen's d interpretation:
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8

        Args:
            x: First sample
            y: Second sample

        Returns:
            Cohen's d value
        """
        n1, n2 = len(x), len(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        d = (np.mean(x) - np.mean(y)) / pooled_std
        return float(d)

    @staticmethod
    def cohen_d_ci(
        x: np.ndarray, y: np.ndarray, n_bootstrap: int = 10000
    ) -> tuple[float, float, float]:
        """
        Compute bootstrap CI for Cohen's d.

        Returns:
            (d_observed, ci_lower, ci_upper)
        """
        d_obs = EffectSize.cohens_d(x, y)

        rng = np.random.RandomState(42)
        boot_d = []

        for _ in range(n_bootstrap):
            idx1 = rng.choice(len(x), size=len(x), replace=True)
            idx2 = rng.choice(len(y), size=len(y), replace=True)
            boot_d.append(EffectSize.cohens_d(x[idx1], y[idx2]))

        ci_lower = np.percentile(boot_d, 2.5)
        ci_upper = np.percentile(boot_d, 97.5)

        return float(d_obs), float(ci_lower), float(ci_upper)


class TwoWayANOVA:
    """
    Two-way ANOVA for model × method interaction analysis.

    Used for Figure 5 analysis (multi-LLM consistency).
    """

    def __init__(self):
        pass

    def analyze(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        factor1: str,
        factor2: str,
    ) -> dict[str, Any]:
        """
        Perform two-way ANOVA.

        Args:
            df: DataFrame with all observations
            dependent_var: Name of dependent variable column
            factor1: First factor (e.g., 'model')
            factor2: Second factor (e.g., 'method')

        Returns:
            dict with ANOVA results
        """
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            formula = f"{dependent_var} ~ C({factor1}) * C({factor2})"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            return {
                "anova_table": anova_table.to_dict(),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(anova_table.loc.get(f"C({factor1}):C({factor2})", {}).get("F", np.nan)),
                "interaction_p_value": float(anova_table.loc.get(f"C({factor1}):C({factor2})", {}).get("PR(>F)", np.nan)),
            }

        except Exception as e:
            logger.warning(f"ANOVA failed: {e}")
            return {"error": str(e)}


class StatisticalAnalyzer:
    """
    Main statistical analysis class.

    Coordinates all statistical tests and generates reports.
    """

    def __init__(
        self,
        data_dir: str = "data/runs",
        n_bootstrap: int = N_BOOTSTRAP,
        alpha: float = ALPHA,
    ):
        self.data_dir = Path(data_dir)
        self.bootstrap = BootstrapCI(n_bootstrap=n_bootstrap, alpha=alpha)
        self.holm = HolmBonferroniCorrection(alpha=alpha)
        self.effect = EffectSize()
        self.anova = TwoWayANOVA()

    def load_results(self, results_file: str) -> pd.DataFrame:
        """Load experiment results from CSV."""
        df = pd.read_csv(self.data_dir / results_file)
        logger.info(f"Loaded {len(df)} rows from {results_file}")
        return df

    def compare_methods(
        self,
        df: pd.DataFrame,
        metric: str = "honest_reporting_fraction",
        groupby: str = "lambda",
    ) -> list[StatisticalResult]:
        """
        Compare methods across different parameter settings.

        Args:
            df: DataFrame with experiment results
            metric: Metric to compare
            groupby: Column to group by

        Returns:
            List of StatisticalResult for each comparison
        """
        results = []

        method_col = "method" if "method" in df.columns else "model"

        for name, group in df.groupby(groupby):
            labels = group[method_col].astype(str).str.lower()
            llm_mask = labels.str.contains("llm") | labels.str.contains("llama") | labels.str.contains("mistral") | labels.str.contains("qwen")
            qre_mask = labels.str.contains("stylised") | labels.str.contains("qre")
            if llm_mask.any() and qre_mask.any():
                llm_vals = group[llm_mask][metric].values
                qre_vals = group[qre_mask][metric].values

                if len(llm_vals) > 0 and len(qre_vals) > 0:
                    n = min(len(llm_vals), len(qre_vals))
                    llm_vals = llm_vals[:n]
                    qre_vals = qre_vals[:n]
                    _, ci_lower, ci_upper = self.bootstrap.paired_ci(llm_vals, qre_vals)
                    p_val = self.bootstrap.paired_p_value(llm_vals, qre_vals)
                    d = self.effect.cohens_d(llm_vals, qre_vals)

                    results.append(
                        StatisticalResult(
                            test_name=f"{groupby}={name}",
                            statistic=float(np.mean(llm_vals) - np.mean(qre_vals)),
                            p_value=p_val,
                            confidence_interval=(ci_lower, ci_upper),
                            effect_size=d,
                            significant=p_val < ALPHA / HOLM_CORRECTION_N,
                            n_obs=len(llm_vals) + len(qre_vals),
                        )
                    )

        return results

    def analyze_figure3(
        self, df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze honest-reporting fraction (Figure 3).

        Variables: tau × lambda × honest_reporting_fraction
        """
        results = {
            "tau_effects": [],
            "lambda_effects": [],
            "interaction": {},
        }

        for tau in df["tau"].unique():
            tau_data = df[df["tau"] == tau][
                "honest_reporting_fraction"
            ].values
            results["tau_effects"].append(
                {
                    "tau": float(tau),
                    "mean": float(np.mean(tau_data)),
                    "std": float(np.std(tau_data)),
                    "ci": self._mean_ci(tau_data),
                }
            )

        for lam in df["lambda"].unique():
            lam_data = df[df["lambda"] == lam][
                "honest_reporting_fraction"
            ].values
            results["lambda_effects"].append(
                {
                    "lambda": float(lam),
                    "mean": float(np.mean(lam_data)),
                    "std": float(np.std(lam_data)),
                    "ci": self._mean_ci(lam_data),
                }
            )

        return results

    def analyze_figure4(
        self, df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze mixed-population welfare gap (Figure 4).

        Variables: lambda × welfare_gap
        """
        results = {
            "lambda_welfare": [],
            "cohens_d_by_lambda": [],
        }

        baseline = df[df["lambda"] == 0.0]["welfare_gap"].values

        for lam in sorted(df["lambda"].unique()):
            lam_data = df[df["lambda"] == lam]["welfare_gap"].values

            d = self.effect.cohens_d(lam_data, baseline) if len(baseline) > 0 else 0

            results["lambda_welfare"].append(
                {
                    "lambda": float(lam),
                    "mean_welfare_gap": float(np.mean(lam_data)),
                    "std_welfare_gap": float(np.std(lam_data)),
                    "cohens_d_vs_baseline": float(d),
                }
            )

        return results

    def analyze_figure5(
        self, df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Analyze multi-LLM consistency (Figure 5).

        Two-way ANOVA: model × method interaction.
        """
        df_copy = df.copy()
        df_copy["method"] = df_copy["model"].apply(
            lambda x: "LLM" if "llm" in str(x).lower() else "QRE"
        )

        anova_results = self.anova.analyze(
            df_copy,
            dependent_var="honest_reporting_fraction",
            factor1="model",
            factor2="method",
        )

        model_means = df_copy.groupby("model")["honest_reporting_fraction"].agg(
            ["mean", "std", "count"]
        )

        return {
            "anova": anova_results,
            "model_means": model_means.to_dict(),
        }

    def generate_report(
        self,
        df: pd.DataFrame,
        output_dir: str = "data/runs",
    ) -> dict[str, Any]:
        """
        Generate full statistical report.

        Args:
            df: DataFrame with experiment results
            output_dir: Directory to save report

        Returns:
            dict with complete analysis
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "descriptive": {
                "n_cells": len(df),
                "n_assets": df["asset"].nunique(),
                "n_models": df["model"].nunique(),
                "n_seeds": df["seed"].nunique(),
                "effect_size_target": "Cohen's d >= 0.5",
            },
            "honest_reporting": {
                "overall_mean": float(df["honest_reporting_fraction"].mean()),
                "overall_std": float(df["honest_reporting_fraction"].std()),
                "by_tau": df.groupby("tau")["honest_reporting_fraction"].mean().to_dict(),
                "by_lambda": df.groupby("lambda")["honest_reporting_fraction"].mean().to_dict(),
            },
            "welfare_gap": {
                "overall_mean": float(df["welfare_gap"].mean()),
                "overall_std": float(df["welfare_gap"].std()),
            },
            "commit_latency": {
                "overall_mean_ms": float(df["avg_commit_latency_ms"].mean()),
                "overall_std_ms": float(df["avg_commit_latency_ms"].std()),
            },
        }

        comparisons = self.compare_methods(df)
        report["comparisons"] = [r.to_dict() for r in comparisons]

        corrected = self.holm.correct([r.p_value for r in comparisons])
        report["holm_corrected"] = [r.to_dict() for r in corrected]

        try:
            report["figure3"] = self.analyze_figure3(df)
            report["figure4"] = self.analyze_figure4(df)
            report["figure5"] = self.analyze_figure5(df)
        except Exception as e:
            logger.warning(f"Some figure analyses failed: {e}")

        report_file = output_dir / "statistical_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Statistical report saved to {report_file}")
        return report

    def _mean_ci(self, data: np.ndarray) -> tuple[float, float]:
        """Compute 95% CI for mean."""
        _, ci_lower, ci_upper = self.bootstrap.paired_ci(
            data, np.zeros_like(data)
        )
        return float(ci_lower), float(ci_upper)


if __name__ == "__main__":
    rng = np.random.RandomState(42)

    n = 100
    llm_honest = rng.normal(0.72, 0.15, n)
    qre_honest = rng.normal(0.65, 0.18, n)

    analyzer = StatisticalAnalyzer()

    _, ci_lower, ci_upper = analyzer.bootstrap.paired_ci(llm_honest, qre_honest)
    p_val = analyzer.bootstrap.paired_p_value(llm_honest, qre_honest)
    d = analyzer.effect.cohens_d(llm_honest, qre_honest)

    print(f"Mean diff: {np.mean(llm_honest - qre_honest):.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"p-value: {p_val:.4f}")
    print(f"Cohen's d: {d:.4f}")

    results = [
        StatisticalResult(
            test_name="LLM vs QRE honest",
            statistic=np.mean(llm_honest - qre_honest),
            p_value=p_val,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=d,
        )
    ]

    corrected = analyzer.holm.correct([r.p_value for r in results], ["LLM vs QRE"])
    for r in corrected:
        print(f"Holm corrected p-value: {r.p_value:.4f}, significant: {r.significant}")
