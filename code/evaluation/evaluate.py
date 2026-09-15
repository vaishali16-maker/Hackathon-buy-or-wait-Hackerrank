"""Evaluate the Buy or Wait? pipeline against the 25 solved sample requests.

Run from the repository root with ``python code/evaluation/evaluate.py``.
The evaluator writes a temporary CSV through the production formatter, compares
the serialized result with ``dataset/sample_requests.csv``, and never replaces
the root-level submission ``output.csv``.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from currency import ExchangeRateBook  # noqa: E402
from data_loader import Dataset, load_dataset  # noqa: E402
from decision_engine import decide  # noqa: E402
from forecast import UnknownCashAmountError, baseline_safety  # noqa: E402
from formatter import OUTPUT_COLUMNS, write_output  # noqa: E402


COMPARED_COLUMNS = (
    "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
)


@dataclass(frozen=True)
class EvaluationReport:
    total_rows: int
    field_matches: Mapping[str, int]
    mismatches: tuple[tuple[str, str, str, str], ...]
    pipeline_errors: tuple[tuple[str, str], ...]

    @property
    def exact_rows(self) -> int:
        mismatch_ids = {request_id for request_id, _, _, _ in self.mismatches}
        error_ids = {request_id for request_id, _ in self.pipeline_errors}
        return self.total_rows - len(mismatch_ids | error_ids)


def evaluate(
    dataset_dir: str | Path | None = None,
    amount_overrides: Mapping[str, Decimal] | None = None,
) -> EvaluationReport:
    """Run all solved samples through loader, forecast, decision, and formatter."""
    data = load_dataset(dataset_dir)
    amount_overrides = amount_overrides or {}
    rate_book = ExchangeRateBook.from_rows(data.exchange_rates)
    decisions = []
    usable_samples = []
    errors: list[tuple[str, str]] = []
    for sample in data.sample_requests:
        request_id = str(sample["request_id"])
        try:
            profile = data.profiles_by_user[sample["user_id"]]
            events = tuple(
                {**event, "amount": amount_overrides.get(event["event_id"], event["amount"])}
                for event in data.events_by_user.get(sample["user_id"], ())
            )
            baseline = baseline_safety(profile, events, sample["request_date"], sample["requested_amount"], rate_book)
            decisions.append(decide(sample, profile, events, data.payment_options_by_request.get(request_id, ()), baseline, rate_book))
            usable_samples.append(sample)
        except (KeyError, UnknownCashAmountError, ValueError) as error:
            errors.append((request_id, f"{type(error).__name__}: {error}"))

    actual_by_id: dict[str, Mapping[str, str]] = {}
    if decisions:
        with tempfile.TemporaryDirectory(prefix="buy_or_wait_sample_") as temporary_dir:
            generated_path = write_output(decisions, usable_samples, Path(temporary_dir) / "output.csv")
            with generated_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
                    raise RuntimeError("Formatter produced an unexpected header during evaluation")
                actual_by_id = {row["request_id"]: row for row in reader}

    matches = {column: 0 for column in COMPARED_COLUMNS}
    mismatches: list[tuple[str, str, str, str]] = []
    for sample in data.sample_requests:
        request_id = str(sample["request_id"])
        actual = actual_by_id.get(request_id)
        if actual is None:
            continue
        for column in COMPARED_COLUMNS:
            expected = _serialize_expected(sample[column])
            observed = actual[column]
            if observed == expected:
                matches[column] += 1
            else:
                mismatches.append((request_id, column, expected, observed))
    return EvaluationReport(len(data.sample_requests), matches, tuple(mismatches), tuple(errors))


def print_report(report: EvaluationReport) -> None:
    """Print concise per-field totals followed by every side-by-side mismatch."""
    print(f"Sample requests: {report.total_rows}")
    print(f"Exact rows: {report.exact_rows}/{report.total_rows}")
    print("Per-field exact matches:")
    for column in COMPARED_COLUMNS:
        print(f"  {column}: {report.field_matches[column]}/{report.total_rows}")
    if report.pipeline_errors:
        print("\nPipeline errors:")
        for request_id, error in report.pipeline_errors:
            print(f"  {request_id}: {error}")
    if report.mismatches:
        print("\nMismatches (request_id | field | expected | actual):")
        for request_id, field, expected, actual in report.mismatches:
            print(f"  {request_id} | {field} | {expected!r} | {actual!r}")
    else:
        print("\nAll compared values match exactly.")


def _serialize_expected(value: Any) -> str:
    if value is None:
        return ""
    # Dates and Decimals are already parsed by data_loader; match formatter's
    # stable non-exponent decimal representation for fair string comparison.
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value.__class__.__name__ == "Decimal":
        rendered = format(value, "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    return str(value)


if __name__ == "__main__":
    print_report(evaluate())
