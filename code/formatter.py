"""Validate Buy or Wait? decisions and write the required root-level output.csv."""

from __future__ import annotations

import csv
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from decision_engine import Decision


OUTPUT_COLUMNS = (
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment",
    "spending_changes_needed", "decision_explanation",
)
_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
_PAYMENT_ENTRY = re.compile(r"^(\d{4}-\d{2}-\d{2}):(-?\d+(?:\.\d+)?)$")
_STOP_ENTRY = re.compile(r"^stop:[^:|]+$")
_REDUCE_ENTRY = re.compile(r"^reduce_to:[^:|]+:-?\d+(?:\.\d+)?$")


class OutputValidationError(ValueError):
    """A decision cannot safely be emitted as a challenge submission row."""


def write_output(
    decisions: Iterable[Decision],
    requests: Iterable[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    """Validate one decision per request and write the exact submission CSV.

    ``output_path`` defaults to ``<repo root>/output.csv``, never the blank
    template under ``dataset/``.  Invalid safe amounts are errors, rather than
    values that this writer silently clips or repairs.
    """
    request_rows = tuple(requests)
    decision_rows = tuple(decisions)
    decisions_by_id = {decision.request_id: decision for decision in decision_rows}
    request_ids = [str(request["request_id"]) for request in request_rows]
    if len(decisions_by_id) != len(decision_rows):
        raise OutputValidationError("Duplicate decision request_id values")
    if set(decisions_by_id) != set(request_ids):
        missing = sorted(set(request_ids) - set(decisions_by_id))
        unexpected = sorted(set(decisions_by_id) - set(request_ids))
        raise OutputValidationError(f"Decision IDs do not match requests; missing={missing}, unexpected={unexpected}")

    rows = []
    for request in request_rows:
        decision = decisions_by_id[str(request["request_id"])]
        _validate_decision(decision, request)
        rows.append(_to_row(decision))

    target = Path(output_path) if output_path is not None else Path(__file__).resolve().parents[1] / "output.csv"
    target = target.resolve()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return target


def _validate_decision(decision: Decision, request: Mapping[str, Any]) -> None:
    requested = _decimal(request["requested_amount"])
    safe = _decimal(decision.amount_safe_to_pay)
    if not Decimal("0") <= safe <= requested:
        raise OutputValidationError(
            f"{decision.request_id}: amount_safe_to_pay {safe} is outside [0, {requested}]"
        )
    if decision.affordability_status not in _STATUSES:
        raise OutputValidationError(f"{decision.request_id}: invalid affordability_status")
    if decision.recommended_payment_method not in _METHODS:
        raise OutputValidationError(f"{decision.request_id}: invalid recommended_payment_method")
    _validate_payment_plan(decision.payment_plan, decision.request_id)
    _validate_spending_changes(decision.spending_changes_needed, decision.request_id)
    if not decision.decision_explanation or len(decision.decision_explanation) > 600:
        raise OutputValidationError(f"{decision.request_id}: explanation must be non-empty and concise")
    if decision.affordability_status == "affordable_now" and decision.earliest_date_for_full_payment != request["request_date"]:
        raise OutputValidationError(f"{decision.request_id}: affordable_now requires earliest_date_for_full_payment=request_date")
    if decision.recommended_payment_method in {"wait", "not_recommended"} and decision.payment_plan != "none":
        raise OutputValidationError(f"{decision.request_id}: {decision.recommended_payment_method} must have payment_plan=none")


def _validate_payment_plan(plan: str, request_id: str) -> None:
    if plan == "none":
        return
    dates: list[date] = []
    for part in plan.split("|"):
        match = _PAYMENT_ENTRY.match(part)
        if not match:
            raise OutputValidationError(f"{request_id}: invalid payment_plan entry {part!r}")
        parsed_date = date.fromisoformat(match.group(1))
        if Decimal(match.group(2)) < 0:
            raise OutputValidationError(f"{request_id}: payment amount cannot be negative")
        dates.append(parsed_date)
    if dates != sorted(dates):
        raise OutputValidationError(f"{request_id}: payment_plan dates are not chronological")


def _validate_spending_changes(changes: str, request_id: str) -> None:
    if changes == "none":
        return
    entries = changes.split("|")
    if len(entries) > 3 or not all(_STOP_ENTRY.match(entry) or _REDUCE_ENTRY.match(entry) for entry in entries):
        raise OutputValidationError(f"{request_id}: invalid spending_changes_needed")
    targets = [entry.split(":")[1] for entry in entries]
    if len(set(targets)) != len(targets):
        raise OutputValidationError(f"{request_id}: spending changes cannot target one event twice")


def _to_row(decision: Decision) -> dict[str, str]:
    return {
        "request_id": decision.request_id,
        "amount_safe_to_pay": _format_amount(_decimal(decision.amount_safe_to_pay)),
        "affordability_status": decision.affordability_status,
        "recommended_payment_method": decision.recommended_payment_method,
        "payment_plan": decision.payment_plan,
        "earliest_date_for_full_payment": "" if decision.earliest_date_for_full_payment is None else decision.earliest_date_for_full_payment.isoformat(),
        "spending_changes_needed": decision.spending_changes_needed,
        "decision_explanation": decision.decision_explanation,
    }


def _format_amount(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))
