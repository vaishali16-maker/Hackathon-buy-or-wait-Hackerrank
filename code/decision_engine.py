"""Choose the safest eligible baseline payment recommendation.

This module consumes the forecast baseline and supplied seller options.  It
never fabricates financing or optional spending reductions: an installment plan
is copied exactly from an option, and this baseline version emits ``none`` for
spending changes.  A later optimisation layer may add changes only after
validating flexible recurring events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Mapping, Sequence

from currency import ExchangeRateBook
from forecast import BaselineSafety, simulate_balance


@dataclass(frozen=True)
class Decision:
    """Fields ready for the output formatter, except Decimal rendering policy."""

    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class _Candidate:
    status: str
    method: str
    payments: tuple[tuple[date, Decimal], ...]
    total_paid: Decimal
    option_id: str
    explanation: str

    @property
    def start_date(self) -> date:
        return self.payments[0][0]


def decide(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    payment_options: Sequence[Mapping[str, Any]],
    baseline: BaselineSafety,
    rate_book: ExchangeRateBook,
    horizon_days: int = 90,
) -> Decision:
    """Return the highest-ranked safe, profile-eligible recommendation.

    A candidate is accepted only after its dated payment schedule passes the
    same conservative forecast used for the baseline.  Candidates are ranked by
    deadline, no spending changes, lower total cost, earlier start, fewer
    payments, then lower payment-option ID.  This version deliberately has no
    optional spending changes, so every generated candidate has ``none``.
    """
    requested = _decimal(request["requested_amount"])
    request_date = request["request_date"]
    desired_date = request["desired_completion_date"]
    currency = str(profile["home_currency"])
    minimum = _decimal(profile["minimum_balance_to_keep"])
    allowed = _split_preferences(profile["payment_methods_user_will_consider"])
    candidates: list[_Candidate] = []

    if "full_payment" in allowed and baseline.amount_safe_to_pay >= requested:
        candidates.append(_Candidate(
            "affordable_now", "full_payment", ((request_date, requested),), requested, "",
            f"Pay {currency} {_format_amount(requested)} today; the 90-day forecast stays above the {currency} {_format_amount(minimum)} minimum.",
        ))

    earliest = baseline.earliest_date_for_full_payment
    if (
        "partial_payment" in allowed
        and request["allows_partial_payment"]
        and Decimal("0") < baseline.amount_safe_to_pay < requested
        and earliest is not None
        and earliest <= desired_date
    ):
        partial_schedule = (
            (request_date, baseline.amount_safe_to_pay),
            (earliest, requested - baseline.amount_safe_to_pay),
        )
        if _schedule_is_safe(profile, events, partial_schedule, request_date, rate_book, horizon_days):
            candidates.append(_Candidate(
                "affordable_with_plan", "partial_payment", partial_schedule, requested, "",
                f"Pay {currency} {_format_amount(baseline.amount_safe_to_pay)} today and {currency} {_format_amount(requested - baseline.amount_safe_to_pay)} on {earliest.isoformat()}; both payments preserve the {currency} {_format_amount(minimum)} minimum.",
            ))

    if "installments" in allowed:
        max_months = profile["max_installment_months"]
        for option in payment_options:
            if option["payment_method"] != "installments":
                continue
            if max_months is not None and option["number_of_payments"] > max_months:
                continue
            schedule = _option_schedule(option)
            if not schedule or schedule[-1][0] > desired_date:
                continue
            if _schedule_is_safe(profile, events, schedule, request_date, rate_book, horizon_days):
                candidates.append(_Candidate(
                    "affordable_with_plan", "installments", schedule,
                    _decimal(option["total_payable_amount"]), str(option["payment_option_id"]),
                    f"Use option {option['payment_option_id']}: {len(schedule)} payments totaling {currency} {_format_amount(_decimal(option['total_payable_amount']))}, while maintaining the {currency} {_format_amount(minimum)} minimum.",
                ))

    # Waiting is not a payment plan: capacity must appear later, the user must
    # accept full payment, and the request still has to complete by its deadline.
    if (
        "full_payment" in allowed
        and earliest is not None
        and earliest > request_date
        and earliest <= desired_date
    ):
        candidates.append(_Candidate(
            "affordable_later", "wait", (), requested, "",
            f"Wait until {earliest.isoformat()}: the full {currency} {_format_amount(requested)} first becomes safe then while preserving the {currency} {_format_amount(minimum)} minimum.",
        ))

    if not candidates:
        return Decision(
            str(request["request_id"]), baseline.amount_safe_to_pay, "not_affordable", "not_recommended",
            "none", baseline.earliest_date_for_full_payment, "none",
            f"Only {currency} {_format_amount(baseline.amount_safe_to_pay)} is safe today versus the requested {currency} {_format_amount(requested)}, and no accepted option completes by {desired_date.isoformat()}.",
        )

    selected = min(candidates, key=lambda candidate: _rank(candidate, desired_date))
    return Decision(
        str(request["request_id"]), baseline.amount_safe_to_pay, selected.status, selected.method,
        _format_plan(selected.payments), baseline.earliest_date_for_full_payment, "none", selected.explanation,
    )


def validate_spending_changes(changes: Sequence[str], events: Sequence[Mapping[str, Any]]) -> bool:
    """Validate future spending-change actions without applying them.

    At most three actions are allowed.  Every target must be a flexible recurring
    debit, and a single event cannot appear in both a stop and reduce action.
    The decision engine currently emits no such actions; this guard is provided
    for the later spending-optimisation step.
    """
    if len(changes) > 3:
        return False
    by_id = {str(event["event_id"]): event for event in events}
    seen: set[str] = set()
    for action in changes:
        parts = action.split(":")
        if parts[0] == "stop" and len(parts) == 2:
            event_id = parts[1]
        elif parts[0] == "reduce_to" and len(parts) == 3:
            event_id = parts[1]
        else:
            return False
        event = by_id.get(event_id)
        if event is None or event_id in seen:
            return False
        if (
            event["flexibility"] != "flexible"
            or event["direction"] != "debit"
            or event["event_type"] not in {"expense", "subscription", "debt_payment"}
        ):
            return False
        seen.add(event_id)
    return True


def _option_schedule(option: Mapping[str, Any]) -> tuple[tuple[date, Decimal], ...]:
    """Copy an offered installment schedule; never derive a custom alternative."""
    first_date = option["first_payment_date"]
    count = option["number_of_payments"]
    frequency = option["payment_frequency_days"]
    if not first_date or not count or not frequency or count < 2:
        return ()
    amount = _decimal(option["payment_amount"])
    return tuple((first_date + timedelta(days=frequency * index), amount) for index in range(count))


def _schedule_is_safe(
    profile: Mapping[str, Any], events: Sequence[Mapping[str, Any]], schedule: Sequence[tuple[date, Decimal]],
    request_date: date, rate_book: ExchangeRateBook, horizon_days: int,
) -> bool:
    end = request_date + timedelta(days=horizon_days - 1)
    if any(day < request_date or day > end for day, _ in schedule):
        return False
    payments: dict[date, Decimal] = {}
    for day, amount in schedule:
        payments[day] = payments.get(day, Decimal("0")) + amount
    return simulate_balance(profile, events, request_date, rate_book, payments, horizon_days).is_safe


def _rank(candidate: _Candidate, desired_date: date) -> tuple[Any, ...]:
    completion = candidate.payments[-1][0] if candidate.payments else desired_date
    return (
        completion > desired_date,
        False,  # all candidates in this baseline engine require no changes
        candidate.total_paid,
        candidate.start_date if candidate.payments else completion,
        len(candidate.payments),
        candidate.option_id,
    )


def _format_plan(payments: Sequence[tuple[date, Decimal]]) -> str:
    if not payments:
        return "none"
    return "|".join(f"{day.isoformat()}:{_format_amount(amount)}" for day, amount in payments)


def _format_amount(amount: Decimal) -> str:
    return format(amount, "f").rstrip("0").rstrip(".") if "." in format(amount, "f") else format(amount, "f")


def _split_preferences(value: str) -> set[str]:
    return {item for item in str(value).split("|") if item}


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))
