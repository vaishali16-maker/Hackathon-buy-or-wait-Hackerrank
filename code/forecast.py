"""Conservative 90-day cash-balance forecasting for Buy or Wait?.

The module turns supplied financial events into a dated cash-flow ledger; it
does not choose a payment method or make optional spending reductions.  Its
purpose is to answer the baseline question: "what can be paid without ever
falling below the user's protected minimum balance?"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from currency import ExchangeRateBook, convert_to_home_currency


EXCLUDED_STATUSES = {"failed", "cancelled", "unrealized"}


class UnknownCashAmountError(ValueError):
    """A relevant cash event has no amount and must be resolved from its image."""


@dataclass(frozen=True)
class CashFlow:
    """One known or history-supported future movement in home currency."""

    flow_date: date
    amount: Decimal
    direction: str  # "debit" or "credit"
    event_id: str
    description: str
    projected: bool = False


@dataclass(frozen=True)
class DayBalance:
    """Balance checkpoints for one date, with confirmed credits processed first."""

    day: date
    opening_balance: Decimal
    debits: tuple[CashFlow, ...]
    payment: Decimal
    credits: tuple[CashFlow, ...]
    closing_balance: Decimal
    lowest_balance: Decimal


@dataclass(frozen=True)
class ForecastResult:
    """The auditable 90-day ledger and its lowest intraday balance."""

    request_date: date
    horizon_end: date
    days: tuple[DayBalance, ...]
    minimum_balance_seen: Decimal
    is_safe: bool


@dataclass(frozen=True)
class BaselineSafety:
    """Baseline affordability measures used by the later decision engine."""

    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: date | None
    forecast: ForecastResult


def simulate_balance(
    profile: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    request_date: date,
    rate_book: ExchangeRateBook,
    payments: Mapping[date, Decimal] | None = None,
    horizon_days: int = 90,
) -> ForecastResult:
    """Simulate one user's baseline balance over an inclusive 90-day window.

    The opening balance is the profile's current available balance at
    ``request_date``.  Pending debits and scheduled debits are reserved on their
    settlement dates.  Only confirmed scheduled income is included as a credit,
    and only on its settlement date; pending credits are excluded.  Failed,
    cancelled, unrealized, non-cash, and exact duplicate rows are excluded.

    Settled historical debit/income records are projected only when at least
    three recent occurrences share a stable 6--35 day cadence.  Projected debit
    amounts use the maximum of the last three occurrences, which is conservative
    for essential variable spend.  Confirmed credits settle before same-day
    debits and a proposed purchase, so settlement-date salary is available for
    that day's obligations.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    minimum_required = _decimal(profile["minimum_balance_to_keep"])
    opening = _decimal(profile["current_available_balance"])
    home_currency = str(profile["home_currency"])
    horizon_end = request_date + timedelta(days=horizon_days - 1)
    payments = {day: _decimal(amount) for day, amount in (payments or {}).items()}
    if any(amount < 0 for amount in payments.values()):
        raise ValueError("Payment amounts cannot be negative")

    flows = _future_explicit_flows(events, request_date, horizon_end, home_currency, rate_book)
    flows.extend(_recurring_flows(events, request_date, horizon_end, home_currency, rate_book))
    flows_by_day: dict[date, list[CashFlow]] = defaultdict(list)
    for flow in flows:
        flows_by_day[flow.flow_date].append(flow)

    balance = opening
    minimum_seen = opening
    daily: list[DayBalance] = []
    for offset in range(horizon_days):
        day = request_date + timedelta(days=offset)
        day_flows = flows_by_day.get(day, [])
        debits = tuple(sorted((flow for flow in day_flows if flow.direction == "debit"), key=lambda f: f.event_id))
        credits = tuple(sorted((flow for flow in day_flows if flow.direction == "credit"), key=lambda f: f.event_id))
        opening_balance = balance
        day_lowest = balance
        for flow in credits:
            balance += flow.amount
            day_lowest = min(day_lowest, balance)
            minimum_seen = min(minimum_seen, balance)
        for flow in debits:
            balance -= flow.amount
            day_lowest = min(day_lowest, balance)
            minimum_seen = min(minimum_seen, balance)
        payment = payments.get(day, Decimal("0"))
        balance -= payment
        day_lowest = min(day_lowest, balance)
        minimum_seen = min(minimum_seen, balance)
        daily.append(DayBalance(day, opening_balance, debits, payment, credits, balance, day_lowest))

    return ForecastResult(request_date, horizon_end, tuple(daily), minimum_seen, minimum_seen >= minimum_required)


def baseline_safety(
    profile: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    request_date: date,
    requested_amount: Decimal | int | float | str,
    rate_book: ExchangeRateBook,
    horizon_days: int = 90,
) -> BaselineSafety:
    """Return today's safe cap and earliest safe full-payment day.

    The safe cap is the minimum balance in the no-purchase forecast minus the
    protected minimum, capped at the requested amount.  This is valid because a
    payment made today lowers every following checkpoint by the same amount.
    Each possible full-payment date is then simulated explicitly, preserving the
    conservative within-day debit-before-credit ordering.
    """
    requested = _decimal(requested_amount)
    if requested < 0:
        raise ValueError("requested_amount cannot be negative")
    baseline = simulate_balance(profile, events, request_date, rate_book, horizon_days=horizon_days)
    minimum_required = _decimal(profile["minimum_balance_to_keep"])
    safe_amount = max(Decimal("0"), min(requested, baseline.minimum_balance_seen - minimum_required))

    earliest: date | None = None
    for day in (request_date + timedelta(days=offset) for offset in range(horizon_days)):
        candidate = simulate_balance(profile, events, request_date, rate_book, {day: requested}, horizon_days)
        if candidate.is_safe:
            earliest = day
            break
    return BaselineSafety(safe_amount, earliest, baseline)


def _future_explicit_flows(
    events: Sequence[Mapping[str, Any]], start: date, end: date, home_currency: str, rate_book: ExchangeRateBook,
) -> list[CashFlow]:
    flows: list[CashFlow] = []
    seen_cash_records: set[tuple[Any, ...]] = set()
    # A linked replacement is safe to retain when its own row is settled/pending;
    # cancelled/failed source rows are already excluded.  Event IDs are unique,
    # avoiding duplicate representation of the same supplied record.
    for event in events:
        status = str(event["status"])
        direction = str(event["direction"])
        settlement = event["settlement_date"]
        if status in EXCLUDED_STATUSES or direction == "non_cash" or not settlement:
            continue
        if not start <= settlement <= end:
            continue
        if status == "pending" and direction == "credit":
            continue
        if status == "scheduled" and direction == "credit" and event["event_type"] != "income":
            continue
        if direction not in {"debit", "credit"}:
            continue
        amount = event["amount"]
        if amount is None:
            raise UnknownCashAmountError(
                f"{event['event_id']} is a relevant {status} {direction} with no amount; resolve its linked image first."
            )
        # Historical settled rows are already represented in current balance.
        if status == "settled" and settlement < start:
            continue
        duplicate_key = (
            settlement, direction, event["amount"], event["currency"], event["event_type"],
            event["category"], event["description"], status,
        )
        if duplicate_key in seen_cash_records:
            continue
        seen_cash_records.add(duplicate_key)
        flows.append(_to_home_flow(event, settlement, home_currency, rate_book, projected=False))
    return flows


def _recurring_flows(
    events: Sequence[Mapping[str, Any]], start: date, end: date, home_currency: str, rate_book: ExchangeRateBook,
) -> list[CashFlow]:
    """Project only stable, recent settled debit/income histories into the horizon."""
    history_start = start - timedelta(days=180)
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    explicit_keys = {
        (event["event_type"], event["category"], event["description"], event["currency"], event["settlement_date"])
        for event in events
        if event["settlement_date"] and start <= event["settlement_date"] <= end and event["status"] not in EXCLUDED_STATUSES
    }
    for event in events:
        settlement = event["settlement_date"]
        if not settlement or not history_start <= settlement < start:
            continue
        if event["status"] != "settled" or event["direction"] not in {"debit", "credit"} or event["amount"] is None:
            continue
        if event["linked_event_id"] or event["event_type"] not in {"expense", "subscription", "debt_payment", "income"}:
            continue
        if event["direction"] == "credit" and event["event_type"] != "income":
            continue
        groups[(event["event_type"], event["category"], event["description"], event["currency"])].append(event)

    projected: list[CashFlow] = []
    for key, records in groups.items():
        records.sort(key=lambda row: row["settlement_date"])
        if len(records) < 3:
            continue
        intervals = [(right["settlement_date"] - left["settlement_date"]).days for left, right in zip(records, records[1:])]
        cadence = int(round(median(intervals[-3:])))
        if not 6 <= cadence <= 35 or max(intervals[-3:]) - min(intervals[-3:]) > max(3, cadence // 5):
            continue
        current = records[-1]["settlement_date"]
        while current < start:
            current = _advance_recurrence(current, cadence)
        amount = max(_decimal(row["amount"]) for row in records[-3:]) if records[-1]["direction"] == "debit" else _decimal(records[-1]["amount"])
        while current <= end:
            if (*key, current) not in explicit_keys:
                template = records[-1]
                projected.append(_to_home_flow({**template, "amount": amount}, current, home_currency, rate_book, projected=True))
            current = _advance_recurrence(current, cadence)
    return projected

def _advance_recurrence(current: date, cadence: int) -> date:
    """Keep monthly patterns on their calendar day; use exact days otherwise."""
    if 26 <= cadence <= 35:
        month = current.month + 1
        year = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        days_in_month = (date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)).day
        return date(year, month, min(current.day, days_in_month))
    return current + timedelta(days=cadence)


def _to_home_flow(event: Mapping[str, Any], flow_date: date, home_currency: str, rate_book: ExchangeRateBook, projected: bool) -> CashFlow:
    amount = convert_to_home_currency(event["amount"], event["currency"], home_currency, flow_date, rate_book)
    return CashFlow(flow_date, amount, str(event["direction"]), str(event["event_id"]), str(event["description"]), projected)


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))
