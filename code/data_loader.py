"""Load and join the participant-facing Buy or Wait? dataset.

This module deliberately performs no financial interpretation.  It validates the
published CSV schemas, parses dates/amounts into safe Python values, and exposes
all evidence associated with a request for later forecasting and decision code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DATASET_FILES = {
    "profiles": "financial_profiles.csv",
    "events": "financial_events.csv",
    "exchange_rates": "exchange_rates.csv",
    "requests": "requests.csv",
    "sample_requests": "sample_requests.csv",
    "payment_options": "request_payment_options.csv",
    "messages": "messages.csv",
    "images": "images.csv",
}
# Manually resolved amounts extracted from linked images (see resolve_images.py
# and log.txt for the extraction evidence). Applied only to rows whose amount
# is genuinely blank in the source CSV.
RESOLVED_IMAGE_AMOUNTS: dict[str, Decimal] = {
    "event_1442": Decimal("200000.00"),
    "event_1786": Decimal("704.05"),
}
REQUIRED_COLUMNS = {
    "profiles": (
        "user_id", "home_currency", "current_available_balance",
        "minimum_balance_to_keep", "financial_priorities",
        "expense_categories_to_protect", "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider", "max_installment_months",
    ),
    "events": (
        "event_id", "user_id", "event_type", "description", "category", "direction",
        "amount", "currency", "event_date", "settlement_date", "status",
        "linked_event_id", "flexibility", "minimum_allowed_amount",
    ),
    "exchange_rates": ("rate_date", "from_currency", "to_currency", "rate"),
    "requests": (
        "request_id", "user_id", "request_date", "request_type", "requested_amount",
        "desired_completion_date", "allows_partial_payment", "request_text",
    ),
    "sample_requests": (
        "request_id", "user_id", "request_date", "request_type", "requested_amount",
        "desired_completion_date", "allows_partial_payment", "request_text",
        "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
        "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation",
    ),
    "payment_options": (
        "payment_option_id", "request_id", "payment_method", "payment_amount",
        "number_of_payments", "first_payment_date", "payment_frequency_days",
        "financing_fee", "total_payable_amount",
    ),
    "messages": (
        "message_id", "user_id", "request_id", "related_event_id", "sent_at",
        "source_type", "message_text",
    ),
    "images": ("image_id", "user_id", "request_id", "related_event_id"),
}

DATE_COLUMNS = {
    "events": {"event_date", "settlement_date"},
    "exchange_rates": {"rate_date"},
    "requests": {"request_date", "desired_completion_date"},
    "sample_requests": {"request_date", "desired_completion_date", "earliest_date_for_full_payment"},
    "payment_options": {"first_payment_date"},
}
DECIMAL_COLUMNS = {
    "profiles": {"current_available_balance", "minimum_balance_to_keep"},
    "events": {"amount", "minimum_allowed_amount"},
    "exchange_rates": {"rate"},
    "requests": {"requested_amount"},
    "sample_requests": {"requested_amount", "amount_safe_to_pay"},
    "payment_options": {"payment_amount", "financing_fee", "total_payable_amount"},
}
INTEGER_COLUMNS = {"profiles": {"max_installment_months"}, "payment_options": {"number_of_payments", "payment_frequency_days"}}


class DatasetSchemaError(ValueError):
    """Raised when a participant-facing CSV does not match its published schema."""


@dataclass(frozen=True)
class RequestBundle:
    """All structured records and image locations relevant to one request."""

    request: Mapping[str, Any]
    profile: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    payment_options: tuple[Mapping[str, Any], ...]
    messages: tuple[Mapping[str, Any], ...]
    images: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Dataset:
    """Validated input tables plus indexes used by the decision pipeline."""

    root: Path
    profiles: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    exchange_rates: tuple[Mapping[str, Any], ...]
    requests: tuple[Mapping[str, Any], ...]
    sample_requests: tuple[Mapping[str, Any], ...]
    payment_options: tuple[Mapping[str, Any], ...]
    messages: tuple[Mapping[str, Any], ...]
    images: tuple[Mapping[str, Any], ...]
    profiles_by_user: Mapping[str, Mapping[str, Any]]
    requests_by_id: Mapping[str, Mapping[str, Any]]
    events_by_user: Mapping[str, tuple[Mapping[str, Any], ...]]
    payment_options_by_request: Mapping[str, tuple[Mapping[str, Any], ...]]
    messages_by_user: Mapping[str, tuple[Mapping[str, Any], ...]]
    images_by_user: Mapping[str, tuple[Mapping[str, Any], ...]]

    def bundle_for_request(self, request_id: str) -> RequestBundle:
        """Return the profile, cash records, offers, and evidence for *request_id*.

        User-level messages/images are included because they can amend a user's
        event or income.  Request-specific records are included even if their
        user_id is blank.  Each image row gets ``image_path`` and ``image_exists``
        fields; image OCR/interpretation belongs in a later component.
        """
        try:
            request = self.requests_by_id[request_id]
        except KeyError as error:
            raise KeyError(f"Unknown request_id: {request_id}") from error
        user_id = request["user_id"]
        try:
            profile = self.profiles_by_user[user_id]
        except KeyError as error:
            raise DatasetSchemaError(
                f"Request {request_id} refers to missing profile {user_id}"
            ) from error

        events = self.events_by_user.get(user_id, ())
        event_ids = {event["event_id"] for event in events}
        messages = _deduplicate_by_id(
            (*self.messages_by_user.get(user_id, ()),
             *(row for row in self.messages if row["request_id"] == request_id),
             *(row for row in self.messages if row["related_event_id"] in event_ids)),
            "message_id",
        )
        images = _deduplicate_by_id(
            (*self.images_by_user.get(user_id, ()),
             *(row for row in self.images if row["request_id"] == request_id),
             *(row for row in self.images if row["related_event_id"] in event_ids)),
            "image_id",
        )
        return RequestBundle(
            request=request,
            profile=profile,
            events=events,
            payment_options=self.payment_options_by_request.get(request_id, ()),
            messages=messages,
            images=images,
        )


def load_dataset(dataset_dir: str | Path | None = None) -> Dataset:
    """Load all participant-facing CSVs from *dataset_dir* (default: repo dataset)."""
    root = Path(dataset_dir) if dataset_dir is not None else Path(__file__).resolve().parents[1] / "dataset"
    root = root.resolve()
    tables = {name: _read_table(root / filename, name) for name, filename in DATASET_FILES.items()}
    _assert_unique(tables["profiles"], "user_id", "profiles")
    _assert_unique(tables["requests"], "request_id", "requests")
    _assert_unique(tables["events"], "event_id", "events")
    _assert_unique(tables["payment_options"], "payment_option_id", "payment_options")
    _assert_unique(tables["messages"], "message_id", "messages")
    _assert_unique(tables["images"], "image_id", "images")

    images = tuple(_with_image_path(row, root) for row in tables["images"])
    return Dataset(
        root=root,
        profiles=tables["profiles"], events=tables["events"], exchange_rates=tables["exchange_rates"],
        requests=tables["requests"], sample_requests=tables["sample_requests"],
        payment_options=tables["payment_options"], messages=tables["messages"], images=images,
        profiles_by_user=_index_one(tables["profiles"], "user_id"),
        requests_by_id=_index_one(tables["requests"], "request_id"),
        events_by_user=_index_many(tables["events"], "user_id"),
        payment_options_by_request=_index_many(tables["payment_options"], "request_id"),
        messages_by_user=_index_many(tables["messages"], "user_id"),
        images_by_user=_index_many(images, "user_id"),
    )


def _read_table(path: Path, table: str) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        expected = REQUIRED_COLUMNS[table]
        if actual != expected:
            raise DatasetSchemaError(f"{path.name} has columns {actual!r}; expected {expected!r}")
        rows = tuple(_parse_row(row, table, path.name, line_number) for line_number, row in enumerate(reader, start=2))
        if table == "events":
            rows = tuple(_apply_resolved_amounts(row) for row in rows)
        return rows
def _apply_resolved_amounts(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fill in amounts manually resolved from linked images (see resolve_images.py)."""
    resolved = RESOLVED_IMAGE_AMOUNTS.get(row["event_id"])
    if resolved is not None and row["amount"] is None:
        return MappingProxyType({**row, "amount": resolved})
    return row

def _parse_row(row: Mapping[str, str | None], table: str, filename: str, line_number: int) -> Mapping[str, Any]:
    parsed: dict[str, Any] = dict(row)
    for column in DATE_COLUMNS.get(table, set()):
        parsed[column] = _parse_date(parsed[column], filename, line_number, column)
    for column in DECIMAL_COLUMNS.get(table, set()):
        parsed[column] = _parse_decimal(parsed[column], filename, line_number, column)
    for column in INTEGER_COLUMNS.get(table, set()):
        parsed[column] = _parse_int(parsed[column], filename, line_number, column)
    if table in {"requests", "sample_requests"}:
        value = parsed["allows_partial_payment"]
        if value not in {"true", "false"}:
            raise DatasetSchemaError(f"{filename}:{line_number} allows_partial_payment must be true or false")
        parsed["allows_partial_payment"] = value == "true"
    return MappingProxyType(parsed)


def _parse_date(value: str | None, filename: str, line_number: int, column: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise DatasetSchemaError(f"{filename}:{line_number} invalid {column}: {value!r}") from error


def _parse_decimal(value: str | None, filename: str, line_number: int, column: str) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise DatasetSchemaError(f"{filename}:{line_number} invalid {column}: {value!r}") from error


def _parse_int(value: str | None, filename: str, line_number: int, column: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise DatasetSchemaError(f"{filename}:{line_number} invalid {column}: {value!r}") from error


def _with_image_path(row: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    image_path = root / "media" / "images" / f"{row['image_id']}.png"
    return MappingProxyType({**row, "image_path": image_path, "image_exists": image_path.is_file()})


def _index_one(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType({row[key]: row for row in rows})


def _index_many(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row[key]:
            grouped.setdefault(row[key], []).append(row)
    return MappingProxyType({group_key: tuple(values) for group_key, values in grouped.items()})


def _deduplicate_by_id(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[Mapping[str, Any], ...]:
    return tuple({row[key]: row for row in rows}.values())


def _assert_unique(rows: Sequence[Mapping[str, Any]], key: str, table: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = row[key]
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise DatasetSchemaError(f"{table} has duplicate {key} values: {sorted(duplicates)!r}")


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Loaded {len(dataset.requests)} requests, {len(dataset.profiles)} profiles, and {len(dataset.events)} events.")
