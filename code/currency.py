"""Fixed, dated currency conversion for the Buy or Wait? dataset."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping


class ExchangeRateNotFoundError(LookupError):
    """Raised when the dataset has no exact rate for a requested conversion."""


class ExchangeRateBook:
    """An exact-match index of supplied exchange rates.

    Rates are directional: a USD->EUR rate is not assumed to imply EUR->USD.
    This avoids silently making an unsupported financial assumption.
    """

    def __init__(self, rates: Mapping[tuple[date, str, str], Decimal]):
        self._rates = dict(rates)

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> "ExchangeRateBook":
        rates: dict[tuple[date, str, str], Decimal] = {}
        for row in rows:
            rate_date = _as_date(row["rate_date"])
            key = (rate_date, str(row["from_currency"]), str(row["to_currency"]))
            if key in rates:
                raise ValueError(f"Duplicate exchange rate for {key!r}")
            rates[key] = _as_decimal(row["rate"])
        return cls(rates)

    def convert_to_home_currency(
        self,
        amount: Decimal | int | float | str,
        from_currency: str,
        to_currency: str,
        on_date: date | str,
    ) -> Decimal:
        return convert_to_home_currency(amount, from_currency, to_currency, on_date, self)

    def rate_for(self, from_currency: str, to_currency: str, on_date: date | str) -> Decimal:
        rate_date = _as_date(on_date)
        key = (rate_date, from_currency, to_currency)
        try:
            return self._rates[key]
        except KeyError as error:
            raise ExchangeRateNotFoundError(
                "No fixed exchange rate for "
                f"{from_currency}->{to_currency} on {rate_date.isoformat()}. "
                "The converter does not infer reverse, cross, or nearest-date rates."
            ) from error


def load_exchange_rates(dataset_dir: str | Path | None = None) -> ExchangeRateBook:
    """Load ``exchange_rates.csv`` through the validated data-loader schema."""
    # Imported lazily so this module also works when run directly from code/.
    from data_loader import load_dataset

    return ExchangeRateBook.from_rows(load_dataset(dataset_dir).exchange_rates)


def convert_to_home_currency(
    amount: Decimal | int | float | str,
    from_currency: str,
    to_currency: str,
    on_date: date | str,
    rate_book: ExchangeRateBook,
) -> Decimal:
    """Convert *amount* using the exact supplied rate for *on_date*.

    ``rate_book`` must be built from ``exchange_rates.csv``.  When both
    currencies match, the original numeric value is returned as a Decimal and
    no rate lookup is performed.
    """
    decimal_amount = _as_decimal(amount)
    if from_currency == to_currency:
        return decimal_amount
    return decimal_amount * rate_book.rate_for(from_currency, to_currency, on_date)


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected YYYY-MM-DD date, got {value!r}") from error


def _as_decimal(value: Decimal | int | float | str) -> Decimal:
    try:
        # str avoids Decimal's binary-float expansion when callers pass a float.
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Expected a numeric amount, got {value!r}") from error
