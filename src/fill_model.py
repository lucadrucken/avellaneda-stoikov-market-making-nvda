import math
from dataclasses import dataclass
from typing import Optional

from config import (
    ORDER_SIZE,
    PARTICIPATION_RATE,
    FILL_RULE,
    TICK_SIZE,
    K_MIN,
    K_MAX,
    ladder_column_name,
)


@dataclass
class Fill:
    side: str
    price: float
    size: float


def _is_finite_number(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _lookup_ladder_volume(
    next_row: dict,
    label: str,
    quote_price: float,
    anchor: Optional[float],
    tick_size: float,
    direction: str,
) -> Optional[float]:
    if not _is_finite_number(anchor):
        return None

    if direction == "down":
        k = round((float(anchor) - quote_price) / tick_size)
    else:
        k = round((quote_price - float(anchor)) / tick_size)

    if k < K_MIN or k > K_MAX:
        return None

    value = next_row.get(ladder_column_name(label, k))
    return float(value) if _is_finite_number(value) else 0.0


def _resolve_fill_size(
    vol_bis_quote: float,
    queue_ahead: float,
    order_size: float,
    participation_rate: float,
    fill_rule: str,
) -> float:
    if fill_rule == "optimistic":
        raw_size = vol_bis_quote * participation_rate
    elif fill_rule == "conservative":
        raw_size = participation_rate * max(0.0, vol_bis_quote - queue_ahead)
    else:
        raise ValueError(
            f"Unbekannte fill_rule: {fill_rule!r} (erwartet 'conservative' oder 'optimistic')."
        )

    return float(math.floor(min(order_size, raw_size)))


def check_fill(
    bid_quote: Optional[float],
    ask_quote: Optional[float],
    best_bid: Optional[float],
    best_ask: Optional[float],
    next_row: dict,
    queue_ahead_bid: float = 0.0,
    queue_ahead_ask: float = 0.0,
    order_size: float = ORDER_SIZE,
    participation_rate: float = PARTICIPATION_RATE,
    fill_rule: str = FILL_RULE,
    tick_size: float = TICK_SIZE,
) -> list[Fill]:
    if not _is_finite_number(order_size) or order_size <= 0:
        raise ValueError("order_size muss positiv und endlich sein.")

    if not _is_finite_number(participation_rate) or not 0 < participation_rate <= 1:
        raise ValueError("participation_rate muss größer als 0 und höchstens 1 sein.")

    fills: list[Fill] = []

    if _is_finite_number(bid_quote):
        bid_quote_f = float(bid_quote)
        vol_bis_quote = _lookup_ladder_volume(
            next_row, "sell", bid_quote_f, best_bid, tick_size, direction="down"
        )
        if vol_bis_quote is not None and vol_bis_quote > 0:
            fill_size = _resolve_fill_size(
                vol_bis_quote, queue_ahead_bid, order_size, participation_rate, fill_rule
            )
            if fill_size > 0:
                fills.append(Fill(side="buy", price=bid_quote_f, size=fill_size))

    if _is_finite_number(ask_quote):
        ask_quote_f = float(ask_quote)
        vol_bis_quote = _lookup_ladder_volume(
            next_row, "buy", ask_quote_f, best_ask, tick_size, direction="up"
        )
        if vol_bis_quote is not None and vol_bis_quote > 0:
            fill_size = _resolve_fill_size(
                vol_bis_quote, queue_ahead_ask, order_size, participation_rate, fill_rule
            )
            if fill_size > 0:
                fills.append(Fill(side="sell", price=ask_quote_f, size=fill_size))

    return fills