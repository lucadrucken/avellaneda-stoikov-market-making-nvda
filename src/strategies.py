import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Quote:
    bid: Optional[float] = None
    ask: Optional[float] = None
    reference_price: Optional[float] = None
    inventory_penalty: Optional[float] = None


TICK_TOLERANCE = 1e-6


def _snap_to_tick(price: float, tick_size: float, mode: str) -> float:
    ticks = price / tick_size
    nearest = round(ticks)

    if abs(ticks - nearest) < TICK_TOLERANCE:
        ticks = nearest
    elif mode == "floor":
        ticks = math.floor(ticks)
    else:
        ticks = math.ceil(ticks)

    return round(ticks * tick_size, 10)


def floor_to_tick(price: float, tick_size: float) -> float:
    return _snap_to_tick(price, tick_size, "floor")


def ceil_to_tick(price: float, tick_size: float) -> float:
    return _snap_to_tick(price, tick_size, "ceil")


def enforce_valid_quote(bid: float, ask: float, tick_size: float) -> tuple[float, float]:
    if bid < tick_size:
        bid = tick_size
    if ask <= bid:
        ask = bid + tick_size
    return bid, ask


def _apply_q_max_limit(quote: Quote, inventory: float, q_max: float) -> Quote:
    bid, ask = quote.bid, quote.ask

    if inventory >= q_max:
        bid = None
    if inventory <= -q_max:
        ask = None

    return Quote(
        bid=bid,
        ask=ask,
        reference_price=quote.reference_price,
        inventory_penalty=quote.inventory_penalty,
    )


class NaiveStrategy:
    name = "naive"

    def __init__(self, half_spread: float = 0.02, tick_size: float = 0.01):
        if half_spread <= 0:
            raise ValueError("half_spread muss positiv sein.")
        if tick_size <= 0:
            raise ValueError("tick_size muss positiv sein.")

        self.half_spread = float(half_spread)
        self.tick_size = float(tick_size)

    def get_quote(
        self,
        mid: float,
        spread: float,
        imbalance_l1: float,
        sigma_ewma_usd: float,
        time_to_close: float,
        inventory: float,
    ) -> Quote:
        mid = float(mid)

        if not math.isfinite(mid):
            raise ValueError("mid muss ein endlicher Wert sein.")

        raw_bid = mid - self.half_spread
        raw_ask = mid + self.half_spread

        bid = floor_to_tick(raw_bid, self.tick_size)
        ask = ceil_to_tick(raw_ask, self.tick_size)
        bid, ask = enforce_valid_quote(bid, ask, self.tick_size)

        return Quote(bid=bid, ask=ask, reference_price=mid, inventory_penalty=0.0)


class ASLiteralStrategy:
    name = "as_literal"

    def __init__(
        self,
        gamma: float,
        k: float,
        tick_size: float = 0.01,
        freeze_sigma: Optional[float] = None,
        freeze_horizon: Optional[float] = None,
        name: Optional[str] = None,
    ):
        if gamma <= 0:
            raise ValueError("gamma muss positiv sein.")
        if k <= 0:
            raise ValueError("k muss positiv sein.")
        if tick_size <= 0:
            raise ValueError("tick_size muss positiv sein.")
        if freeze_sigma is not None and freeze_sigma <= 0:
            raise ValueError("freeze_sigma muss positiv sein, falls gesetzt.")
        if freeze_horizon is not None and not 0.0 <= freeze_horizon <= 1.0:
            raise ValueError("freeze_horizon muss in [0, 1] liegen, falls gesetzt.")

        self.gamma = float(gamma)
        self.k = float(k)
        self.tick_size = float(tick_size)
        self.freeze_sigma = freeze_sigma
        self.freeze_horizon = freeze_horizon

        if name is not None:
            self.name = name

    def get_quote(
        self,
        mid: float,
        spread: float,
        imbalance_l1: float,
        sigma_ewma_usd: float,
        time_to_close: float,
        inventory: float,
    ) -> Quote:
        mid = float(mid)
        inventory = float(inventory)

        if not math.isfinite(mid):
            raise ValueError("mid muss ein endlicher Wert sein.")
        if not math.isfinite(inventory):
            raise ValueError("inventory muss ein endlicher Wert sein.")

        sigma = self.freeze_sigma if self.freeze_sigma is not None else float(sigma_ewma_usd)
        horizon = self.freeze_horizon if self.freeze_horizon is not None else float(time_to_close)

        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError(f"sigma_ewma_usd muss endlich und nicht-negativ sein, war {sigma}.")
        if not math.isfinite(horizon) or not 0.0 <= horizon <= 1.0:
            raise ValueError(f"time_to_close muss in [0, 1] liegen, war {horizon}.")

        sigma2 = sigma ** 2

        reservation_price = mid - inventory * self.gamma * sigma2 * horizon

        risk_term = self.gamma * sigma2 * horizon
        liquidity_term = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        half_spread = (risk_term + liquidity_term) / 2.0

        raw_bid = reservation_price - half_spread
        raw_ask = reservation_price + half_spread

        bid = floor_to_tick(raw_bid, self.tick_size)
        ask = ceil_to_tick(raw_ask, self.tick_size)
        bid, ask = enforce_valid_quote(bid, ask, self.tick_size)

        return Quote(
            bid=bid,
            ask=ask,
            reference_price=reservation_price,
            inventory_penalty=risk_term,
        )


class HardLimitStrategy:
    name = "hard_limit"

    def __init__(
        self,
        half_spread: float = 0.02,
        q_max: float = 1000.0,
        tick_size: float = 0.01,
    ):
        if q_max <= 0:
            raise ValueError("q_max muss positiv sein.")

        self.q_max = float(q_max)
        self.tick_size = float(tick_size)
        self._naive = NaiveStrategy(half_spread=half_spread, tick_size=tick_size)

    def get_quote(
        self,
        mid: float,
        spread: float,
        imbalance_l1: float,
        sigma_ewma_usd: float,
        time_to_close: float,
        inventory: float,
    ) -> Quote:
        quote = self._naive.get_quote(
            mid=mid,
            spread=spread,
            imbalance_l1=imbalance_l1,
            sigma_ewma_usd=sigma_ewma_usd,
            time_to_close=time_to_close,
            inventory=inventory,
        )
        return _apply_q_max_limit(quote, inventory, self.q_max)


class HybridStrategy:
    name = "hybrid"

    def __init__(
        self,
        gamma: float,
        k: float,
        q_max: float,
        tick_size: float = 0.01,
        freeze_sigma: Optional[float] = None,
        freeze_horizon: Optional[float] = None,
        name: Optional[str] = None,
    ):
        if q_max <= 0:
            raise ValueError("q_max muss positiv sein.")

        self.q_max = float(q_max)
        self.tick_size = float(tick_size)
        self._as_literal = ASLiteralStrategy(
            gamma=gamma,
            k=k,
            tick_size=tick_size,
            freeze_sigma=freeze_sigma,
            freeze_horizon=freeze_horizon,
        )

        if name is not None:
            self.name = name

    def get_quote(
        self,
        mid: float,
        spread: float,
        imbalance_l1: float,
        sigma_ewma_usd: float,
        time_to_close: float,
        inventory: float,
    ) -> Quote:
        quote = self._as_literal.get_quote(
            mid=mid,
            spread=spread,
            imbalance_l1=imbalance_l1,
            sigma_ewma_usd=sigma_ewma_usd,
            time_to_close=time_to_close,
            inventory=inventory,
        )
        return _apply_q_max_limit(quote, inventory, self.q_max)