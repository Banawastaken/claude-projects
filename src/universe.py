"""The FundedNext CFD universe, mapped to Dukascopy symbols.

Contract sizes are FundedNext's published figures: forex 100,000, indices 10,
metals 100 (silver 5,000), crypto varies. Commission is theirs too: $5 per lot
on forex and oil, 0.0016% of notional on metals, 0.04% on crypto, nothing
stated for indices so they are treated as spread-only.

`usd_per_point` is the contract size already converted to USD per 1.00 point of
price movement, using an approximate constant FX rate for instruments quoted in
something other than dollars. That approximation is nearly free here: position
size is set from a risk percentage, so the FX factor cancels out of every R
multiple and only reaches the result through commission and the minimum-lot
check.

`price_range` is the plausible band for the instrument over 2025-2026 and is
used to pick the right Dukascopy price divisor automatically, since it differs
per symbol (100,000 for five-decimal FX, 1,000 for gold and the indices).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Instrument:
    fn_name: str        # FundedNext's symbol
    duka: str           # Dukascopy datafeed symbol
    asset_class: str
    usd_per_point: float
    price_range: tuple[float, float]
    commission_usd_per_lot: float = 0.0   # flat, e.g. forex and oil
    commission_pct_notional: float = 0.0  # e.g. metals, crypto
    leverage: float = 25.0
    min_lot: float = 0.01


FX = 100_000.0

UNIVERSE = [
    # ---- forex majors and crosses (contract 100,000, $5/lot) --------------
    Instrument("EURUSD", "EURUSD", "forex", FX, (0.95, 1.35), 5.0, leverage=30),
    Instrument("GBPUSD", "GBPUSD", "forex", FX, (1.10, 1.50), 5.0, leverage=30),
    Instrument("AUDUSD", "AUDUSD", "forex", FX, (0.55, 0.80), 5.0, leverage=30),
    Instrument("NZDUSD", "NZDUSD", "forex", FX, (0.50, 0.75), 5.0, leverage=30),
    # USD-base pairs: a point is worth 100,000/price dollars, but using the
    # constant below only shifts lot size, which the risk sizing absorbs.
    Instrument("USDJPY", "USDJPY", "forex", FX / 150.0, (120.0, 180.0), 5.0, leverage=30),
    Instrument("USDCAD", "USDCAD", "forex", FX / 1.38, (1.25, 1.55), 5.0, leverage=30),
    Instrument("USDCHF", "USDCHF", "forex", FX / 0.85, (0.75, 1.00), 5.0, leverage=30),
    Instrument("EURJPY", "EURJPY", "forex", FX / 150.0, (140.0, 200.0), 5.0, leverage=30),
    Instrument("GBPJPY", "GBPJPY", "forex", FX / 150.0, (170.0, 230.0), 5.0, leverage=30),
    Instrument("EURGBP", "EURGBP", "forex", FX * 1.30, (0.78, 0.95), 5.0, leverage=30),
    Instrument("AUDJPY", "AUDJPY", "forex", FX / 150.0, (85.0, 115.0), 5.0, leverage=30),
    Instrument("EURAUD", "EURAUD", "forex", FX * 0.65, (1.55, 1.95), 5.0, leverage=30),

    # ---- metals (contract 100, silver 5,000; 0.0016% of notional) ---------
    Instrument("XAUUSD", "XAUUSD", "metal", 100.0, (2400.0, 5800.0),
               commission_pct_notional=0.000016),
    Instrument("XAGUSD", "XAGUSD", "metal", 5000.0, (20.0, 90.0),
               commission_pct_notional=0.000016),
    # Platinum is listed by FundedNext but Dukascopy publishes no candle data
    # for XPTUSD, so it cannot be tested here.

    # ---- indices (contract 10 per point, spread only) ---------------------
    Instrument("SPX500", "USA500IDXUSD", "index", 10.0, (4500.0, 9000.0)),
    Instrument("NDX100", "USATECHIDXUSD", "index", 10.0, (15000.0, 35000.0)),
    Instrument("US30", "USA30IDXUSD", "index", 10.0, (35000.0, 70000.0)),
    Instrument("US2000", "USSC2000IDXUSD", "index", 10.0, (1800.0, 4000.0)),
    Instrument("GER30", "DEUIDXEUR", "index", 10.0 * 1.10, (16000.0, 32000.0)),
    Instrument("UK100", "GBRIDXGBP", "index", 10.0 * 1.30, (7000.0, 13000.0)),
    Instrument("JP225", "JPNIDXJPY", "index", 10.0 / 150.0, (30000.0, 65000.0)),
    Instrument("FRA40", "FRAIDXEUR", "index", 10.0 * 1.10, (6500.0, 11000.0)),
    Instrument("AUS200", "AUSIDXAUD", "index", 10.0 * 0.65, (7000.0, 12000.0)),
    Instrument("HK50", "HKGIDXHKD", "index", 10.0 / 7.8, (15000.0, 32000.0)),
    Instrument("EUSTX50", "EUSIDXEUR", "index", 10.0 * 1.10, (4200.0, 7500.0)),

    # ---- energy (grouped with commodities at contract 100; $5/lot) --------
    Instrument("USOUSD", "LIGHTCMDUSD", "energy", 100.0, (40.0, 110.0), 5.0),
    Instrument("UKOUSD", "BRENTCMDUSD", "energy", 100.0, (45.0, 115.0), 5.0),

    # ---- crypto (0.04% of notional, 1:1 leverage) -------------------------
    Instrument("BTCUSD", "BTCUSD", "crypto", 1.0, (40000.0, 250000.0),
               commission_pct_notional=0.0004, leverage=1),
    Instrument("ETHUSD", "ETHUSD", "crypto", 1.0, (1000.0, 9000.0),
               commission_pct_notional=0.0004, leverage=1),
]

BY_FN = {i.fn_name: i for i in UNIVERSE}
BY_DUKA = {i.duka: i for i in UNIVERSE}


def commission_per_lot(inst: Instrument, price: float) -> float:
    """Round-turn commission in USD for one lot at the given price."""
    if inst.commission_pct_notional:
        return inst.commission_pct_notional * inst.usd_per_point * price
    return inst.commission_usd_per_lot


if __name__ == "__main__":
    print(f"{'FundedNext':10s} {'Dukascopy':16s} {'class':7s} {'usd/pt':>10s} {'comm/lot':>9s}")
    for i in UNIVERSE:
        mid = (i.price_range[0] + i.price_range[1]) / 2
        print(f"{i.fn_name:10s} {i.duka:16s} {i.asset_class:7s} {i.usd_per_point:10.2f} "
              f"{commission_per_lot(i, mid):9.2f}")
    print(f"\n{len(UNIVERSE)} instruments")
