# Inventory-Aware Market Making: Avellaneda-Stoikov Backtest on NVDA

An event-driven backtest of the Avellaneda-Stoikov (2008) market making model on Nasdaq order book data. The project compares a naive symmetric quoting strategy with inventory-aware strategies and decomposes the profit and loss into spread capture, adverse selection (market impact) and inventory effects.

## Data

- Nasdaq TotalView-ITCH, MBP-10 schema, NVDA
- Backtest: 21 trading days (1 to 30 June 2026), 124 million events, 980,554 bins at 500 ms resolution during regular trading hours
- Calibration: the last 5 trading days of May 2026, so the model parameters are not fitted on the backtest period
- Validation per trading day: no crossed books, no timestamp violations, 274 missing top-of-book values in 124 million rows
- Raw data is not included in this repository (licensed market data)

## Approach

1. **Parsing and features** (`parsing.py`, `features.py`): raw daily files are converted to one parquet file and processed day by day to keep memory low. Features per 500 ms bin: mid price, spread, order book imbalance, traded volume on each side, a causal EWMA volatility estimate (60-minute half-life) and the time to close.
2. **Fill model** (`fill_model.py`): quotes live for one bin. For each quote the traded volume at or through its price in the next bin is read from a volume ladder. A conservative rule requires that the queue ahead is traded through first and allows a 5% participation of the remaining volume, capped at the order size of 20 shares.
3. **Strategies** (`strategies.py`):
   - Naive: fixed symmetric half-spread of 5 cents
   - Hard limit: naive quotes plus a position cap of 300 shares
   - Avellaneda-Stoikov: reservation price shifted by inventory, risk and liquidity terms, with γ in {1e-5, 3e-5, 1e-4}
   - Hybrid: Avellaneda-Stoikov plus the same position cap
4. **Calibration** (`calibration.py`): the order arrival parameter k is estimated by regressing log trade intensity on the distance from the touch, with acceptance criteria (R², confidence interval, curvature test). The result is k = 21.4. The module also runs sweeps over half-spread, γ and the position cap.
5. **Simulation and evaluation** (`simulation.py`, `analysis.py`, `metrics.py`): quotes are clamped so they stay passive, positions are liquidated against the visible book at the end of each day, and results are reported as P&L decomposition, inventory statistics, minute-level VaR and expected shortfall and intraday drawdown.

## Key results

21 trading days, order size 20 shares, conservative fill rule:

| Strategy | P&L (USD) | Mean abs. inventory | Max abs. inventory | Minute VaR 99% |
|---|---|---|---|---|
| Naive | -25,197 | 602 | 2,356 | -535 |
| Hard limit | -6,893 | 183 | 319 | -147 |
| Avellaneda-Stoikov, γ = 1e-5 | -6,591 | 51 | 786 | -51 |
| Hybrid, γ = 1e-5 | -5,994 | 50 | 319 | -50 |
| Avellaneda-Stoikov, γ = 1e-4 | -8,864 | 16 | 598 | -23 |
| Hybrid, γ = 1e-4 | -8,850 | 16 | 318 | -23 |

- **Inventory skewing controls risk.** Against the naive strategy the average absolute position falls from about 600 to 50 shares at γ = 1e-5 and to 16 shares at γ = 1e-4. The worst intraday drawdown falls from -14,653 to -1,023 USD for the hybrid strategy at γ = 1e-5.
- **The hybrid strategy cuts the tail.** The position cap binds in under 0.4% of the time at γ = 1e-5, but it reduces the maximum position from 786 to 319 shares and halves the dispersion of the final position.
- **The best P&L comes from the hybrid with the smallest γ.** The loss is about 76% lower than for the naive strategy (-5,994 vs. -25,197 USD).
- **A larger γ trades profit for safety.** Quoted spread capture falls from 5.2 to 3.3 cents per share and the realised result per share falls from -1.8 to -2.0 cents, while inventory risk shrinks further.
- **Adverse selection dominates.** For every strategy the spread captured when quoting (3 to 5 cents per share) is smaller than the adverse selection cost (5 to 7 cents per share), so the realised result per share is negative.

## Limitations

All strategies lose money under the conservative fill rule, so the results show how well the strategies control risk, not that market making is profitable. Fills are simulated from observed trades. Queue dynamics, latency, fees and rebates, and the price impact of the strategy's own orders are not modelled. The empirical fill intensities only approximate the exponential form assumed by the model, so k is a local estimate. The sample covers one month and one stock.

## Repository

```
notebooks/
  01_data_processing.ipynb   parsing and validation output, feature overview
  02_test_simulation.ipynb   strategy comparison, risk metrics, plots
src/
  parsing.py  validation.py  features.py  fill_model.py
  strategies.py  simulation.py  calibration.py  analysis.py  metrics.py  config.py
results/                     calibration diagnostics, plots
```

Python (pandas, NumPy, statsmodels, polars, pyarrow, matplotlib). Parameters (k, γ grid, position cap, order size) are set in `src/config.py`. Place the raw files in `data/`, then run parsing, features and the simulation notebook in that order.
