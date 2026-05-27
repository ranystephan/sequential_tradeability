# WRDS Data Contract for Stock-Level Extension

The current OSAP experiment uses prebuilt anomaly long-short returns. The stock-level
extension should measure implementable signal admission with explicit turnover and
transaction-cost proxies. The cleanest next data pull is a monthly stock-signal panel
with one row per signal, month, and stock.

## Required Output File

Preferred path:

`data/processed/wrds_signal_panel.csv`

The current downloader writes the compressed version:

`data/processed/wrds_signal_panel.csv.gz`

Run it from the repo root with:

```bash
uv run python scripts/download_wrds_stock_level.py --start 1968-01-01 --end 2025-12-31
```

It uses the official `wrds` Python package and will prompt for WRDS credentials. It also
writes:

- `data/raw/wrds/crsp_monthly.csv.gz`
- `data/raw/wrds/crsp_delist.csv.gz`
- `data/raw/wrds/compustat_funda.csv.gz`
- `data/raw/wrds/ccm_link.csv.gz`
- `data/processed/wrds_signal_returns.csv`

Required columns:

| column | type | meaning |
| --- | --- | --- |
| `date` | YYYY-MM-DD | portfolio formation month end |
| `signal` | string | signal name, e.g. `BM`, `Mom12m`, `OperProf` |
| `permno` | integer | CRSP permanent identifier |
| `raw_signal` | float | signal value known at formation date |
| `ret_next_month` | float | CRSP next-month return including delisting return when available |
| `me` | float | market equity at formation, in dollars or thousands consistently |
| `price` | float | absolute CRSP price at formation |
| `dollar_volume` | float | monthly dollar volume proxy |
| `spread_proxy` | float | proportional round-trip or half-spread cost proxy |
| `exchange_code` | integer | CRSP exchange code |
| `share_code` | integer | CRSP share code |

Optional but useful:

| column | type | meaning |
| --- | --- | --- |
| `sic` | integer | industry code |
| `gvkey` | string | Compustat identifier from CCM link |
| `market_ret` | float | market return for residualization |
| `rf` | float | risk-free rate |

## WRDS Sources

Use CRSP monthly stock data for returns, prices, shares, exchange/share codes, and
delisting returns:

- `crsp.msf`: `permno`, `date`, `ret`, `prc`, `vol`, `shrout`, `bidlo`, `askhi`
- `crsp.msedelist`: `permno`, `dlstdt`, `dlret`
- `crsp.msenames`: `permno`, `namedt`, `nameendt`, `shrcd`, `exchcd`, `siccd`

Use CRSP/Compustat Merged only for accounting signals:

- `crsp.ccmxpf_linktable`: `gvkey`, `lpermno`, `linkdt`, `linkenddt`, `linktype`, `linkprim`
- `comp.funda`: annual accounting variables with standard reporting lags

## Filters

Use ordinary common stocks only:

- `share_code` in `{10, 11}`
- NYSE/AMEX/NASDAQ exchange codes when available
- positive price and market equity
- enough return history for the chosen signal construction

Avoid look-ahead:

- Monthly market signals can be formed using information available at month end.
- Annual Compustat signals should use at least a six-month reporting lag after fiscal
  year end unless the exact report date is used.

## Transaction-Cost Proxy

If daily bid/ask data are not available, use a conservative monthly proxy:

```text
spread_proxy = clip((askhi - bidlo) / ((askhi + bidlo) / 2), lower=0, upper=p99)
```

Then estimate portfolio implementation cost from turnover:

```text
net_return_t = gross_return_t - sum_i abs(weight_i,t - weight_i,t-) * spread_proxy_i,t / 2
```

This is not a perfect trading-cost model, but it is materially better than treating
prebuilt long-short returns as frictionless.

## Portfolio Construction Target

For each signal and month:

1. Rank eligible stocks by `raw_signal`.
2. Form long top decile and short bottom decile portfolios.
3. Use value weights within each side unless the experiment explicitly compares equal
   weights.
4. Compute gross next-month long-short return.
5. Compute turnover from changes in security weights.
6. Compute net long-short return after spread-based cost.

The existing Bayesian admission pipeline can then be rerun on gross and net returns
side by side.
