"""Download and build stock-level WRDS signal returns for the MSE342 project.

This script uses the official PyWRDS package. It prompts for WRDS credentials through
wrds.Connection(), downloads real CRSP/Compustat data, and constructs monthly long-short
portfolio returns for a small set of signals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START = "1968-01-01"
DEFAULT_END = "2025-12-31"


@dataclass(frozen=True)
class OutputPaths:
    raw_dir: Path
    crsp_monthly: Path
    crsp_delist: Path
    compustat_funda: Path
    ccm_link: Path
    signal_panel: Path
    signal_returns: Path


def main() -> None:
    args = parse_args()
    paths = output_paths(args.output_dir)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.signal_panel.parent.mkdir(parents=True, exist_ok=True)

    print("Connecting to WRDS. Enter your WRDS username/password when prompted.")
    with wrds.Connection() as db:
        crsp = download_crsp_monthly(db, args.start, args.end)
        delist = download_crsp_delist(db, args.start, args.end)
        comp = download_compustat_funda(db, args.start, args.end)
        link = download_ccm_link(db)

    write_frame(crsp, paths.crsp_monthly)
    write_frame(delist, paths.crsp_delist)
    write_frame(comp, paths.compustat_funda)
    write_frame(link, paths.ccm_link)

    crsp = prepare_crsp(crsp, delist)
    accounting = prepare_accounting_signals(comp, link)
    panel = build_signal_panel(crsp, accounting)
    returns = build_signal_returns(panel)

    write_frame(panel, paths.signal_panel)
    returns.to_csv(paths.signal_returns, index=False)

    print(f"Wrote CRSP monthly rows: {len(crsp):,}")
    print(f"Wrote stock-signal panel rows: {len(panel):,} -> {paths.signal_panel}")
    print(f"Wrote signal return rows: {len(returns):,} -> {paths.signal_returns}")
    print(returns.groupby("signal")[["gross_return_percent", "net_return_percent"]].describe())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START, help="CRSP start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END, help="CRSP end date, YYYY-MM-DD.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Repo data directory.",
    )
    return parser.parse_args()


def output_paths(output_dir: Path) -> OutputPaths:
    raw_dir = output_dir / "raw" / "wrds"
    processed_dir = output_dir / "processed"
    return OutputPaths(
        raw_dir=raw_dir,
        crsp_monthly=raw_dir / "crsp_monthly.csv.gz",
        crsp_delist=raw_dir / "crsp_delist.csv.gz",
        compustat_funda=raw_dir / "compustat_funda.csv.gz",
        ccm_link=raw_dir / "ccm_link.csv.gz",
        signal_panel=processed_dir / "wrds_signal_panel.csv.gz",
        signal_returns=processed_dir / "wrds_signal_returns.csv",
    )


def download_crsp_monthly(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    sql = f"""
        select
            msf.permno,
            msf.date,
            msf.ret,
            msf.retx,
            msf.prc,
            msf.shrout,
            msf.vol,
            msf.bidlo,
            msf.askhi,
            names.shrcd as share_code,
            names.exchcd as exchange_code,
            names.siccd as sic
        from crsp.msf as msf
        inner join crsp.msenames as names
            on msf.permno = names.permno
            and names.namedt <= msf.date
            and msf.date <= names.nameendt
        where msf.date between '{start}' and '{end}'
            and names.shrcd in (10, 11)
            and names.exchcd in (1, 2, 3)
    """
    print("Downloading CRSP monthly stock data...")
    return db.raw_sql(sql, date_cols=["date"])


def download_crsp_delist(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    sql = f"""
        select permno, dlstdt as date, dlret
        from crsp.msedelist
        where dlstdt between '{start}' and '{end}'
    """
    print("Downloading CRSP delisting returns...")
    return db.raw_sql(sql, date_cols=["date"])


def download_compustat_funda(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    start_year = pd.Timestamp(start).year - 3
    sql = f"""
        select
            gvkey,
            datadate,
            fyear,
            at,
            lt,
            seq,
            ceq,
            txditc,
            pstkrv,
            pstkl,
            pstk,
            revt,
            cogs,
            xsga,
            xint
        from comp.funda
        where indfmt = 'INDL'
            and datafmt = 'STD'
            and popsrc = 'D'
            and consol = 'C'
            and datadate between '{start_year}-01-01' and '{end}'
    """
    print("Downloading Compustat annual fundamentals...")
    return db.raw_sql(sql, date_cols=["datadate"])


def download_ccm_link(db: wrds.Connection) -> pd.DataFrame:
    sql = """
        select
            gvkey,
            lpermno as permno,
            linkdt,
            linkenddt,
            linktype,
            linkprim
        from crsp.ccmxpf_linktable
        where lpermno is not null
            and linktype in ('LU', 'LC')
            and linkprim in ('P', 'C')
    """
    print("Downloading CRSP/Compustat link table...")
    return db.raw_sql(sql, date_cols=["linkdt", "linkenddt"])


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def prepare_crsp(crsp: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    crsp = crsp.copy()
    crsp["date"] = month_end(crsp["date"])
    delist = delist.copy()
    delist["date"] = month_end(delist["date"])
    crsp = crsp.merge(delist, on=["permno", "date"], how="left")

    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp["dlret"] = pd.to_numeric(crsp["dlret"], errors="coerce")
    crsp["ret_adj"] = (1.0 + crsp["ret"].fillna(0.0)) * (1.0 + crsp["dlret"].fillna(0.0)) - 1.0
    crsp.loc[crsp["ret"].isna() & crsp["dlret"].isna(), "ret_adj"] = np.nan
    crsp["me"] = crsp["prc"].abs() * crsp["shrout"]
    crsp["dollar_volume"] = crsp["prc"].abs() * crsp["vol"]
    crsp["spread_proxy"] = spread_proxy(crsp)

    crsp = crsp.sort_values(["permno", "date"])
    crsp["ret_next_month"] = crsp.groupby("permno", sort=False)["ret_adj"].shift(-1)
    crsp["mom12m"] = momentum_12_2(crsp)
    return crsp


def month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp("M")


def spread_proxy(crsp: pd.DataFrame) -> pd.Series:
    midpoint = (crsp["askhi"] + crsp["bidlo"]) / 2.0
    spread = (crsp["askhi"] - crsp["bidlo"]) / midpoint
    spread = spread.where((crsp["askhi"] > 0.0) & (crsp["bidlo"] > 0.0) & (midpoint > 0.0))
    monthly_cap = spread.groupby(month_end(crsp["date"])).transform(lambda x: x.quantile(0.99))
    return spread.clip(lower=0.0).clip(upper=monthly_cap)


def momentum_12_2(crsp: pd.DataFrame) -> pd.Series:
    safe_ret = crsp["ret_adj"].clip(lower=-0.999)
    log_ret = np.log1p(safe_ret)
    rolling = (
        log_ret.groupby(crsp["permno"], sort=False)
        .rolling(window=11, min_periods=8)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return np.expm1(rolling.groupby(crsp["permno"], sort=False).shift(2))


def prepare_accounting_signals(comp: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    comp = comp.copy()
    comp["preferred_stock"] = (
        comp["pstkrv"].combine_first(comp["pstkl"]).combine_first(comp["pstk"])
    )
    comp["preferred_stock"] = comp["preferred_stock"].fillna(0.0)
    comp["book_equity"] = (
        comp["seq"].combine_first(comp["ceq"]).combine_first(comp["at"] - comp["lt"])
        + comp["txditc"].fillna(0.0)
        - comp["preferred_stock"]
    )
    operating_profit = (
        comp["revt"]
        - comp["cogs"]
        - comp["xsga"].fillna(0.0)
        - comp["xint"].fillna(0.0)
    )
    comp["operprof"] = operating_profit / comp["book_equity"]
    comp = comp[(comp["book_equity"] > 0.0) & np.isfinite(comp["operprof"])]
    comp["signal_date"] = month_end(pd.to_datetime(comp["datadate"]) + pd.DateOffset(months=6))

    link = link.copy()
    link["linkenddt"] = link["linkenddt"].fillna(pd.Timestamp("today").normalize())
    linked = comp.merge(link, on="gvkey", how="inner")
    linked = linked[
        (linked["datadate"] >= linked["linkdt"]) & (linked["datadate"] <= linked["linkenddt"])
    ]
    return linked[["permno", "signal_date", "book_equity", "operprof"]].dropna()


def build_signal_panel(crsp: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    print("Building stock-level signal panel...")
    base_columns = [
        "date",
        "permno",
        "ret_next_month",
        "me",
        "prc",
        "dollar_volume",
        "spread_proxy",
        "exchange_code",
        "share_code",
        "sic",
    ]

    mom = crsp[base_columns + ["mom12m"]].rename(columns={"mom12m": "raw_signal"})
    mom["signal"] = "Mom12m"

    crsp_for_asof = crsp[base_columns].sort_values(["permno", "date"])
    merged = merge_accounting_asof(crsp_for_asof, accounting)
    bm = merged.copy()
    bm["raw_signal"] = bm["book_equity"] / bm["me"]
    bm["signal"] = "BM"
    op = merged.copy()
    op["raw_signal"] = op["operprof"]
    op["signal"] = "OperProf"

    panel = pd.concat([mom, bm, op], ignore_index=True)
    panel = panel[
        np.isfinite(panel["raw_signal"])
        & np.isfinite(panel["ret_next_month"])
        & (panel["me"] > 0.0)
    ]
    keep = [
        "date",
        "signal",
        "permno",
        "raw_signal",
        "ret_next_month",
        "me",
        "prc",
        "dollar_volume",
        "spread_proxy",
        "exchange_code",
        "share_code",
        "sic",
    ]
    return panel[keep].sort_values(["signal", "date", "permno"])


def merge_accounting_asof(crsp: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    accounting_groups = {
        permno: group.sort_values("signal_date").drop(columns=["permno"])
        for permno, group in accounting.groupby("permno", sort=False)
    }
    for permno, left in crsp.groupby("permno", sort=False):
        right = accounting_groups.get(permno)
        if right is None:
            continue
        pieces.append(
            pd.merge_asof(
                left.sort_values("date"),
                right,
                left_on="date",
                right_on="signal_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    if not pieces:
        raise ValueError("No CRSP rows matched the Compustat/CCM accounting signal table")
    return pd.concat(pieces, ignore_index=True)


def build_signal_returns(panel: pd.DataFrame) -> pd.DataFrame:
    print("Building monthly long-short gross and net returns...")
    rows = []
    previous_weights: dict[str, pd.Series] = {}
    for (signal, date), group in panel.groupby(["signal", "date"], sort=True):
        group = group.dropna(subset=["raw_signal", "ret_next_month", "me"])
        if group["permno"].nunique() < 50:
            continue
        low_cut = group["raw_signal"].quantile(0.10)
        high_cut = group["raw_signal"].quantile(0.90)
        long_leg = group[group["raw_signal"] >= high_cut]
        short_leg = group[group["raw_signal"] <= low_cut]
        if long_leg.empty or short_leg.empty:
            continue

        long_weights = long_leg["me"] / long_leg["me"].sum()
        short_weights = -short_leg["me"] / short_leg["me"].sum()
        weights = pd.concat(
            [
                pd.Series(long_weights.to_numpy(), index=long_leg["permno"].astype(int)),
                pd.Series(short_weights.to_numpy(), index=short_leg["permno"].astype(int)),
            ]
        )
        returns = pd.concat(
            [
                pd.Series(
                    long_leg["ret_next_month"].to_numpy(),
                    index=long_leg["permno"].astype(int),
                ),
                pd.Series(
                    short_leg["ret_next_month"].to_numpy(),
                    index=short_leg["permno"].astype(int),
                ),
            ]
        )
        spreads = pd.concat(
            [
                pd.Series(
                    long_leg["spread_proxy"].to_numpy(),
                    index=long_leg["permno"].astype(int),
                ),
                pd.Series(
                    short_leg["spread_proxy"].to_numpy(),
                    index=short_leg["permno"].astype(int),
                ),
            ]
        ).fillna(0.0)

        gross_return = float((weights * returns).sum())
        previous = previous_weights.get(signal, pd.Series(dtype=float))
        all_names = weights.index.union(previous.index)
        trades = weights.reindex(all_names, fill_value=0.0) - previous.reindex(
            all_names,
            fill_value=0.0,
        )
        cost_spreads = spreads.reindex(
            all_names,
            fill_value=spreads.median() if spreads.size else 0.0,
        )
        cost = float((trades.abs() * cost_spreads / 2.0).sum())
        previous_weights[signal] = weights

        rows.append(
            {
                "date": date,
                "signal": signal,
                "n_stocks": int(group["permno"].nunique()),
                "n_long": int(long_leg["permno"].nunique()),
                "n_short": int(short_leg["permno"].nunique()),
                "gross_return_percent": 100.0 * gross_return,
                "net_return_percent": 100.0 * (gross_return - cost),
                "turnover": float(trades.abs().sum()),
                "cost_percent": 100.0 * cost,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
