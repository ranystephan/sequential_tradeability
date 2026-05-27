import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_wrds_stock_level.py"
SPEC = importlib.util.spec_from_file_location("download_wrds_stock_level", SCRIPT_PATH)
assert SPEC is not None
wrds_download = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["download_wrds_stock_level"] = wrds_download
SPEC.loader.exec_module(wrds_download)


def test_wrds_signal_panel_and_returns_helpers_are_consistent() -> None:
    dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    rows = []
    for permno in range(1, 61):
        for month_index, date in enumerate(dates):
            rows.append(
                {
                    "permno": permno,
                    "date": date,
                    "ret_adj": 0.01 * np.sin(permno + month_index),
                    "ret_next_month": 0.01 * np.cos(permno + month_index),
                    "me": 100.0 + permno,
                    "prc": 10.0,
                    "dollar_volume": 1_000.0,
                    "spread_proxy": 0.01,
                    "exchange_code": 1,
                    "share_code": 10,
                    "sic": 1000,
                    "mom12m": permno / 100.0,
                }
            )
    crsp = pd.DataFrame(rows)
    accounting = pd.DataFrame(
        {
            "permno": range(1, 61),
            "signal_date": [pd.Timestamp("2020-06-30")] * 60,
            "book_equity": np.arange(1, 61) * 10.0,
            "operprof": np.arange(1, 61) / 100.0,
        }
    )

    panel = wrds_download.build_signal_panel(crsp, accounting)
    returns = wrds_download.build_signal_returns(panel)

    assert set(panel["signal"]) == {"BM", "Mom12m", "OperProf"}
    assert set(returns["signal"]) == {"BM", "Mom12m", "OperProf"}
    assert (returns["n_long"] > 0).all()
    assert (returns["n_short"] > 0).all()
    assert np.allclose(
        returns["net_return_percent"],
        returns["gross_return_percent"] - returns["cost_percent"],
    )
