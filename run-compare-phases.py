"""Phase 1-3 개선 전/후 비교 백테스트 (Phase 4).

11종목 × 5m × 90일 Binance 데이터로 baseline vs improved 성적 비교.

- Baseline: Phase 1-3 필터 전부 비활성 (초기 구축 시점 파라미터 모사)
- Improved: config.yaml 기본값 (RSI≥55, vol≥1.5x, MTF, ATR≤5%, TP+5%, time_stop=48봉)

완료 기준 (판정):
  - 승률 ≥ 30%
  - R/R (평균익절/평균손절) ≥ 2.0
  - 기대값 > +0.2% per trade
  - 종목 편중 ≤ 40% (단일 종목 거래 비중)

산출물:
  - 콘솔 출력: 종목별 Baseline vs Improved 비교표
  - data/phase_comparison_2026-04-20.csv: 전체 결과 CSV
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestEngine
from data.binance_collector import load_binance_data
from strategy.tucker_v3 import TuckerStrategyV3
from utils.logger import logger


# ────────────────────────────────────────────────────────────────────
# 11종목 설정: (Upbit 마켓명, Binance 심볼, EMA 기간, EMA 근접 %)
# config.yaml에서 지정된 11종목과 동일
# ────────────────────────────────────────────────────────────────────
MARKETS: list[tuple[str, str, int, float]] = [
    ("KRW-BTC",  "BTCUSDT",  9,  0.5),
    ("KRW-ETH",  "ETHUSDT",  21, 0.3),
    ("KRW-IOST", "IOSTUSDT", 21, 0.3),
    ("KRW-RED",  "REDUSDT",  9,  0.5),
    ("KRW-SOL",  "SOLUSDT",  21, 0.3),
    ("KRW-DOGE", "DOGEUSDT", 9,  0.5),
    ("KRW-XRP",  "XRPUSDT",  21, 0.3),
    ("KRW-TREE", "TREEUSDT", 9,  0.5),
    ("KRW-TAO",  "TAOUSDT",  21, 0.3),
    ("KRW-HOLO", "HOLOUSDT", 21, 0.3),
    ("KRW-XPL",  "XPLUSDT",  21, 0.3),
]

# 공통 기본 파라미터 (변경 없음)
COMMON = dict(
    reset_hour_utc=0,
    swing_lookback=5,
    swing_min_distance_pct=1.0,
    vwap_chop_lookback=10,
    vwap_chop_cross_threshold=4,
    vp_num_bins=20,
    vp_thin_threshold_pct=30,
    exit_confirm_bars=3,
    cooldown_bars=5,
    atr_period=14,
    atr_stop_multiplier=0,
)

# Baseline: Phase 1-3 필터 전부 비활성
BASELINE_FILTERS = dict(
    rsi_threshold=0.0,
    volume_ratio_threshold=0.0,
    require_mtf_agreement=False,
    atr_max_pct=0.0,
    take_profit_pct=0.0,
    time_stop_bars=0,
)

# Improved: config.yaml 기본값
IMPROVED_FILTERS = dict(
    rsi_period=14,
    rsi_threshold=55.0,
    volume_ratio_lookback=20,
    volume_ratio_threshold=1.5,
    require_mtf_agreement=True,
    atr_max_pct=5.0,
    take_profit_pct=5.0,
    time_stop_bars=48,
)


def load_5m_with_datetime(binance_sym: str) -> pd.DataFrame | None:
    """Binance 5m 데이터를 datetime 인덱스로 로드."""
    try:
        df = load_binance_data(binance_sym, "5m")
    except Exception as e:
        logger.warning(f"{binance_sym} 데이터 로드 실패: {e}")
        return None
    if df is None or df.empty:
        return None
    # datetime 컬럼을 인덱스로
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
    return df


def resample_15m(df5: pd.DataFrame) -> pd.DataFrame:
    """5분봉을 15분봉으로 리샘플링 (MTF 필터용 HTF 생성)."""
    df15 = df5.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df15


def run_backtest(
    df: pd.DataFrame,
    ema_period: int,
    ema_proximity_pct: float,
    filters: dict,
    htf_dfs: dict | None,
    market: str,
) -> dict:
    """단일 백테스트 실행 + 결과 dict 반환."""
    strategy = TuckerStrategyV3(
        ema_period=ema_period,
        ema_proximity_pct=ema_proximity_pct,
        **COMMON,
        **filters,
    )
    engine = BacktestEngine(initial_capital=1_000_000, fee_rate=0.0005, slippage_pct=0.05)
    result = engine.run(df.copy(), strategy, market=market, timeframe="5m", htf_dfs=htf_dfs)
    return {
        "market": market,
        "trades": result.total_trades,
        "win_rate": result.win_rate_pct,
        "return_pct": result.total_return_pct,
        "profit_factor": result.profit_factor,
        "avg_profit": result.avg_profit_pct,
        "avg_loss": result.avg_loss_pct,
        "mdd": result.max_drawdown_pct,
        "sharpe": result.sharpe_ratio,
    }


def summarize(rows: list[dict], label: str) -> dict:
    """종목별 결과를 집계하여 전체 지표 산출."""
    total_trades = sum(r["trades"] for r in rows)
    active_rows = [r for r in rows if r["trades"] > 0]

    # 가중 평균 (거래수 기준)
    def weighted(key: str) -> float:
        if total_trades == 0:
            return 0.0
        return sum(r[key] * r["trades"] for r in active_rows) / total_trades

    # 종목 편중 (최대 거래 종목의 비중)
    if total_trades == 0:
        max_concentration = 0.0
        top_symbol = "-"
    else:
        max_row = max(active_rows, key=lambda r: r["trades"])
        max_concentration = max_row["trades"] / total_trades * 100
        top_symbol = max_row["market"]

    # R/R
    avg_profit = weighted("avg_profit")
    avg_loss = weighted("avg_loss")
    rr = avg_profit / abs(avg_loss) if avg_loss < 0 else float("inf")

    # 기대값 per trade (단순: 전체 종목 수익률 산술평균)
    if total_trades == 0:
        expectancy = 0.0
    else:
        total_pnl_weighted = sum(r["return_pct"] for r in active_rows)
        expectancy = total_pnl_weighted / len(active_rows) if active_rows else 0.0

    return {
        "label": label,
        "total_trades": total_trades,
        "active_markets": len(active_rows),
        "avg_win_rate": weighted("win_rate"),
        "avg_return_pct": sum(r["return_pct"] for r in rows) / len(rows) if rows else 0,
        "avg_profit_pct": avg_profit,
        "avg_loss_pct": avg_loss,
        "rr_ratio": rr,
        "expectancy_pct": expectancy,
        "top_symbol": top_symbol,
        "top_concentration": max_concentration,
    }


def check_criteria(summary: dict) -> dict:
    """완료 기준 4가지 만족 여부 판정."""
    return {
        "win_rate_≥30": summary["avg_win_rate"] >= 30.0,
        "rr_≥2": summary["rr_ratio"] >= 2.0,
        "expectancy_>0.2": summary["expectancy_pct"] > 0.2,
        "concentration_≤40": summary["top_concentration"] <= 40.0,
    }


def main() -> None:
    logger.info("=" * 80)
    logger.info("  Phase 1-3 비교 백테스트 (11종목 × 5m × 90일)")
    logger.info("=" * 80)

    baseline_rows: list[dict] = []
    improved_rows: list[dict] = []
    skipped: list[str] = []

    for upbit_sym, binance_sym, ema, prox in MARKETS:
        logger.info(f"\n─── {upbit_sym} ({binance_sym}, EMA{ema}) ───")
        df5 = load_5m_with_datetime(binance_sym)
        if df5 is None or len(df5) < 500:
            logger.warning(f"  데이터 부족 — 건너뜀")
            skipped.append(upbit_sym)
            continue

        df15 = resample_15m(df5)
        htf_dfs = {"15m": df15}

        # Baseline
        try:
            bl = run_backtest(df5, ema, prox, BASELINE_FILTERS, htf_dfs=None, market=upbit_sym)
            bl["phase"] = "baseline"
            baseline_rows.append(bl)
            logger.info(
                f"  [baseline] 거래={bl['trades']:3d}, 승률={bl['win_rate']:5.1f}%, "
                f"수익={bl['return_pct']:+7.2f}%, PF={bl['profit_factor']:.2f}"
            )
        except Exception as e:
            logger.error(f"  baseline 실패: {e}")

        # Improved
        try:
            im = run_backtest(df5, ema, prox, IMPROVED_FILTERS, htf_dfs=htf_dfs, market=upbit_sym)
            im["phase"] = "improved"
            improved_rows.append(im)
            logger.info(
                f"  [improved] 거래={im['trades']:3d}, 승률={im['win_rate']:5.1f}%, "
                f"수익={im['return_pct']:+7.2f}%, PF={im['profit_factor']:.2f}"
            )
        except Exception as e:
            logger.error(f"  improved 실패: {e}")

    # ───────────────────────── 요약 ─────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("  집계 결과 비교")
    logger.info("=" * 80)

    bl_sum = summarize(baseline_rows, "Baseline")
    im_sum = summarize(improved_rows, "Improved")

    print_keys = [
        ("total_trades", "총 거래수", "{:>10.0f}"),
        ("active_markets", "활성 종목", "{:>10.0f}"),
        ("avg_win_rate", "승률(가중)%", "{:>10.2f}"),
        ("avg_profit_pct", "평균 익절%", "{:>10.2f}"),
        ("avg_loss_pct", "평균 손절%", "{:>10.2f}"),
        ("rr_ratio", "R/R 비율", "{:>10.2f}"),
        ("expectancy_pct", "기대값%", "{:>10.2f}"),
        ("top_symbol", "최다 거래 종목", "{:>14}"),
        ("top_concentration", "집중도%", "{:>10.2f}"),
    ]

    logger.info(f"{'지표':<18} | {'Baseline':>14} | {'Improved':>14}")
    logger.info("-" * 54)
    for key, name, fmt in print_keys:
        bl_v = bl_sum.get(key, "-")
        im_v = im_sum.get(key, "-")
        bl_s = fmt.format(bl_v) if not isinstance(bl_v, str) else f"{bl_v:>14}"
        im_s = fmt.format(im_v) if not isinstance(im_v, str) else f"{im_v:>14}"
        logger.info(f"{name:<18} | {bl_s} | {im_s}")

    # ───────────────────────── 판정 ─────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("  완료 기준 판정 (Improved 기준)")
    logger.info("=" * 80)
    criteria = check_criteria(im_sum)
    for k, passed in criteria.items():
        mark = "✅" if passed else "❌"
        logger.info(f"  {mark} {k}: {im_sum.get(k.split('_')[0], '?')}")

    all_pass = all(criteria.values())
    logger.info(f"\n  종합: {'🎯 모두 통과 (main 병합 권장)' if all_pass else '⚠️ 일부 미달 (추가 튜닝 필요)'}")

    if skipped:
        logger.info(f"\n  건너뛴 종목: {skipped}")

    # CSV 저장
    all_rows = baseline_rows + improved_rows
    if all_rows:
        out_path = Path(__file__).resolve().parent / "data" / (
            f"phase_comparison_{datetime.now().strftime('%Y-%m-%d')}.csv"
        )
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        logger.info(f"\n  결과 CSV: {out_path}")


if __name__ == "__main__":
    main()
