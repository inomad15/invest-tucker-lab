"""Tucker v2 전략 최적화 실행 스크립트.

v1 대비 청산 확인 봉 수, 쿨다운, ATR 손절 파라미터를 포함하여 최적화한다.
Binance 캐시 데이터를 사용한다 (run-optimize.py에서 먼저 수집 필요).
"""

from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from backtest.engine import BacktestEngine
from data.binance_collector import fetch_binance_ohlcv, load_binance_data, save_binance_data
from strategy.tucker_v2 import TuckerStrategyV2
from utils.logger import logger


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()

    symbols = ["BTCUSDT", "ETHUSDT"]
    # 5분봉 기준으로 최적화 (속도와 데이터량 균형)
    timeframe = "5m"
    lookback_days = 90

    # ================================================================
    # Step 1: 데이터 로드 (캐시 없으면 수집)
    # ================================================================
    logger.info("=" * 60)
    logger.info("  Tucker v2 최적화 시작")
    logger.info("=" * 60)

    data_cache: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            data_cache[symbol] = load_binance_data(symbol, timeframe)
        except FileNotFoundError:
            logger.info(f"{symbol} {timeframe} 데이터 수집 중...")
            df = fetch_binance_ohlcv(symbol, timeframe, lookback_days=lookback_days)
            save_binance_data(df, symbol, timeframe)
            data_cache[symbol] = df

    # ================================================================
    # Step 2: v2 파라미터 그리드 서치
    # ================================================================
    # v1 고정 파라미터 (이미 영향이 적음이 확인됨)
    # v2 신규 파라미터에 집중
    param_grid = {
        "ema_period": [9, 21],
        "ema_proximity_pct": [0.3, 0.5, 1.0],
        "vwap_chop_cross_threshold": [3, 5],
        # v2 핵심 파라미터
        "exit_confirm_bars": [2, 3, 5],
        "cooldown_bars": [3, 5, 10],
        "atr_stop_multiplier": [0, 2.0, 3.0],
    }

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    total = len(combinations)

    logger.info(f"총 {total}개 파라미터 조합 탐색")

    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital_krw"],
        fee_rate=config["backtest"]["fee_rate"],
        slippage_pct=config["backtest"]["slippage_pct"],
    )

    for symbol in symbols:
        df = data_cache[symbol]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  {symbol} v2 최적화 ({len(df)}개 캔들)")
        logger.info(f"{'=' * 60}")

        results_data: list[dict] = []
        best_score = -float("inf")
        best_result = None
        best_params: dict = {}

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            strategy = TuckerStrategyV2(
                ema_period=params["ema_period"],
                reset_hour_utc=0,
                ema_proximity_pct=params["ema_proximity_pct"],
                vwap_chop_lookback=10,
                vwap_chop_cross_threshold=params["vwap_chop_cross_threshold"],
                vp_num_bins=20,
                vp_thin_threshold_pct=30,
                exit_confirm_bars=params["exit_confirm_bars"],
                cooldown_bars=params["cooldown_bars"],
                vwap_min_distance_pct=0.1,
                atr_period=14,
                atr_stop_multiplier=params["atr_stop_multiplier"],
            )

            try:
                result = engine.run(df.copy(), strategy, market=symbol, timeframe=timeframe)
            except Exception as e:
                logger.warning(f"  조합 {i + 1}/{total} 실패: {e}")
                continue

            row = {
                **params,
                "total_return_pct": result.total_return_pct,
                "total_trades": result.total_trades,
                "win_rate_pct": result.win_rate_pct,
                "profit_factor": result.profit_factor,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
            }
            results_data.append(row)

            # 스코어링: profit_factor 우선, 동률시 MDD 작은 것
            if result.total_trades >= 5:  # 최소 거래 횟수 필터
                score = result.profit_factor
                if score > best_score or (
                    score == best_score
                    and best_result is not None
                    and result.max_drawdown_pct > best_result.max_drawdown_pct
                ):
                    best_score = score
                    best_result = result
                    best_params = params

            if (i + 1) % 100 == 0:
                logger.info(
                    f"  진행: {i + 1}/{total} ({(i + 1) / total * 100:.0f}%) "
                    f"| 현재 최적 PF={best_score:.2f}"
                )

        # 결과 정리
        results_df = pd.DataFrame(results_data)

        # 거래 5회 이상인 조합만 필터
        valid = results_df[results_df["total_trades"] >= 5].copy()

        if not valid.empty:
            top20 = valid.sort_values("profit_factor", ascending=False).head(20)
            logger.info(f"\n{'=' * 100}")
            logger.info(f"  {symbol} 상위 20개 파라미터 조합 (거래 5회 이상)")
            logger.info(f"{'=' * 100}")
            logger.info(
                f"  {'#':>3} | {'EMA':>3} {'Prox':>5} {'Chop':>4} "
                f"{'Exit':>4} {'Cool':>4} {'ATR':>4} "
                f"| {'PF':>6} {'Win%':>6} {'Ret%':>8} {'MDD%':>8} {'Trades':>6}"
            )
            logger.info(f"  {'-' * 95}")

            for rank, (_, r) in enumerate(top20.iterrows(), 1):
                logger.info(
                    f"  {rank:>3} | {int(r['ema_period']):>3} "
                    f"{r['ema_proximity_pct']:>5.1f} "
                    f"{int(r['vwap_chop_cross_threshold']):>4} "
                    f"{int(r['exit_confirm_bars']):>4} "
                    f"{int(r['cooldown_bars']):>4} "
                    f"{r['atr_stop_multiplier']:>4.1f} "
                    f"| {r['profit_factor']:>6.2f} "
                    f"{r['win_rate_pct']:>5.1f}% "
                    f"{r['total_return_pct']:>7.2f}% "
                    f"{r['max_drawdown_pct']:>7.2f}% "
                    f"{int(r['total_trades']):>6}"
                )

        if best_result:
            logger.info(f"\n  {symbol} 최적 파라미터: {best_params}")
            logger.info(best_result.summary())

        # 결과 CSV 저장
        output_path = Path(__file__).resolve().parent / "data" / f"optimize_v2_{symbol.lower()}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"전체 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
