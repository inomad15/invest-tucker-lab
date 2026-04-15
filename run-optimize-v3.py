"""Tucker v3 전략 최적화 + 전체 타임프레임 검증.

v3 핵심: 다봉 패턴 기반 눌림목 진입 (선행 상승 확인).
"""

from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from backtest.engine import BacktestEngine
from data.binance_collector import load_binance_data
from strategy.tucker_v3 import TuckerStrategyV3
from utils.logger import logger


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()

    symbols = ["BTCUSDT", "ETHUSDT"]
    optimize_tf = "5m"
    all_timeframes = ["1m", "5m", "15m"]

    # 데이터 로드
    data_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        data_cache[symbol] = {}
        for tf in all_timeframes:
            data_cache[symbol][tf] = load_binance_data(symbol, tf)

    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital_krw"],
        fee_rate=config["backtest"]["fee_rate"],
        slippage_pct=config["backtest"]["slippage_pct"],
    )

    # ================================================================
    # v3 파라미터 그리드: 진입 선행 상승 + 청산 확인에 집중
    # ================================================================
    param_grid = {
        "ema_period": [9, 21],
        "swing_lookback": [5, 10, 20],
        "swing_min_distance_pct": [0.2, 0.5, 1.0],
        "ema_proximity_pct": [0.3, 0.5, 1.0],
        "exit_confirm_bars": [2, 3],
        "cooldown_bars": [5, 10],
        "atr_stop_multiplier": [0, 2.0],
    }

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    total = len(combinations)

    logger.info(f"v3 최적화: {total}개 조합")

    best_per_symbol: dict[str, tuple[dict, object]] = {}

    for symbol in symbols:
        df = data_cache[symbol][optimize_tf]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  {symbol} v3 최적화 ({len(df)} 캔들, {optimize_tf})")
        logger.info(f"{'=' * 60}")

        results_data: list[dict] = []
        best_score = -float("inf")
        best_result = None
        best_params: dict = {}

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            strategy = TuckerStrategyV3(
                ema_period=params["ema_period"],
                reset_hour_utc=0,
                swing_lookback=params["swing_lookback"],
                swing_min_distance_pct=params["swing_min_distance_pct"],
                ema_proximity_pct=params["ema_proximity_pct"],
                vwap_chop_lookback=10,
                vwap_chop_cross_threshold=4,
                vp_num_bins=20,
                vp_thin_threshold_pct=30,
                exit_confirm_bars=params["exit_confirm_bars"],
                cooldown_bars=params["cooldown_bars"],
                atr_period=14,
                atr_stop_multiplier=params["atr_stop_multiplier"],
            )

            try:
                result = engine.run(df.copy(), strategy, market=symbol, timeframe=optimize_tf)
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
                "avg_profit_pct": result.avg_profit_pct,
                "avg_loss_pct": result.avg_loss_pct,
            }
            results_data.append(row)

            if result.total_trades >= 3:
                score = result.profit_factor
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_params = params

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  진행: {i + 1}/{total} ({(i + 1) / total * 100:.0f}%) "
                    f"| 현재 최적 PF={best_score:.2f}"
                )

        # 결과 정리
        results_df = pd.DataFrame(results_data)
        valid = results_df[results_df["total_trades"] >= 3].copy()

        if not valid.empty:
            top20 = valid.sort_values("profit_factor", ascending=False).head(20)
            logger.info(f"\n{'=' * 110}")
            logger.info(f"  {symbol} 상위 20개 (거래 3회 이상)")
            logger.info(f"{'=' * 110}")
            logger.info(
                f"  {'#':>3} | {'EMA':>3} {'Swing':>5} {'SwDst':>5} "
                f"{'Prox':>5} {'Exit':>4} {'Cool':>4} {'ATR':>4} "
                f"| {'PF':>6} {'Win%':>6} {'Ret%':>8} {'MDD%':>8} {'Trades':>6} {'AvgW%':>6} {'AvgL%':>6}"
            )
            logger.info(f"  {'-' * 105}")

            for rank, (_, r) in enumerate(top20.iterrows(), 1):
                logger.info(
                    f"  {rank:>3} | {int(r['ema_period']):>3} "
                    f"{int(r['swing_lookback']):>5} "
                    f"{r['swing_min_distance_pct']:>5.1f} "
                    f"{r['ema_proximity_pct']:>5.1f} "
                    f"{int(r['exit_confirm_bars']):>4} "
                    f"{int(r['cooldown_bars']):>4} "
                    f"{r['atr_stop_multiplier']:>4.1f} "
                    f"| {r['profit_factor']:>6.2f} "
                    f"{r['win_rate_pct']:>5.1f}% "
                    f"{r['total_return_pct']:>7.2f}% "
                    f"{r['max_drawdown_pct']:>7.2f}% "
                    f"{int(r['total_trades']):>6} "
                    f"{r['avg_profit_pct']:>5.2f}% "
                    f"{r['avg_loss_pct']:>5.2f}%"
                )

        if best_result:
            best_per_symbol[symbol] = (best_params, best_result)
            logger.info(f"\n  {symbol} 최적 파라미터: {best_params}")
            logger.info(best_result.summary())

        # CSV 저장
        output_path = Path(__file__).resolve().parent / "data" / f"optimize_v3_{symbol.lower()}.csv"
        results_df.to_csv(output_path, index=False)

    # ================================================================
    # 최적 파라미터로 전체 타임프레임 검증
    # ================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("  최적 파라미터로 전체 타임프레임 재검증")
    logger.info(f"{'=' * 80}")

    logger.info(
        f"  {'마켓':<12} {'TF':<6} {'수익률':>10} {'거래수':>8} "
        f"{'승률':>8} {'PF':>8} {'MDD':>10} {'평균수익':>8} {'평균손실':>8}"
    )
    logger.info(f"  {'-' * 85}")

    for symbol in symbols:
        if symbol not in best_per_symbol:
            continue

        best_params, _ = best_per_symbol[symbol]
        strategy = TuckerStrategyV3(
            ema_period=best_params["ema_period"],
            reset_hour_utc=0,
            swing_lookback=best_params["swing_lookback"],
            swing_min_distance_pct=best_params["swing_min_distance_pct"],
            ema_proximity_pct=best_params["ema_proximity_pct"],
            vwap_chop_lookback=10,
            vwap_chop_cross_threshold=4,
            vp_num_bins=20,
            vp_thin_threshold_pct=30,
            exit_confirm_bars=best_params["exit_confirm_bars"],
            cooldown_bars=best_params["cooldown_bars"],
            atr_period=14,
            atr_stop_multiplier=best_params["atr_stop_multiplier"],
        )

        for tf in all_timeframes:
            df = data_cache[symbol][tf]
            result = engine.run(df.copy(), strategy, market=symbol, timeframe=tf)
            logger.info(
                f"  {symbol:<12} {tf:<6} "
                f"{result.total_return_pct:>9.2f}% "
                f"{result.total_trades:>8d} "
                f"{result.win_rate_pct:>7.1f}% "
                f"{result.profit_factor:>8.2f} "
                f"{result.max_drawdown_pct:>9.2f}% "
                f"{result.avg_profit_pct:>7.2f}% "
                f"{result.avg_loss_pct:>7.2f}%"
            )


if __name__ == "__main__":
    main()
