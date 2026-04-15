"""Binance 장기 데이터 수집 + 파라미터 최적화 + 최종 백테스트.

Phase 1 개선: 장기 데이터 확보 및 최적 파라미터 탐색.
"""

from pathlib import Path

import yaml

from backtest.engine import BacktestEngine
from backtest.optimizer import run_optimization
from data import binance_collector
from strategy.tucker import TuckerStrategy
from utils.logger import logger


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()

    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["1m", "5m", "15m"]
    lookback_days = 90

    # ================================================================
    # Step 1: Binance 장기 데이터 수집
    # ================================================================
    logger.info("=" * 60)
    logger.info("  Step 1: Binance 장기 데이터 수집")
    logger.info("=" * 60)

    data_cache: dict[str, dict] = {}

    for symbol in symbols:
        data_cache[symbol] = {}
        for tf in timeframes:
            try:
                df = binance_collector.load_binance_data(symbol, tf)
                logger.info(f"캐시 사용: {symbol} {tf}")
            except FileNotFoundError:
                df = binance_collector.fetch_binance_ohlcv(symbol, tf, lookback_days=lookback_days)
                binance_collector.save_binance_data(df, symbol, tf)
            data_cache[symbol][tf] = df

    # ================================================================
    # Step 2: 기존 파라미터로 장기 데이터 백테스트 (기준선 확인)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  Step 2: 기본 파라미터 백테스트 (기준선)")
    logger.info("=" * 60)

    strategy_cfg = config["strategy"]
    default_strategy = TuckerStrategy(
        ema_period=strategy_cfg["ema"]["period"],
        reset_hour_utc=strategy_cfg["vwap"]["reset_hour_utc"],
        ema_proximity_pct=strategy_cfg["entry"]["ema_proximity_pct"],
        vwap_chop_lookback=strategy_cfg["entry"]["vwap_chop_lookback"],
        vwap_chop_cross_threshold=strategy_cfg["entry"]["vwap_chop_cross_threshold"],
        vp_num_bins=strategy_cfg["volume_profile"]["num_bins"],
        vp_thin_threshold_pct=strategy_cfg["volume_profile"]["thin_threshold_pct"],
    )

    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital_krw"],
        fee_rate=config["backtest"]["fee_rate"],
        slippage_pct=config["backtest"]["slippage_pct"],
    )

    baseline_results = []
    for symbol in symbols:
        for tf in timeframes:
            df = data_cache[symbol][tf]
            result = engine.run(df.copy(), default_strategy, market=symbol, timeframe=tf)
            baseline_results.append(result)
            logger.info(result.summary())

    # 기준선 비교표
    logger.info("\n" + "=" * 60)
    logger.info("  기준선 비교 요약 (Binance 90일 데이터)")
    logger.info("=" * 60)
    logger.info(f"{'마켓':<12} {'TF':<6} {'수익률':>10} {'거래수':>8} {'승률':>8} {'PF':>8} {'MDD':>10}")
    logger.info("-" * 60)
    for r in baseline_results:
        logger.info(
            f"{r.market:<12} {r.timeframe:<6} "
            f"{r.total_return_pct:>9.2f}% "
            f"{r.total_trades:>8d} "
            f"{r.win_rate_pct:>7.1f}% "
            f"{r.profit_factor:>8.2f} "
            f"{r.max_drawdown_pct:>9.2f}%"
        )

    # ================================================================
    # Step 3: 파라미터 최적화 (5m 기준, BTC/ETH 각각)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  Step 3: 파라미터 최적화 (5m 봉 기준)")
    logger.info("=" * 60)

    optimization_results = {}
    for symbol in symbols:
        df = data_cache[symbol]["5m"]
        opt_result = run_optimization(
            df=df,
            initial_capital=config["backtest"]["initial_capital_krw"],
            fee_rate=config["backtest"]["fee_rate"],
            slippage_pct=config["backtest"]["slippage_pct"],
            market=symbol,
            timeframe="5m",
            reset_hour_utc=strategy_cfg["vwap"]["reset_hour_utc"],
        )
        optimization_results[symbol] = opt_result

    # ================================================================
    # Step 4: 최적 파라미터로 전체 타임프레임 재검증
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  Step 4: 최적 파라미터로 전체 타임프레임 재검증")
    logger.info("=" * 60)

    final_results = []
    for symbol in symbols:
        best = optimization_results[symbol].best_params
        optimized_strategy = TuckerStrategy(
            ema_period=best.get("ema_period", 9),
            reset_hour_utc=strategy_cfg["vwap"]["reset_hour_utc"],
            ema_proximity_pct=best.get("ema_proximity_pct", 0.3),
            vwap_chop_lookback=best.get("vwap_chop_lookback", 10),
            vwap_chop_cross_threshold=best.get("vwap_chop_cross_threshold", 3),
            vp_num_bins=strategy_cfg["volume_profile"]["num_bins"],
            vp_thin_threshold_pct=best.get("vp_thin_threshold_pct", 30),
        )

        logger.info(f"\n  {symbol} 최적 파라미터: {best}")

        for tf in timeframes:
            df = data_cache[symbol][tf]
            result = engine.run(df.copy(), optimized_strategy, market=symbol, timeframe=tf)
            final_results.append((symbol, best, result))

    # 최종 비교표
    logger.info("\n" + "=" * 80)
    logger.info("  최종 결과: 기준선 vs 최적화")
    logger.info("=" * 80)
    logger.info(f"{'마켓':<12} {'TF':<6} {'기준 수익률':>12} {'최적 수익률':>12} {'기준 PF':>10} {'최적 PF':>10} {'최적 승률':>10}")
    logger.info("-" * 80)

    for baseline, (symbol, params, optimized) in zip(baseline_results, final_results):
        logger.info(
            f"{baseline.market:<12} {baseline.timeframe:<6} "
            f"{baseline.total_return_pct:>11.2f}% "
            f"{optimized.total_return_pct:>11.2f}% "
            f"{baseline.profit_factor:>10.2f} "
            f"{optimized.profit_factor:>10.2f} "
            f"{optimized.win_rate_pct:>9.1f}%"
        )


if __name__ == "__main__":
    main()
