"""Tucker 전략 백테스트 실행 스크립트.

Upbit에서 BTC/KRW, ETH/KRW 데이터를 수집하고,
Tucker 전략으로 백테스트를 실행하여 결과를 출력한다.
"""

from pathlib import Path

import yaml

from backtest.engine import BacktestEngine
from data.collector import fetch_ohlcv, load_data, save_data
from strategy.tucker import TuckerStrategy
from utils.logger import logger


def load_config() -> dict:
    """config.yaml을 로드한다."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_backtest() -> None:
    """전체 백테스트 파이프라인을 실행한다."""
    config = load_config()

    markets = config["trading"]["markets"]
    timeframes = config["backtest"]["timeframes"]
    lookback_days = config["data"]["lookback_days"]

    # 전략 파라미터
    strategy_cfg = config["strategy"]
    strategy = TuckerStrategy(
        ema_period=strategy_cfg["ema"]["period"],
        reset_hour_utc=strategy_cfg["vwap"]["reset_hour_utc"],
        ema_proximity_pct=strategy_cfg["entry"]["ema_proximity_pct"],
        vwap_chop_lookback=strategy_cfg["entry"]["vwap_chop_lookback"],
        vwap_chop_cross_threshold=strategy_cfg["entry"]["vwap_chop_cross_threshold"],
        vp_num_bins=strategy_cfg["volume_profile"]["num_bins"],
        vp_thin_threshold_pct=strategy_cfg["volume_profile"]["thin_threshold_pct"],
    )

    # 백테스트 엔진
    engine = BacktestEngine(
        initial_capital=config["backtest"]["initial_capital_krw"],
        fee_rate=config["backtest"]["fee_rate"],
        slippage_pct=config["backtest"]["slippage_pct"],
    )

    results = []

    for market in markets:
        for timeframe in timeframes:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  {market} / {timeframe} 백테스트 시작")
            logger.info(f"{'=' * 60}")

            # 1. 데이터 수집 (캐시 확인)
            try:
                df = load_data(market, timeframe)
                logger.info(f"캐시된 데이터 사용: {market} {timeframe}")
            except FileNotFoundError:
                logger.info(f"데이터 수집 중: {market} {timeframe}")
                df = fetch_ohlcv(market, timeframe, lookback_days=lookback_days)
                save_data(df, market, timeframe)

            # 2. 백테스트 실행
            result = engine.run(df, strategy, market=market, timeframe=timeframe)
            results.append(result)

            # 3. 결과 출력
            logger.info(result.summary())

    # 전체 결과 비교
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("  타임프레임별 비교 요약")
        logger.info("=" * 60)
        logger.info(f"{'마켓':<12} {'TF':<6} {'수익률':>10} {'거래수':>8} {'승률':>8} {'MDD':>10} {'샤프':>8}")
        logger.info("-" * 60)

        for r in results:
            logger.info(
                f"{r.market:<12} {r.timeframe:<6} "
                f"{r.total_return_pct:>9.2f}% "
                f"{r.total_trades:>8d} "
                f"{r.win_rate_pct:>7.1f}% "
                f"{r.max_drawdown_pct:>9.2f}% "
                f"{r.sharpe_ratio:>8.2f}"
            )


if __name__ == "__main__":
    run_backtest()
