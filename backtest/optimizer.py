"""파라미터 최적화 모듈.

Tucker 전략의 주요 파라미터를 그리드 서치하여
최적의 조합을 탐색한다.
"""

from dataclasses import dataclass
from itertools import product

import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult
from strategy.tucker import TuckerStrategy
from utils.logger import logger


@dataclass
class OptimizationResult:
    """최적화 결과."""
    best_params: dict
    best_result: BacktestResult
    all_results: pd.DataFrame  # 전체 조합별 결과 요약


# 탐색할 파라미터 그리드
DEFAULT_PARAM_GRID: dict[str, list] = {
    "ema_period": [9, 12, 21],
    "ema_proximity_pct": [0.1, 0.3, 0.5, 1.0],
    "vwap_chop_lookback": [5, 10, 20],
    "vwap_chop_cross_threshold": [2, 3, 5],
    "vp_thin_threshold_pct": [20, 30, 50],
}


def run_optimization(
    df: pd.DataFrame,
    param_grid: dict[str, list] | None = None,
    initial_capital: float = 1_000_000,
    fee_rate: float = 0.0005,
    slippage_pct: float = 0.05,
    market: str = "",
    timeframe: str = "",
    reset_hour_utc: int = 0,
) -> OptimizationResult:
    """파라미터 그리드 서치를 실행한다.

    최적화 기준: Profit Factor > 1 이면서 MDD가 가장 작은 조합.
    Profit Factor > 1인 조합이 없으면, Profit Factor가 가장 높은 조합.

    Args:
        df: OHLCV DataFrame
        param_grid: 파라미터 그리드 (None이면 기본값 사용)
        initial_capital: 초기 자본
        fee_rate: 수수료율
        slippage_pct: 슬리피지
        market: 마켓 코드 (표시용)
        timeframe: 타임프레임 (표시용)
        reset_hour_utc: VWAP 리셋 시각

    Returns:
        OptimizationResult
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    engine = BacktestEngine(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
    )

    # 모든 파라미터 조합 생성
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    total = len(combinations)

    logger.info(f"파라미터 최적화 시작: {total}개 조합 ({market} {timeframe})")

    results_data: list[dict] = []
    best_pf = -float("inf")
    best_result: BacktestResult | None = None
    best_params: dict = {}

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        strategy = TuckerStrategy(
            ema_period=params.get("ema_period", 9),
            reset_hour_utc=reset_hour_utc,
            ema_proximity_pct=params.get("ema_proximity_pct", 0.3),
            vwap_chop_lookback=params.get("vwap_chop_lookback", 10),
            vwap_chop_cross_threshold=params.get("vwap_chop_cross_threshold", 3),
            vp_num_bins=20,
            vp_thin_threshold_pct=params.get("vp_thin_threshold_pct", 30),
        )

        try:
            result = engine.run(df.copy(), strategy, market=market, timeframe=timeframe)
        except Exception as e:
            logger.warning(f"  조합 {i + 1}/{total} 실패: {params} - {e}")
            continue

        # 결과 기록
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

        # 최적 조합 갱신 기준:
        # 1차: profit_factor 높은 것
        # 2차: 동일 PF면 MDD가 작은 것
        # 거래 0회인 경우 제외
        if result.total_trades > 0:
            score = result.profit_factor
            if score > best_pf or (
                score == best_pf
                and best_result is not None
                and result.max_drawdown_pct > best_result.max_drawdown_pct
            ):
                best_pf = score
                best_result = result
                best_params = params

        if (i + 1) % 50 == 0:
            logger.info(f"  진행: {i + 1}/{total} ({(i + 1) / total * 100:.0f}%)")

    results_df = pd.DataFrame(results_data)

    if best_result is None:
        raise RuntimeError("유효한 백테스트 결과가 없습니다.")

    # 상위 10개 조합 출력
    if not results_df.empty:
        top10 = results_df.sort_values("profit_factor", ascending=False).head(10)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"  상위 10개 파라미터 조합 ({market} {timeframe})")
        logger.info(f"{'=' * 80}")
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            logger.info(
                f"  #{rank:2d} | EMA={int(row['ema_period']):2d} "
                f"Prox={row['ema_proximity_pct']:.1f}% "
                f"Chop={int(row['vwap_chop_lookback'])}/{int(row['vwap_chop_cross_threshold'])} "
                f"VP={int(row['vp_thin_threshold_pct'])}% "
                f"| PF={row['profit_factor']:.2f} "
                f"Win={row['win_rate_pct']:.0f}% "
                f"Ret={row['total_return_pct']:.2f}% "
                f"MDD={row['max_drawdown_pct']:.2f}% "
                f"Trades={int(row['total_trades'])}"
            )

    logger.info(f"\n  최적 파라미터: {best_params}")
    logger.info(best_result.summary())

    return OptimizationResult(
        best_params=best_params,
        best_result=best_result,
        all_results=results_df,
    )
