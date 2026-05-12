"""Tucker v3 종목 후보 발굴 도구.

Upbit KRW 마켓 중 거래대금 상위 N개를 자동 선정하여
Tucker v3 전략으로 90일 5분봉 백테스트를 일괄 실행하고,
Profit Factor 순으로 정렬하여 매매 후보 목록을 제시한다.

BTC형(EMA9)/ETH형(EMA21) 두 파라미터로 각각 백테스트하여
더 좋은 결과를 종목별 채택값으로 선택한다.

사용 예:
    python tools/find-candidates.py
    python tools/find-candidates.py --top 30 --min-volume 30 --min-pf 1.5
    python tools/find-candidates.py --no-mtf --period 60 --output csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# 프로젝트 루트를 path에 추가하여 패키지 import 보장
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine  # noqa: E402
from data.binance_collector import fetch_binance_ohlcv  # noqa: E402
from data.collector import fetch_ohlcv  # noqa: E402
from strategy.tucker_v3 import TuckerStrategyV3  # noqa: E402
from utils.logger import logger  # noqa: E402

UPBIT_MARKET_ALL_URL = "https://api.upbit.com/v1/market/all"
UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"

# 거래대금/유동성은 Upbit(국내) 기준, 백테스트 데이터는 source 선택 가능.
# "binance"는 USDT 페어로 매핑하여 풍부한 90일+ 5m 데이터 확보.


def upbit_to_binance(market: str) -> str:
    """KRW 마켓 코드를 Binance USDT 페어로 변환.

    예: "KRW-BTC" -> "BTCUSDT"
    """
    base = market.split("-", 1)[-1] if "-" in market else market
    return f"{base}USDT"


def fetch_data(market: str, timeframe: str, lookback_days: int, source: str) -> pd.DataFrame:
    """source에 따라 Upbit 또는 Binance에서 OHLCV를 조회한다.

    Args:
        market: Upbit 마켓 코드 (예: "KRW-BTC")
        timeframe: "5m", "15m", "1h"
        lookback_days: 기간(일)
        source: "upbit" 또는 "binance"
    """
    if source == "upbit":
        return fetch_ohlcv(market, timeframe, lookback_days=lookback_days)
    if source == "binance":
        return fetch_binance_ohlcv(
            upbit_to_binance(market), timeframe, lookback_days=lookback_days
        )
    raise ValueError(f"지원하지 않는 data source: {source}")

# 현재 봇이 운영 중인 종목 (참고 표시용)
CURRENT_BOT_MARKETS = {
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
    "KRW-ADA", "KRW-ONDO", "KRW-HOLO", "KRW-XPL", "KRW-TAO",
    "KRW-TREE", "KRW-RED",
}


@dataclass
class CandidateResult:
    """단일 후보 종목 백테스트 결과."""
    market: str
    volume_24h_billion: float    # 24h 거래대금 (억원)
    type_label: str               # "BTC형(EMA9)" or "ETH형(EMA21)"
    profit_factor: float
    total_trades: int
    win_rate_pct: float
    total_return_pct: float
    avg_profit_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    in_bot_now: bool              # 현재 봇 종목인지

    def is_qualified(self, min_pf: float) -> bool:
        """후보 자격 충족 여부 (PF ≥ 임계값 + 최소 거래수)."""
        return (
            self.profit_factor >= min_pf
            and self.total_trades >= 3
            and not pd.isna(self.profit_factor)
        )


def fetch_top_krw_markets(top_n: int, min_volume_billion: float) -> list[tuple[str, float]]:
    """Upbit KRW 마켓 중 24h 거래대금 상위 N개를 반환한다.

    Args:
        top_n: 상위 몇 개를 반환할지
        min_volume_billion: 최소 거래대금 (억원)

    Returns:
        [(market_code, volume_24h_billion), ...] 내림차순 정렬
    """
    logger.info(f"Upbit KRW 마켓 거래대금 상위 {top_n}개 조회 시작")

    # 1) 전체 KRW 마켓 코드 조회
    resp = requests.get(UPBIT_MARKET_ALL_URL, params={"isDetails": "false"}, timeout=10)
    resp.raise_for_status()
    all_markets = [m["market"] for m in resp.json() if m["market"].startswith("KRW-")]
    logger.info(f"  KRW 마켓 총 {len(all_markets)}개 발견")

    # 2) 100개씩 batch로 24h 거래대금 조회
    market_volumes: list[tuple[str, float]] = []
    batch_size = 100
    for i in range(0, len(all_markets), batch_size):
        batch = all_markets[i:i + batch_size]
        params = {"markets": ",".join(batch)}
        resp = requests.get(UPBIT_TICKER_URL, params=params, timeout=10)
        resp.raise_for_status()
        for t in resp.json():
            volume_krw = t["acc_trade_price_24h"]
            volume_billion = volume_krw / 1e8  # 원 → 억원
            market_volumes.append((t["market"], volume_billion))

    # 3) 거래대금 내림차순 정렬 + 임계값 필터
    market_volumes.sort(key=lambda x: x[1], reverse=True)
    qualified = [(m, v) for m, v in market_volumes if v >= min_volume_billion]
    selected = qualified[:top_n]

    logger.info(
        f"  거래대금 ≥ {min_volume_billion}억원 통과: {len(qualified)}개 → "
        f"상위 {len(selected)}개 채택"
    )
    return selected


def run_backtest_for_market(
    market: str,
    df_5m: pd.DataFrame,
    htf_dfs: dict[str, pd.DataFrame] | None,
    type_label: str,
    ema_period: int,
    ema_proximity_pct: float,
    require_mtf: bool,
) -> tuple[float, int, float, float, float, float, float]:
    """단일 종목 + 단일 파라미터 형식 백테스트.

    Returns:
        (pf, trades, win_rate, total_return, avg_profit, avg_loss, mdd)
    """
    strategy = TuckerStrategyV3(
        ema_period=ema_period,
        reset_hour_utc=0,
        swing_lookback=5,
        swing_min_distance_pct=1.0,
        ema_proximity_pct=ema_proximity_pct,
        vwap_chop_lookback=10,
        vwap_chop_cross_threshold=4,
        vp_num_bins=20,
        vp_thin_threshold_pct=30,
        exit_confirm_bars=3,
        cooldown_bars=5,
        atr_period=14,
        atr_stop_multiplier=0,
        rsi_period=14,
        rsi_threshold=55.0,
        volume_ratio_lookback=20,
        volume_ratio_threshold=1.5,
        require_mtf_agreement=require_mtf,
        atr_max_pct=5.0,
        take_profit_pct=5.0,
        time_stop_bars=48,
    )

    engine = BacktestEngine(initial_capital=10_000_000, fee_rate=0.0005, slippage_pct=0.05)
    result = engine.run(
        df_5m.copy(),
        strategy,
        market=market,
        timeframe="5m",
        htf_dfs=htf_dfs,
    )
    return (
        result.profit_factor,
        result.total_trades,
        result.win_rate_pct,
        result.total_return_pct,
        result.avg_profit_pct,
        result.avg_loss_pct,
        result.max_drawdown_pct,
    )


def evaluate_candidate(
    market: str,
    volume_billion: float,
    period_days: int,
    require_mtf: bool,
    source: str,
) -> CandidateResult | None:
    """단일 종목에 대해 BTC형/ETH형 두 백테스트 → 더 좋은 쪽 선택.

    Args:
        source: "upbit" (국내 5m API, 데이터 보유 짧음) 또는
                "binance" (USDT 페어, 90일+ 데이터 풍부).
    """
    label = market if source == "upbit" else f"{market}→{upbit_to_binance(market)}"
    logger.info(
        f"[{label}] 데이터 수집 + 백테스트 시작 "
        f"(거래대금 {volume_billion:.1f}억원, source={source})"
    )

    # 1) 5m 데이터 수집
    try:
        df_5m = fetch_data(market, "5m", period_days, source)
    except Exception as exc:
        logger.warning(f"[{label}] 5m 데이터 수집 실패: {exc}")
        return None
    if df_5m is None or len(df_5m) < 500:
        logger.warning(f"[{label}] 5m 캔들 수 부족 ({0 if df_5m is None else len(df_5m)}개)")
        return None

    # 2) MTF 데이터 (옵션)
    htf_dfs: dict[str, pd.DataFrame] | None = None
    if require_mtf:
        htf_dfs = {}
        for tf in ("15m", "1h"):
            try:
                htf_dfs[tf] = fetch_data(market, tf, period_days, source)
            except Exception as exc:
                logger.warning(f"[{label}] {tf} 데이터 수집 실패 → MTF 비활성: {exc}")
                htf_dfs = None
                break

    # 3) BTC형 + ETH형 백테스트
    best: tuple[float, int, float, float, float, float, float, str] | None = None
    for label, ema_p, prox in [("BTC형(EMA9)", 9, 0.5), ("ETH형(EMA21)", 21, 0.3)]:
        try:
            pf, n, wr, ret, avgp, avgl, mdd = run_backtest_for_market(
                market, df_5m, htf_dfs, label, ema_p, prox, require_mtf=(htf_dfs is not None),
            )
        except Exception as exc:
            logger.warning(f"[{market}] {label} 백테스트 실패: {exc}")
            continue
        # 가장 좋은 결과 = PF 큰 쪽 (단, 최소 거래수 3 이상)
        if n >= 3 and (best is None or pf > best[0]):
            best = (pf, n, wr, ret, avgp, avgl, mdd, label)

    if best is None:
        logger.info(f"[{market}] 자격 미달 (거래 3건 미만 또는 백테스트 실패)")
        return None

    pf, n, wr, ret, avgp, avgl, mdd, label = best
    return CandidateResult(
        market=market,
        volume_24h_billion=volume_billion,
        type_label=label,
        profit_factor=pf,
        total_trades=n,
        win_rate_pct=wr,
        total_return_pct=ret,
        avg_profit_pct=avgp,
        avg_loss_pct=avgl,
        max_drawdown_pct=mdd,
        in_bot_now=market in CURRENT_BOT_MARKETS,
    )


def format_table(results: list[CandidateResult], min_pf: float) -> str:
    """결과를 마크다운 표로 포맷팅."""
    if not results:
        return "후보 없음."

    # PF 내림차순 정렬
    results.sort(key=lambda r: r.profit_factor, reverse=True)

    lines = [
        f"\n{'=' * 86}",
        f"  Tucker 후보 발굴 결과 (PF ≥ {min_pf}, 최소 3거래)",
        f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}",
        f"{'=' * 86}",
        f"  {'#':<3} {'종목':<10} {'PF':>6} {'거래':>5} {'승률':>6} {'수익':>8} "
        f"{'24h억':>7} {'타입':<14} {'현재봇'}",
        f"  {'-' * 84}",
    ]
    for i, r in enumerate(results, 1):
        bot_mark = "★" if r.in_bot_now else ""
        lines.append(
            f"  {i:<3} {r.market.replace('KRW-', ''):<10} "
            f"{r.profit_factor:>6.2f} {r.total_trades:>5d} {r.win_rate_pct:>5.1f}% "
            f"{r.total_return_pct:>7.2f}% {r.volume_24h_billion:>7.1f} "
            f"{r.type_label:<14} {bot_mark}"
        )
    lines.append(f"{'=' * 86}")

    # 추천 메모
    qualified = [r for r in results if r.is_qualified(min_pf)]
    new_candidates = [r for r in qualified if not r.in_bot_now]
    weak_current = [r for r in results if r.in_bot_now and r.profit_factor < min_pf]

    if new_candidates:
        lines.append("\n[추가 후보 — 현재 봇에 없는 우수 종목]")
        for r in new_candidates[:5]:
            lines.append(
                f"  + {r.market.replace('KRW-', '')}: PF={r.profit_factor:.2f}, "
                f"{r.total_trades}거래, {r.type_label}"
            )
    if weak_current:
        lines.append("\n[제거 검토 — 현재 봇에 있으나 PF 부진]")
        for r in weak_current:
            lines.append(
                f"  - {r.market.replace('KRW-', '')}: PF={r.profit_factor:.2f}, "
                f"{r.total_trades}거래 (현재 봇 운영 중)"
            )
    return "\n".join(lines)


def save_csv(results: list[CandidateResult], path: Path) -> None:
    """CSV로 저장."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "market", "volume_24h_billion", "type_label", "profit_factor",
            "total_trades", "win_rate_pct", "total_return_pct",
            "avg_profit_pct", "avg_loss_pct", "max_drawdown_pct", "in_bot_now",
        ])
        for r in sorted(results, key=lambda x: x.profit_factor, reverse=True):
            writer.writerow([
                r.market, f"{r.volume_24h_billion:.2f}", r.type_label,
                f"{r.profit_factor:.4f}", r.total_trades,
                f"{r.win_rate_pct:.2f}", f"{r.total_return_pct:.4f}",
                f"{r.avg_profit_pct:.4f}", f"{r.avg_loss_pct:.4f}",
                f"{r.max_drawdown_pct:.4f}", r.in_bot_now,
            ])
    logger.info(f"CSV 저장: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tucker v3 종목 후보 발굴 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top", type=int, default=20,
                   help="거래대금 상위 N개 종목을 후보로 검증")
    p.add_argument("--min-volume", type=float, default=30.0,
                   help="후보 최소 24h 거래대금 (억원)")
    p.add_argument("--min-pf", type=float, default=1.5,
                   help="결과 표시 임계 PF (이 아래도 출력은 되지만 [추가후보]에선 제외)")
    p.add_argument("--period", type=int, default=90,
                   help="백테스트 기간 (일)")
    p.add_argument("--no-mtf", action="store_true",
                   help="MTF(15m, 1h) 일치 검증 비활성화 (실행 빠르나 production과 다소 다름)")
    p.add_argument("--include-bot", action="store_true",
                   help="현재 봇 운영 중인 12종목도 후보군에 포함")
    p.add_argument("--source", choices=["upbit", "binance"], default="binance",
                   help="백테스트 데이터 소스. binance=USDT페어(90일 풀데이터), "
                        "upbit=KRW 5m(소형 종목은 데이터 부족 가능)")
    p.add_argument("--markets-only", type=str, default=None,
                   help="쉼표로 구분된 KRW 마켓 코드(예: 'KRW-BTC,KRW-ETH'). "
                        "지정 시 거래대금 top-N 필터를 건너뛰고 명시된 종목만 평가. "
                        "특정 종목군(예: 현재 봇 12종목) 재검증 용도.")
    p.add_argument("--output", choices=["text", "csv", "both"], default="both",
                   help="출력 포맷")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Tucker v3 종목 후보 발굴")
    logger.info(f"  파라미터: top={args.top}, min_volume={args.min_volume}억, "
                f"min_pf={args.min_pf}, period={args.period}일, "
                f"MTF={'OFF' if args.no_mtf else 'ON'}, source={args.source}")
    logger.info("=" * 60)

    # 1) 후보 종목 발굴 — markets-only 지정 시 그 목록을, 아니면 거래대금 상위 N개
    if args.markets_only:
        explicit = [m.strip() for m in args.markets_only.split(",") if m.strip()]
        # 거래대금 정보를 추가로 조회 (참고용)
        volume_lookup = dict(fetch_top_krw_markets(300, 0.0))
        top_markets = [(m, volume_lookup.get(m, 0.0)) for m in explicit]
        logger.info(f"명시 종목 {len(top_markets)}개: {', '.join(explicit)}")
    else:
        top_markets = fetch_top_krw_markets(args.top, args.min_volume)
        if not top_markets:
            logger.error("후보 종목 0개 — 임계값 또는 네트워크 확인 필요")
            sys.exit(1)

        # 1-a) 현재 봇 종목 제외 옵션 (markets-only가 아닌 경우만 적용)
        if not args.include_bot:
            before = len(top_markets)
            top_markets = [(m, v) for m, v in top_markets if m not in CURRENT_BOT_MARKETS]
            logger.info(f"현재 봇 종목 제외: {before} → {len(top_markets)}개")

    # 2) 각 후보 백테스트
    results: list[CandidateResult] = []
    for i, (market, volume_b) in enumerate(top_markets, 1):
        logger.info(f"\n[진행: {i}/{len(top_markets)}] {market}")
        cand = evaluate_candidate(
            market, volume_b, args.period,
            require_mtf=not args.no_mtf, source=args.source,
        )
        if cand is not None:
            results.append(cand)

    # 3) 출력
    if args.output in ("text", "both"):
        print(format_table(results, args.min_pf))

    if args.output in ("csv", "both"):
        out_dir = PROJECT_ROOT / "data"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_csv(results, out_dir / f"candidates_{ts}.csv")

    logger.info("\n완료: 후보 발굴 + 백테스트 종료")


if __name__ == "__main__":
    main()
