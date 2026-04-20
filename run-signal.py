"""Tucker v3 실시간 시그널 모니터 실행 스크립트.

Phase 2: Upbit 실시간 데이터 모니터링 + 텔레그램 알림.

사용법:
    # .env 파일에 텔레그램 설정 후 실행
    python run-signal.py

    # 텔레그램 없이 콘솔 로그만으로 실행 (테스트용)
    python run-signal.py --no-telegram
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
import os

from signal.monitor import SignalMonitor
from signal.telegram_notifier import TelegramNotifier
from strategy.tucker_v3 import TuckerStrategyV3
from utils.logger import logger


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tucker v3 실시간 시그널 모니터")
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="텔레그램 알림 없이 콘솔 로그만 출력",
    )
    parser.add_argument(
        "--timeframe",
        default="5m",
        choices=["1m", "3m", "5m", "15m"],
        help="모니터링 타임프레임 (기본: 5m)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="폴링 간격 초 (기본: 30)",
    )
    args = parser.parse_args()

    config = load_config()

    # .env 로드
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    # 텔레그램 설정
    notifier = None
    if not args.no_telegram:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not bot_token or not chat_id:
            logger.error(
                ".env 파일에 TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID를 설정하세요. "
                "텔레그램 없이 실행하려면 --no-telegram 옵션을 사용하세요."
            )
            sys.exit(1)

        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        logger.info("텔레그램 알림 활성화")
    else:
        logger.info("텔레그램 알림 비활성화 (콘솔 로그만 출력)")

    # Phase 1-2: config.yaml에서 신규 필터 파라미터 로드
    _filters = config.get("strategy", {}).get("filters", {})
    _rsi_cfg = _filters.get("rsi", {})
    _vol_cfg = _filters.get("volume_ratio", {})
    _mtf_cfg = _filters.get("mtf", {})
    _atr_cfg = _filters.get("atr", {})
    _filter_kwargs = dict(
        rsi_period=_rsi_cfg.get("period", 14),
        rsi_threshold=_rsi_cfg.get("threshold", 55.0),
        volume_ratio_lookback=_vol_cfg.get("lookback", 20),
        volume_ratio_threshold=_vol_cfg.get("threshold", 1.5),
        require_mtf_agreement=_mtf_cfg.get("require_agreement", True),
        atr_period=_atr_cfg.get("period", 14),
        atr_max_pct=_atr_cfg.get("max_pct", 5.0),
    )
    logger.info(
        f"Phase 1-2 필터: RSI≥{_filter_kwargs['rsi_threshold']}, "
        f"vol≥{_filter_kwargs['volume_ratio_threshold']}x, "
        f"MTF={'ON' if _filter_kwargs['require_mtf_agreement'] else 'OFF'}, "
        f"ATR≤{_filter_kwargs['atr_max_pct']:.1f}%"
    )

    # Phase 2: 시총 가중치 로드 (메타 속성, 실매매 연동 시 포지션 사이징에 사용)
    _portfolio = config.get("portfolio", {})
    _mcap_weights: dict = _portfolio.get("market_cap_weights", {})
    _default_weight: float = _portfolio.get("default_weight", 1.0)
    def _weight_for(market: str) -> float:
        return _mcap_weights.get(market, _default_weight)

    # 마켓별 최적 전략 파라미터 (v3 멀티코인 백테스트 결과)
    # BTC형(EMA9): swing_min=1.0, prox=0.5
    # ETH형(EMA21): swing_min=1.0, prox=0.3
    _common = dict(
        reset_hour_utc=0, swing_lookback=5, swing_min_distance_pct=1.0,
        vwap_chop_lookback=10, vwap_chop_cross_threshold=4,
        vp_num_bins=20, vp_thin_threshold_pct=30,
        exit_confirm_bars=3, cooldown_bars=5, atr_stop_multiplier=0,
        **_filter_kwargs,
    )

    _btc_type = dict(ema_period=9, ema_proximity_pct=0.5, **_common)
    _eth_type = dict(ema_period=21, ema_proximity_pct=0.3, **_common)

    def _make_strategy(market: str, base: dict) -> TuckerStrategyV3:
        return TuckerStrategyV3(**base, market_cap_weight=_weight_for(market))

    # PF > 1.0 달성 11종목 (90일 백테스트 검증 완료)
    strategies: dict[str, TuckerStrategyV3] = {
        "KRW-BTC":  _make_strategy("KRW-BTC",  _btc_type),   # PF=2.68, 13회
        "KRW-ETH":  _make_strategy("KRW-ETH",  _eth_type),   # PF=5.19, 7회
        "KRW-IOST": _make_strategy("KRW-IOST", _eth_type),   # PF=2.45, 4회
        "KRW-RED":  _make_strategy("KRW-RED",  _btc_type),   # PF=2.28, 45회
        "KRW-SOL":  _make_strategy("KRW-SOL",  _eth_type),   # PF=1.87, 12회
        "KRW-DOGE": _make_strategy("KRW-DOGE", _btc_type),   # PF=1.72, 37회
        "KRW-XRP":  _make_strategy("KRW-XRP",  _eth_type),   # PF=1.61, 8회
        "KRW-TREE": _make_strategy("KRW-TREE", _btc_type),   # PF=1.42, 61회
        "KRW-TAO":  _make_strategy("KRW-TAO",  _eth_type),   # PF=1.39, 37회
        "KRW-HOLO": _make_strategy("KRW-HOLO", _eth_type),   # PF=1.36, 14회
    }
    markets = list(strategies.keys())

    # 가중치 할당 로그
    logger.info(
        "Phase 2 시총 가중치: "
        + ", ".join(f"{m.replace('KRW-','')}={s.market_cap_weight:.1f}"
                    for m, s in strategies.items())
    )

    # 모니터 실행
    monitor = SignalMonitor(
        markets=markets,
        strategies=strategies,
        notifier=notifier,
        timeframe=args.timeframe,
        lookback_candles=200,
        poll_interval_sec=args.poll_interval,
    )

    logger.info("=" * 50)
    logger.info("  Tucker v3 Signal Monitor")
    logger.info(f"  마켓: {', '.join(markets)}")
    logger.info(f"  타임프레임: {args.timeframe}")
    logger.info(f"  폴링 간격: {args.poll_interval}초")
    logger.info(f"  텔레그램: {'ON' if notifier else 'OFF'}")
    logger.info("=" * 50)

    monitor.run()


if __name__ == "__main__":
    main()
