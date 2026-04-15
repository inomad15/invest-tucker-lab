"""텔레그램 알림 모듈.

매매 시그널 발생 시 텔레그램 봇을 통해 알림을 전송한다.
"""

from datetime import datetime

import requests

from utils.logger import logger

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """텔레그램 봇 알림 전송기.

    Args:
        bot_token: 텔레그램 봇 API 토큰
        chat_id: 알림을 받을 채팅 ID
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._url = TELEGRAM_API_URL.format(token=bot_token)

    def send(self, message: str) -> bool:
        """텔레그램 메시지를 전송한다.

        Args:
            message: 전송할 메시지 (마크다운 지원)

        Returns:
            True면 전송 성공
        """
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(self._url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"텔레그램 알림 전송 완료")
            return True
        except requests.RequestException as e:
            logger.error(f"텔레그램 알림 전송 실패: {e}")
            return False

    def send_signal(
        self,
        signal_type: str,
        market: str,
        price: float,
        ema: float,
        vwap: float,
        reason: str = "",
    ) -> bool:
        """매매 시그널 알림을 전송한다.

        Args:
            signal_type: "BUY" 또는 "SELL"
            market: 마켓 코드 (예: "KRW-BTC")
            price: 현재 가격
            ema: EMA 값
            vwap: VWAP 값
            reason: 청산 사유 (SELL일 때)

        Returns:
            True면 전송 성공
        """
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if signal_type == "BUY":
            message = (
                f"{emoji} *매수 시그널* — {market}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"가격: `{price:,.0f}` KRW\n"
                f"EMA: `{ema:,.0f}`\n"
                f"VWAP: `{vwap:,.0f}`\n"
                f"괴리: `{(price - vwap) / vwap * 100:.2f}%` (VWAP 대비)\n"
                f"시각: {now}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"_Tucker v3 — 눌림목 진입_"
            )
        else:
            reason_text = {
                "ema_exit": "EMA 아래 연속 마감",
                "atr_stop": "ATR 긴급 손절",
            }.get(reason, reason)

            message = (
                f"{emoji} *매도 시그널* — {market}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"가격: `{price:,.0f}` KRW\n"
                f"EMA: `{ema:,.0f}`\n"
                f"VWAP: `{vwap:,.0f}`\n"
                f"사유: {reason_text}\n"
                f"시각: {now}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"_Tucker v3 — 청산_"
            )

        return self.send(message)

    def send_startup(self, markets: list[str], timeframe: str) -> bool:
        """시스템 시작 알림을 전송한다."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        market_list = ", ".join(markets)
        message = (
            f"⚡ *Tucker Signal Bot 시작*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"모니터링: {market_list}\n"
            f"타임프레임: {timeframe}\n"
            f"시작 시각: {now}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"_시그널 발생 시 알림을 전송합니다_"
        )
        return self.send(message)

    def send_status(
        self,
        market: str,
        price: float,
        position: str,
        entry_price: float | None = None,
        pnl_pct: float | None = None,
    ) -> bool:
        """현재 상태 알림을 전송한다."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if position == "holding" and entry_price:
            pnl = ((price / entry_price) - 1) * 100
            message = (
                f"📊 *상태 보고* — {market}\n"
                f"현재가: `{price:,.0f}` KRW\n"
                f"진입가: `{entry_price:,.0f}` KRW\n"
                f"수익률: `{pnl:+.2f}%`\n"
                f"시각: {now}"
            )
        else:
            message = (
                f"📊 *상태 보고* — {market}\n"
                f"현재가: `{price:,.0f}` KRW\n"
                f"포지션: 관망 중\n"
                f"시각: {now}"
            )
        return self.send(message)
