# 작업일지 — invest-tucker-lab

Tucker Brooks 매매법 기반 암호화폐 자동매매 시스템 개발 프로젝트.

---

## 2026-04-15 (Day 1) — 프로젝트 구축 + 전략 검증 + 멀티코인 확장

### 완료된 작업

#### Phase 1: 백테스트 시스템 구축
- Python 3.11 venv 생성, 의존성 설치
- 프로젝트 구조 설계: `strategy/`, `data/`, `backtest/`, `signal/`, `utils/`
- Upbit API 데이터 수집 모듈 (`data/collector.py`)
- Binance 장기 데이터 수집 모듈 (`data/binance_collector.py`) — 90일 5분봉
- 지표 모듈 구현: VWAP(KST 09:00 리셋), EMA, Volume Profile (`strategy/indicators.py`)
- 백테스트 엔진 구현 (`backtest/engine.py`)
- 파라미터 최적화 엔진 구현 (`backtest/optimizer.py`)

#### 전략 진화 과정
1. **v1** (`strategy/tucker.py`) — 원본 Tucker 전략 그대로 구현
   - 결과: 모든 조합에서 손실 (-47% ~ -73%)
   - 원인: 청산 조건(종가 < EMA 1봉 마감)이 너무 민감 → whipsaw

2. **v2** (`strategy/tucker_v2.py`) — 청산 확인 봉 수 + 쿨다운 추가
   - 결과: 개선되었으나 여전히 손실

3. **v3** (`strategy/tucker_v3.py`) — **다봉 패턴 기반 눌림목 진입** (핵심 개선)
   - 선행 상승 확인 (`swing_min_distance_pct=1.0%`) 추가
   - BTC 최적: EMA9, PF=2.68, +5.96%, 13회/90일
   - ETH 최적: EMA21, PF=5.19, +9.77%, 7회/90일
   - **수익 달성, 그러나 거래 빈도 낮음 (7~13회/90일)**

4. **v4** (`strategy/tucker_v4.py`) — 평균회귀 전략 시도
   - 기술적 지표 예측력 검증 결과: 모든 지표가 50% 미만 (동전 던지기)
   - 유일한 예측력: 평균회귀 (EMA 괴리 -1% → 상승확률 57%)
   - 결과: 거래 빈도↑ 가능하지만 PF < 1.0 → 수수료를 못 이김
   - **결론: 일 1회 이상 거래 + 수익은 단일 종목으로 불가**

#### B안: 멀티코인 확장 (최종 채택)
- Upbit 거래대금 상위 20개 종목 × v3 전략 백테스트
- **PF > 1.0 달성 11종목 선별** (XPL 제외 → 최종 10종목)
- 합산 거래: **292회/90일 = 일 3.2회** → 목표 달성

#### Phase 2: 실시간 시그널 알림
- 텔레그램 봇 연동 (`signal/telegram_notifier.py`)
- 10종목 실시간 모니터 (`signal/monitor.py`)
- 실행 스크립트 (`run-signal.py`)
- **현재 가동 중**

### 현재 모니터링 종목 (10개)

| 종목 | 파라미터 | PF | 90일 거래수 | 90일 수익률 |
|------|---------|-----|-----------|-----------|
| KRW-BTC | BTC형(EMA9) | 2.68 | 13 | +5.96% |
| KRW-ETH | ETH형(EMA21) | 5.19 | 7 | +9.77% |
| KRW-IOST | ETH형(EMA21) | 2.45 | 4 | +1.98% |
| KRW-RED | BTC형(EMA9) | 2.28 | 45 | +31.86% |
| KRW-SOL | ETH형(EMA21) | 1.87 | 12 | +3.92% |
| KRW-DOGE | BTC형(EMA9) | 1.72 | 37 | +8.91% |
| KRW-XRP | ETH형(EMA21) | 1.61 | 8 | +1.42% |
| KRW-TREE | BTC형(EMA9) | 1.42 | 61 | +11.47% |
| KRW-TAO | ETH형(EMA21) | 1.39 | 37 | +3.75% |
| KRW-HOLO | ETH형(EMA21) | 1.36 | 14 | +1.06% |

### 전략 파라미터

**BTC형 (EMA9):**
- ema_period=9, swing_lookback=5, swing_min_distance_pct=1.0
- ema_proximity_pct=0.5, exit_confirm_bars=3, cooldown_bars=5

**ETH형 (EMA21):**
- ema_period=21, swing_lookback=5, swing_min_distance_pct=1.0
- ema_proximity_pct=0.3, exit_confirm_bars=3, cooldown_bars=5

**공통:** vwap_chop_cross_threshold=4, atr_stop_multiplier=0

### 핵심 발견 사항

1. **추세추종 지표(VWAP, EMA, ADX, RSI, MACD)는 5분봉에서 미래 방향 예측 불가** (전부 50% 미만)
2. **평균회귀가 유일한 통계적 우위** (EMA 괴리 -1% → 상승 57%) — 단, 수수료를 이길 만큼 강하지 않음
3. **Tucker 전략의 가치는 "추세 예측"이 아니라 "좋은 손익비 자리 선별"**
4. **거래 빈도와 수익률은 반비례** — 거래를 늘리면 일관되게 손실 증가
5. **멀티코인 확장이 유일한 해법** — 전략 우위를 유지하면서 빈도를 높이는 방법

### 미완료 / 다음 작업

- [ ] Phase 3: Upbit 자동 주문 실행 (pyupbit 매매 연동)
- [ ] 종목 정기 재검증 시스템 (1~2주 간격)
- [ ] 포트폴리오 자본 배분 로직 (종목별 자금 비중)
- [ ] 시그널 실전 검증 (수동 매매로 1~2주 관찰)
- [ ] Git 저장소 초기화 및 커밋

### 실행 방법

```bash
cd /Users/chb/Projects_Mini2/invest-tucker-lab

# 텔레그램 알림 포함 실행
./venv/bin/python run-signal.py

# 텔레그램 없이 테스트
./venv/bin/python run-signal.py --no-telegram

# 백테스트 실행
./venv/bin/python main.py

# v3 최적화
./venv/bin/python run-optimize-v3.py
```

### 환경

- Python 3.11.15 (venv)
- 텔레그램 봇 토큰/Chat ID: `.env` 파일에 저장
- 데이터 캐시: `data/` 디렉토리 (Binance 90일 5분봉)
