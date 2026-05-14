"""
15-Minute BTC Trading Strategy
Main strategy that coordinates signal processing and trading decisions
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import deque
from loguru import logger
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine, FusedSignal
from core.strategy_brain.signal_processors.base_processor import SignalDirection
from execution.risk_engine import get_risk_engine


# Market interval in minutes
INTERVAL_MINUTES = 15
# Close position this many seconds before market expiry to avoid getting stuck
TIME_STOP_BUFFER_SECONDS = 60


class BTCStrategy15Min:
    """
    15-minute BTC trading strategy.

    Workflow:
    1. Collect price data every 15 minutes
    2. Run all 6 signal processors
    3. Fuse signals into consensus
    4. Validate through Risk Engine before placing order
    5. Manage positions with SL / TP / time-based stop
    """

    def __init__(
        self,
        max_position_size: Decimal = Decimal("1.0"),
        # Fixed R:R: reward must be >= risk (TP >= SL)
        stop_loss_pct: float = 0.15,   # 15% stop loss
        take_profit_pct: float = 0.25, # 25% take profit  → R:R = 1.67
        max_positions: int = 2,
        initial_capital: Decimal = Decimal("10.0"),
    ):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions

        # --- Signal processors (all 6) ---
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=0.05,   # Fixed: 0.15 was calibrated for dollar prices, not probabilities
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25,
            extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=0.05,
        )
        self.orderbook_processor = OrderBookImbalanceProcessor(
            imbalance_threshold=0.30,
        )
        self.tick_velocity_processor = TickVelocityProcessor(
            velocity_threshold_60s=0.015,
        )
        self.deribit_pcr_processor = DeribitPCRProcessor(
            bullish_pcr_threshold=1.20,
            bearish_pcr_threshold=0.70,
        )

        # --- Fusion engine ---
        self.fusion_engine = get_fusion_engine()

        # --- Risk engine: wired with close callback ---
        self.risk_engine = get_risk_engine(initial_capital=initial_capital)
        self.risk_engine.set_close_callback(self._on_risk_close)

        # Price history (rolling, for signal processors)
        self.price_history: deque = deque(maxlen=100)

        # Tick buffer for TickVelocityProcessor (last 90 s of ticks)
        self._tick_buffer: deque = deque(maxlen=300)

        # Current market data
        self._current_price: Optional[Decimal] = None
        self._spot_price: Optional[Decimal] = None
        self._sentiment_score: Optional[float] = None
        self._yes_token_id: Optional[str] = None
        self._market_expiry: Optional[datetime] = None

        # Open positions: position_id → dict
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        self._is_running = False
        self._last_decision_time: Optional[datetime] = None

        self._signals_processed = 0
        self._trades_executed = 0
        self._total_pnl = Decimal("0")

        logger.info(
            f"Initialized 15-Min BTC Strategy: "
            f"max_position=${max_position_size}, "
            f"SL={stop_loss_pct:.0%}, TP={take_profit_pct:.0%}, "
            f"R:R={take_profit_pct/stop_loss_pct:.2f}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._is_running:
            logger.warning("Strategy already running")
            return
        self._is_running = True
        logger.info("Strategy started")
        asyncio.create_task(self._decision_loop())

    async def stop(self) -> None:
        self._is_running = False
        logger.info("Strategy stopped")

    # ------------------------------------------------------------------
    # Market data updates (called externally)
    # ------------------------------------------------------------------

    def update_market_data(
        self,
        price: Decimal,
        spot_price: Optional[Decimal] = None,
        sentiment: Optional[float] = None,
        yes_token_id: Optional[str] = None,
        market_expiry: Optional[datetime] = None,
    ) -> None:
        self._current_price = price
        self.price_history.append(price)
        self._tick_buffer.append({"ts": datetime.now(), "price": price})

        if spot_price is not None:
            self._spot_price = spot_price
        if sentiment is not None:
            self._sentiment_score = sentiment
        if yes_token_id is not None:
            self._yes_token_id = yes_token_id
        if market_expiry is not None:
            self._market_expiry = market_expiry

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    async def _decision_loop(self) -> None:
        while self._is_running:
            try:
                await self._wait_for_next_interval()
                await self._make_decision()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(60)

    async def _wait_for_next_interval(self) -> None:
        now = datetime.now()
        minutes_past = now.minute % INTERVAL_MINUTES
        wait_minutes = INTERVAL_MINUTES if minutes_past == 0 else INTERVAL_MINUTES - minutes_past
        next_time = (now + timedelta(minutes=wait_minutes)).replace(second=0, microsecond=0)
        wait_seconds = (next_time - now).total_seconds()
        logger.info(f"Waiting {wait_seconds:.0f}s until next decision ({next_time.strftime('%H:%M')})")
        await asyncio.sleep(wait_seconds)

    async def _make_decision(self) -> None:
        logger.info("=" * 60)
        logger.info("MAKING TRADING DECISION")
        logger.info("=" * 60)

        if not self._current_price:
            logger.warning("No current price data available")
            return

        # Update existing positions (triggers SL/TP/time-stop via Risk Engine)
        for pos_id in list(self.open_positions):
            self.risk_engine.update_position(pos_id, self._current_price)

        signals = self._process_signals()
        if not signals:
            logger.info("No signals generated")
            return

        logger.info(f"Generated {len(signals)} signals")
        for sig in signals:
            logger.info(
                f"  [{sig.source}] {sig.direction}: "
                f"score={sig.score:.1f}, confidence={sig.confidence:.2%}"
            )

        fused = self.fusion_engine.fuse_signals(
            signals,
            min_signals=2,   # Fixed: require at least 2 independent signals
            min_score=60.0,
        )
        if not fused:
            logger.info("No actionable fused signal (need ≥2 signals, score ≥60)")
            return

        logger.info(
            f"FUSED SIGNAL: {fused.direction} "
            f"(score={fused.score:.1f}, confidence={fused.confidence:.2%}, "
            f"from {fused.num_signals} signals)"
        )

        if not fused.is_actionable:
            logger.info("Fused signal not strong enough to trade")
            return

        if len(self.open_positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return

        await self._execute_trade(fused)
        self._last_decision_time = datetime.now()

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def _process_signals(self) -> list:
        signals = []

        if len(self.price_history) < 20:
            logger.debug("Not enough price history yet")
            return signals

        metadata: Dict[str, Any] = {}
        if self._spot_price is not None:
            metadata["spot_price"] = float(self._spot_price)
        if self._sentiment_score is not None:
            metadata["sentiment_score"] = self._sentiment_score
        if self._yes_token_id:
            metadata["yes_token_id"] = self._yes_token_id
        metadata["tick_buffer"] = list(self._tick_buffer)

        history = list(self.price_history)

        # 1. Spike Detection (weight 0.40)
        sig = self.spike_detector.process(self._current_price, history, metadata)
        if sig:
            signals.append(sig)

        # 2. Sentiment (weight 0.15 — reduced, slow daily indicator)
        sig = self.sentiment_processor.process(self._current_price, history, metadata)
        if sig:
            signals.append(sig)

        # 3. Price Divergence (weight 0.25)
        if self._spot_price is not None:
            sig = self.divergence_processor.process(self._current_price, history, metadata)
            if sig:
                signals.append(sig)

        # 4. Order Book Imbalance (weight 0.10)
        if self._yes_token_id:
            sig = self.orderbook_processor.process(self._current_price, history, metadata)
            if sig:
                signals.append(sig)

        # 5. Tick Velocity (weight 0.10)
        sig = self.tick_velocity_processor.process(self._current_price, history, metadata)
        if sig:
            signals.append(sig)

        # 6. Deribit PCR (weight 0.10 — but deduplicated against Sentiment)
        pcr_sig = self.deribit_pcr_processor.process(self._current_price, history, metadata)
        if pcr_sig:
            # Only add PCR if Sentiment didn't already fire in the same direction,
            # preventing double-counting of two correlated contrarian indicators.
            sentiment_directions = {
                str(s.direction) for s in signals if s.source == "SentimentAnalysis"
            }
            if str(pcr_sig.direction) not in sentiment_directions:
                signals.append(pcr_sig)
            else:
                logger.debug(
                    "DeribitPCR skipped: Sentiment already covers this direction "
                    "(avoiding double-counting)"
                )

        self._signals_processed += len(signals)
        return signals

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    async def _execute_trade(self, signal: FusedSignal) -> None:
        logger.info("=" * 60)
        logger.info("EXECUTING TRADE")
        logger.info("=" * 60)

        # Let Risk Engine calculate signal-scaled size
        size = self.risk_engine.calculate_position_size(
            signal_confidence=signal.confidence,
            signal_score=signal.score,
        )
        if size <= Decimal("0"):
            logger.info("Calculated position size too small — skipping trade")
            return

        # Validate against all risk limits
        direction_str = "long" if "BULLISH" in str(signal.direction).upper() else "short"
        ok, reason = self.risk_engine.validate_new_position(size, direction_str, self._current_price)
        if not ok:
            logger.warning(f"Risk Engine blocked trade: {reason}")
            return

        entry = self._current_price
        if "BULLISH" in str(signal.direction).upper():
            stop_loss = entry * Decimal(str(1 - self.stop_loss_pct))
            take_profit = entry * Decimal(str(1 + self.take_profit_pct))
        else:
            stop_loss = entry * Decimal(str(1 + self.stop_loss_pct))
            take_profit = entry * Decimal(str(1 - self.take_profit_pct))

        # Market expires at next 15-min boundary
        expiry = self._market_expiry or (
            datetime.now() + timedelta(minutes=INTERVAL_MINUTES)
        )

        pos_id = f"pos_{datetime.now().timestamp():.0f}"

        # Register with Risk Engine (enables SL/TP/time-stop monitoring)
        self.risk_engine.add_position(
            position_id=pos_id,
            size=size,
            entry_price=entry,
            direction=direction_str,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expiry_time=expiry,
        )

        self.open_positions[pos_id] = {
            "id": pos_id,
            "direction": direction_str,
            "entry_price": entry,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "expiry_time": expiry,
            "entry_time": datetime.now(),
            "signal_score": signal.score,
            "status": "open",
        }
        self._trades_executed += 1

        logger.info(f"Position opened: {signal.direction}")
        logger.info(f"  ID:          {pos_id}")
        logger.info(f"  Entry:       {float(entry):.4f}")
        logger.info(f"  Size:        ${float(size):.4f}")
        logger.info(f"  Stop Loss:   {float(stop_loss):.4f}  (-{self.stop_loss_pct:.0%})")
        logger.info(f"  Take Profit: {float(take_profit):.4f}  (+{self.take_profit_pct:.0%})")
        logger.info(f"  R:R:         {self.take_profit_pct/self.stop_loss_pct:.2f}")
        logger.info(f"  Expiry:      {expiry.strftime('%H:%M:%S')}")
        logger.info(f"  Signal Score:{signal.score:.1f} ({signal.num_signals} signals)")

    # ------------------------------------------------------------------
    # Risk Engine callback (SL / TP / time-stop)
    # ------------------------------------------------------------------

    def _on_risk_close(self, position_id: str, reason: str) -> None:
        """Called by Risk Engine when a position must be closed."""
        if position_id not in self.open_positions:
            return
        pos = self.open_positions[position_id]
        pos["status"] = f"closed:{reason}"

        # Record realized PnL in Risk Engine using current price
        if self._current_price:
            pnl = self.risk_engine.remove_position(position_id, self._current_price)
            self._total_pnl += pnl or Decimal("0")

        del self.open_positions[position_id]
        logger.info(f"Position {position_id} closed by Risk Engine: {reason}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "is_running": self._is_running,
            "signals_processed": self._signals_processed,
            "trades_executed": self._trades_executed,
            "open_positions": len(self.open_positions),
            "total_pnl": float(self._total_pnl),
            "last_decision": (
                self._last_decision_time.isoformat() if self._last_decision_time else None
            ),
            "risk_summary": self.risk_engine.get_risk_summary(),
            "processors": {
                "spike_detector": self.spike_detector.get_stats(),
                "sentiment": self.sentiment_processor.get_stats(),
                "divergence": self.divergence_processor.get_stats(),
                "orderbook": self.orderbook_processor.get_stats(),
                "tick_velocity": self.tick_velocity_processor.get_stats(),
                "deribit_pcr": self.deribit_pcr_processor.get_stats(),
            },
            "fusion_engine": self.fusion_engine.get_statistics(),
        }


# Singleton instance
_strategy_instance = None


def get_btc_strategy() -> BTCStrategy15Min:
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = BTCStrategy15Min()
    return _strategy_instance
