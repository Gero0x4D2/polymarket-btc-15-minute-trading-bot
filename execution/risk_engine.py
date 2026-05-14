"""
Risk Engine
Manages position sizing, risk limits, and portfolio constraints
"""
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: Decimal  # Max USD per position
    max_total_exposure: Decimal  # Max total USD exposure
    max_positions: int  # Max concurrent positions
    max_drawdown_pct: float  # Max drawdown % before stop
    max_loss_per_day: Decimal  # Max daily loss
    max_leverage: float = 1.0  # Max leverage (1.0 = no leverage)


@dataclass
class PositionRisk:
    """Risk assessment for a position."""
    position_id: str
    current_size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    risk_level: RiskLevel
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    time_held: float  # seconds
    expiry_time: Optional[datetime]  # market expiry for time-based stop
    metadata: Dict[str, Any]


class RiskEngine:
    """
    Risk management engine.

    Enforces:
    - Position size limits (signal-scaled, capped at max)
    - Portfolio exposure limits
    - Drawdown controls
    - Loss limits
    - Time-based stops (closes positions before market expiry)
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_capital: Decimal = Decimal("10.0"),
        close_position_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.limits = limits or RiskLimits(
            max_position_size=Decimal("1.0"),
            max_total_exposure=Decimal("10.0"),
            max_positions=5,
            max_drawdown_pct=0.15,
            max_loss_per_day=Decimal("5.0"),
            max_leverage=1.0,
        )

        # Callback invoked when engine decides to close a position (SL/TP/time-stop).
        # Signature: close_position_callback(position_id, reason)
        self._close_cb = close_position_callback

        self._positions: Dict[str, PositionRisk] = {}

        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        # Use actual initial capital so drawdown maths are meaningful
        self._peak_balance = initial_capital
        self._current_balance = initial_capital

        self._alerts: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized Risk Engine: "
            f"max_position=${self.limits.max_position_size}, "
            f"max_exposure=${self.limits.max_total_exposure}, "
            f"initial_capital=${initial_capital}"
        )

    def validate_new_position(
        self,
        size: Decimal,
        direction: str,
        current_price: Decimal,
    ) -> tuple[bool, Optional[str]]:
        if size > self.limits.max_position_size:
            return False, f"Position size ${size} exceeds max ${self.limits.max_position_size}"

        if len(self._positions) >= self.limits.max_positions:
            return False, f"Max positions reached ({self.limits.max_positions})"

        current_exposure = self.get_total_exposure()
        new_exposure = current_exposure + size
        if new_exposure > self.limits.max_total_exposure:
            return False, (
                f"Total exposure ${new_exposure} would exceed max ${self.limits.max_total_exposure}"
            )

        if self._daily_pnl < -self.limits.max_loss_per_day:
            return False, f"Daily loss limit reached (${abs(self._daily_pnl):.2f})"

        drawdown = self.get_current_drawdown()
        if drawdown > self.limits.max_drawdown_pct:
            return False, f"Drawdown {drawdown:.1%} exceeds max {self.limits.max_drawdown_pct:.1%}"

        return True, None

    def calculate_position_size(
        self,
        signal_confidence: float,
        signal_score: float,
        current_price: Decimal,
        risk_percent: float = 0.02,
    ) -> Decimal:
        """
        Calculate position size scaled by signal quality, capped at max.

        Uses 2% of current balance as base, then scales by confidence × score.
        Result is always in [min_bet, max_position_size].
        """
        base = self._current_balance * Decimal(str(risk_percent))
        quality = Decimal(str(signal_confidence)) * Decimal(str(signal_score / 100))
        size = base * quality

        # Hard cap at configured maximum
        size = min(size, self.limits.max_position_size)

        # Minimum viable bet: $0.10 — below this don't bother placing an order
        min_bet = Decimal("0.10")
        if size < min_bet:
            logger.info(
                f"Calculated size ${float(size):.4f} below minimum ${float(min_bet):.2f} — skipping"
            )
            return Decimal("0")

        logger.info(
            f"Position size: ${float(size):.4f} "
            f"(balance=${float(self._current_balance):.2f}, "
            f"confidence={signal_confidence:.2%}, score={signal_score:.1f})"
        )
        return size

    def add_position(
        self,
        position_id: str,
        size: Decimal,
        entry_price: Decimal,
        direction: str,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        expiry_time: Optional[datetime] = None,
    ) -> None:
        position = PositionRisk(
            position_id=position_id,
            current_size=size,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl=Decimal("0"),
            risk_level=RiskLevel.LOW,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_held=0.0,
            expiry_time=expiry_time,
            metadata={
                "direction": direction,
                "entry_time": datetime.now(),
            }
        )
        self._positions[position_id] = position
        self._daily_trades += 1
        logger.info(f"Added position: {position_id} (${size:.2f} @ {entry_price:.4f})")

    def update_position(
        self,
        position_id: str,
        current_price: Decimal,
    ) -> Optional[PositionRisk]:
        if position_id not in self._positions:
            return None

        position = self._positions[position_id]
        position.current_price = current_price

        direction = position.metadata.get("direction", "long")
        if direction == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price

        position.unrealized_pnl = position.current_size * pnl_pct

        entry_time = position.metadata.get("entry_time", datetime.now())
        position.time_held = (datetime.now() - entry_time).total_seconds()

        position.risk_level = self._assess_risk_level(position)

        # --- Stop Loss ---
        if position.stop_loss and self._check_stop_loss(position, current_price):
            logger.warning(f"Stop loss triggered for {position_id} at {current_price:.4f}")
            self._create_alert("STOP_LOSS", f"Stop loss hit for {position_id}", RiskLevel.HIGH)
            self._trigger_close(position_id, "stop_loss")

        # --- Take Profit ---
        elif position.take_profit and self._check_take_profit(position, current_price):
            logger.info(f"Take profit triggered for {position_id} at {current_price:.4f}")
            self._create_alert("TAKE_PROFIT", f"Take profit hit for {position_id}", RiskLevel.LOW)
            self._trigger_close(position_id, "take_profit")

        # --- Time-based stop: close 60 s before expiry ---
        elif position.expiry_time:
            seconds_to_expiry = (position.expiry_time - datetime.now()).total_seconds()
            if seconds_to_expiry <= 60:
                logger.warning(
                    f"Time stop triggered for {position_id}: "
                    f"{seconds_to_expiry:.0f}s to expiry"
                )
                self._create_alert(
                    "TIME_STOP",
                    f"Position {position_id} closed 60s before expiry",
                    RiskLevel.MEDIUM,
                )
                self._trigger_close(position_id, "time_stop")

        return position

    def _trigger_close(self, position_id: str, reason: str) -> None:
        """Invoke the registered callback to close a position."""
        if self._close_cb:
            try:
                self._close_cb(position_id, reason)
            except Exception as e:
                logger.error(f"close_position_callback raised for {position_id}: {e}")
        else:
            logger.warning(
                f"No close_position_callback registered — "
                f"position {position_id} flagged for closure ({reason}) but NOT closed automatically"
            )

    def remove_position(
        self,
        position_id: str,
        exit_price: Decimal,
    ) -> Optional[Decimal]:
        if position_id not in self._positions:
            return None

        position = self._positions[position_id]
        direction = position.metadata.get("direction", "long")

        if direction == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        realized_pnl = position.current_size * pnl_pct
        self._current_balance += realized_pnl
        self._daily_pnl += realized_pnl

        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance

        del self._positions[position_id]
        logger.info(
            f"Closed position: {position_id} "
            f"P&L: ${realized_pnl:+.4f} ({pnl_pct:+.2%})"
        )
        return realized_pnl

    def _assess_risk_level(self, position: PositionRisk) -> RiskLevel:
        pnl_pct = float(
            position.unrealized_pnl / position.current_size
            if position.current_size > 0
            else Decimal("0")
        )
        if pnl_pct < -0.10:
            return RiskLevel.CRITICAL
        elif pnl_pct < -0.05:
            return RiskLevel.HIGH
        elif pnl_pct < -0.02:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _check_stop_loss(self, position: PositionRisk, current_price: Decimal) -> bool:
        if not position.stop_loss:
            return False
        direction = position.metadata.get("direction", "long")
        return current_price <= position.stop_loss if direction == "long" else current_price >= position.stop_loss

    def _check_take_profit(self, position: PositionRisk, current_price: Decimal) -> bool:
        if not position.take_profit:
            return False
        direction = position.metadata.get("direction", "long")
        return current_price >= position.take_profit if direction == "long" else current_price <= position.take_profit

    def _create_alert(self, alert_type: str, message: str, risk_level: RiskLevel) -> None:
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message,
            "risk_level": risk_level.value,
        }
        self._alerts.append(alert)
        logger.warning(f"[{risk_level.value.upper()}] {alert_type}: {message}")

    def get_total_exposure(self) -> Decimal:
        return sum(pos.current_size for pos in self._positions.values())

    def get_total_unrealized_pnl(self) -> Decimal:
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def get_current_drawdown(self) -> float:
        if self._peak_balance == 0:
            return 0.0
        return float((self._peak_balance - self._current_balance) / self._peak_balance)

    def get_risk_summary(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(),
            "positions": {
                "count": len(self._positions),
                "max_allowed": self.limits.max_positions,
            },
            "exposure": {
                "current": float(self.get_total_exposure()),
                "max_allowed": float(self.limits.max_total_exposure),
                "utilization_pct": float(
                    self.get_total_exposure() / self.limits.max_total_exposure * 100
                ) if self.limits.max_total_exposure > 0 else 0,
            },
            "pnl": {
                "daily": float(self._daily_pnl),
                "unrealized": float(self.get_total_unrealized_pnl()),
                "daily_limit": float(self.limits.max_loss_per_day),
            },
            "balance": {
                "current": float(self._current_balance),
                "peak": float(self._peak_balance),
                "drawdown_pct": self.get_current_drawdown() * 100,
                "max_drawdown_pct": self.limits.max_drawdown_pct * 100,
            },
            "daily_stats": {
                "trades": self._daily_trades,
                "pnl": float(self._daily_pnl),
            },
            "alerts": len(
                [a for a in self._alerts if (datetime.now() - a["timestamp"]).seconds < 3600]
            ),
        }

    def reset_daily_stats(self) -> None:
        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        logger.info("Reset daily statistics")

    def set_close_callback(self, cb: Callable[[str, str], None]) -> None:
        """Register or replace the callback used to close positions."""
        self._close_cb = cb


# Singleton instance
_risk_engine_instance = None

def get_risk_engine(
    initial_capital: Decimal = Decimal("10.0"),
) -> "RiskEngine":
    global _risk_engine_instance
    if _risk_engine_instance is None:
        _risk_engine_instance = RiskEngine(initial_capital=initial_capital)
    return _risk_engine_instance
