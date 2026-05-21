"""
Abstract base class for prediction market clients.

Both PolymarketClient and ManifoldClient implement this interface so that
the ExecutionEngine and strategy are completely platform-agnostic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class MarketInfo:
    """Normalised market descriptor returned by all clients."""
    market_id: str
    question: str
    platform: str                        # "polymarket" | "manifold"
    probability: Decimal                 # current YES probability (0-1)
    volume: Decimal                      # total traded volume (platform currency)
    close_time: Optional[datetime]
    yes_token_id: Optional[str] = None   # Polymarket only
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBook:
    """Normalised order book (or AMM snapshot for platforms without CLOB)."""
    market_id: str
    timestamp: datetime
    bids: List[Dict[str, Decimal]]   # [{"price": ..., "size": ...}, ...]
    asks: List[Dict[str, Decimal]]
    is_amm: bool = False             # True when bids/asks are synthetic (AMM)


@dataclass
class TradeRecord:
    """Normalised executed trade."""
    trade_id: str
    market_id: str
    side: str           # "buy" | "sell"
    outcome: str        # "YES" | "NO"
    price: Decimal
    size: Decimal       # platform currency units
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMarketClient(ABC):
    """
    Abstract interface every prediction-market client must implement.

    Platform-specific quirks (authentication, order types, currency) live
    in the subclass. The ExecutionEngine only talks to this interface.
    """

    # Human-readable name used in logs / stats
    platform_name: str = "unknown"

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def connect(self) -> bool:
        """Authenticate and verify connectivity. Returns True on success."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connections and release resources."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the client is authenticated and ready."""

    # ------------------------------------------------------------------ #
    # Market discovery                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_btc_markets(self, limit: int = 10) -> List[MarketInfo]:
        """Return open BTC price-prediction markets, newest first."""

    @abstractmethod
    async def get_market_price(self, market_id: str) -> Optional[Decimal]:
        """Current YES probability for *market_id* (0-1 range)."""

    @abstractmethod
    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """
        Order book for *market_id*.
        For AMM platforms this returns a synthetic single-level snapshot
        derived from the current probability.
        """

    # ------------------------------------------------------------------ #
    # Trading                                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def place_order(
        self,
        market_id: str,
        outcome: str,          # "YES" | "NO"
        size: Decimal,         # amount in platform currency
        price: Optional[Decimal] = None,  # limit price; None → best available
    ) -> Optional[str]:
        """Place an order. Returns platform order/bet ID on success."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""

    # ------------------------------------------------------------------ #
    # Account                                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Account balance. Key is currency symbol (e.g. "USDC", "MANA")."""

    @abstractmethod
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """All currently open / unmatched orders."""

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Current token/share positions (excluding cash balance)."""

    @abstractmethod
    async def get_trades(self, limit: int = 100) -> List[TradeRecord]:
        """Recent filled trades, most recent first."""
