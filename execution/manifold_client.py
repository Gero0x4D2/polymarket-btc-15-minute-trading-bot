"""
Manifold Markets Client

Manifold uses play-money (Mana) and an AMM model instead of a CLOB.
There are no real bids/asks — the market price is the current probability
returned by the AMM. This makes it ideal for testing the full bot logic
(signals → fusion → risk engine → order placement) without real capital.

Authentication:
  Set MANIFOLD_API_KEY in your .env file.
  Get your key at: https://manifold.markets/profile (under API key)

Currency:
  All sizes are in Mana (M). The default free balance is M500 on signup.

API reference: https://docs.manifold.markets/api
"""
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from loguru import logger

from execution.base_client import BaseMarketClient, MarketInfo, OrderBook, TradeRecord


_BASE = "https://api.manifold.markets/v0"


class ManifoldClient(BaseMarketClient):
    """
    Manifold Markets client implementing BaseMarketClient.

    Key differences from Polymarket:
    - Play money (Mana) — no real capital at risk
    - AMM pricing: no order book, price = current probability
    - Bets are instant-fill (no partial fills / pending state)
    - "YES" / "NO" maps to BULLISH / BEARISH in the bot
    """

    platform_name = "manifold"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("MANIFOLD_API_KEY", "")
        self._connected = False
        self._user_id: Optional[str] = None
        self._username: Optional[str] = None

        if not self._api_key:
            logger.warning(
                "MANIFOLD_API_KEY not set — read-only mode. "
                "Set it in .env to enable bet placement."
            )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Key {self._api_key}"
        return h

    def _get(self, path: str, params: Dict = None) -> Any:
        resp = httpx.get(f"{_BASE}{path}", params=params, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: Dict) -> Any:
        resp = httpx.post(f"{_BASE}{path}", json=body, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def connect(self) -> bool:
        try:
            if self._api_key:
                me = self._get("/me")
                self._user_id = me.get("id")
                self._username = me.get("username")
                balance = me.get("balance", 0)
                logger.info(
                    f"Connected to Manifold Markets as @{self._username} "
                    f"(balance: M{balance:.0f})"
                )
            else:
                # Verify public API is reachable with a lightweight call
                self._get("/markets", {"limit": 1})
                logger.info("Connected to Manifold Markets (read-only — no API key)")

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Manifold: {e}")
            return False

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Disconnected from Manifold Markets")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    # Market discovery                                                     #
    # ------------------------------------------------------------------ #

    async def get_btc_markets(self, limit: int = 10) -> List[MarketInfo]:
        """Search for open BINARY markets containing 'BTC' in the question."""
        try:
            raw = self._get(
                "/search-markets",
                {"term": "BTC price", "contractType": "BINARY", "isOpen": "true", "limit": limit},
            )

            markets: List[MarketInfo] = []
            for m in raw:
                if m.get("outcomeType") != "BINARY":
                    continue
                close_time = None
                if m.get("closeTime"):
                    try:
                        close_time = datetime.fromtimestamp(
                            m["closeTime"] / 1000, tz=timezone.utc
                        )
                    except (ValueError, OSError):
                        pass

                markets.append(MarketInfo(
                    market_id=m["id"],
                    question=m.get("question", ""),
                    platform=self.platform_name,
                    probability=Decimal(str(m.get("probability", 0.5))),
                    volume=Decimal(str(m.get("volume", 0))),
                    close_time=close_time,
                    yes_token_id=None,  # Manifold has no token IDs
                    metadata={
                        "url": m.get("url"),
                        "creatorUsername": m.get("creatorUsername"),
                        "totalLiquidity": m.get("totalLiquidity"),
                    },
                ))

            logger.info(f"Found {len(markets)} BTC markets on Manifold")
            return markets

        except Exception as e:
            logger.error(f"Error fetching Manifold BTC markets: {e}")
            return []

    async def get_market_price(self, market_id: str) -> Optional[Decimal]:
        """Current YES probability for the given market."""
        try:
            m = self._get(f"/market/{market_id}")
            return Decimal(str(m.get("probability", 0.5)))
        except Exception as e:
            logger.error(f"Error fetching Manifold price for {market_id}: {e}")
            return None

    async def get_orderbook(self, market_id: str) -> Optional[OrderBook]:
        """
        Manifold uses an AMM — there is no real order book.
        Returns a synthetic single-level snapshot so that downstream code
        (e.g. OrderBookImbalanceProcessor) receives a valid structure.
        The imbalance will always be ~0 (balanced AMM), which is correct:
        the signal simply won't fire, which is the honest answer.
        """
        try:
            prob = await self.get_market_price(market_id)
            if prob is None:
                return None

            # Synthetic spread: ±0.5 % around current probability
            spread = Decimal("0.005")
            bid = max(Decimal("0.01"), prob - spread)
            ask = min(Decimal("0.99"), prob + spread)

            return OrderBook(
                market_id=market_id,
                timestamp=datetime.now(timezone.utc),
                bids=[{"price": bid, "size": Decimal("100")}],
                asks=[{"price": ask, "size": Decimal("100")}],
                is_amm=True,
            )

        except Exception as e:
            logger.error(f"Error constructing Manifold order book for {market_id}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Trading                                                              #
    # ------------------------------------------------------------------ #

    async def place_order(
        self,
        market_id: str,
        outcome: str,
        size: Decimal,
        price: Optional[Decimal] = None,
    ) -> Optional[str]:
        """
        Place a bet on Manifold.

        Args:
            market_id: Manifold market ID (e.g. "abc123xyz")
            outcome:   "YES" or "NO"
            size:      Amount in Mana (M)
            price:     Limit probability; None → market bet (instant fill)

        Returns:
            Bet ID string on success, None on failure.
        """
        if not self._api_key:
            logger.error("Cannot place bet: MANIFOLD_API_KEY not set")
            return None

        if outcome.upper() not in ("YES", "NO"):
            logger.error(f"Invalid outcome '{outcome}' — must be YES or NO")
            return None

        body: Dict[str, Any] = {
            "contractId": market_id,
            "outcome": outcome.upper(),
            "amount": float(size),
        }
        if price is not None:
            # Limit bet: only fills if probability crosses *price*
            body["limitProb"] = float(price)

        try:
            resp = self._post("/bet", body)
            bet_id = resp.get("betId") or resp.get("id")
            if bet_id:
                prob_after = resp.get("probAfter", "?")
                logger.info(
                    f"Manifold bet placed: {bet_id} "
                    f"{outcome.upper()} M{float(size):.1f} "
                    f"(prob after: {prob_after})"
                )
                return str(bet_id)

            logger.error(f"Manifold bet response missing betId: {resp}")
            return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Manifold bet failed [{e.response.status_code}]: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Error placing Manifold bet: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a limit bet.
        Only limit bets (those with limitProb set) can be cancelled.
        Market bets are instant-fill and cannot be undone.
        """
        if not self._api_key:
            logger.error("Cannot cancel bet: MANIFOLD_API_KEY not set")
            return False

        try:
            self._post(f"/bet/cancel/{order_id}", {})
            logger.info(f"Manifold limit bet cancelled: {order_id}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Bet {order_id} not found (already filled or cancelled)")
            else:
                logger.error(f"Cancel failed [{e.response.status_code}]: {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling Manifold bet: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Account                                                              #
    # ------------------------------------------------------------------ #

    async def get_balance(self) -> Dict[str, Decimal]:
        """Returns current Mana balance under key 'MANA'."""
        if not self._api_key:
            return {"MANA": Decimal("0")}
        try:
            me = self._get("/me")
            return {"MANA": Decimal(str(me.get("balance", 0)))}
        except Exception as e:
            logger.error(f"Error fetching Manifold balance: {e}")
            return {"MANA": Decimal("0")}

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Return unfilled limit bets."""
        if not self._api_key or not self._user_id:
            return []
        try:
            bets = self._get("/bets", {"userId": self._user_id, "filterChallenges": False})
            return [
                {
                    "order_id": b["id"],
                    "market_id": b.get("contractId"),
                    "outcome": b.get("outcome"),
                    "amount": Decimal(str(b.get("amount", 0))),
                    "limit_prob": b.get("limitProb"),
                    "timestamp": datetime.fromtimestamp(
                        b["createdTime"] / 1000, tz=timezone.utc
                    ) if b.get("createdTime") else None,
                }
                for b in bets
                if b.get("limitProb") and not b.get("isFilled") and not b.get("isCancelled")
            ]
        except Exception as e:
            logger.error(f"Error fetching Manifold open orders: {e}")
            return []

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return markets where the user holds a net position
        (total YES shares - total NO shares ≠ 0).
        """
        if not self._api_key or not self._user_id:
            return []
        try:
            # Manifold tracks positions as portfolio entries
            portfolio = self._get(f"/user/{self._username}/portfolio")
            positions = []
            for entry in portfolio:
                net = entry.get("netShares", 0)
                if net != 0:
                    positions.append({
                        "market_id": entry.get("contractId"),
                        "question": entry.get("question"),
                        "net_shares": Decimal(str(net)),
                        "direction": "long" if net > 0 else "short",
                    })
            return positions
        except Exception as e:
            logger.error(f"Error fetching Manifold positions: {e}")
            return []

    async def get_trades(self, limit: int = 100) -> List[TradeRecord]:
        """Return recent filled bets as normalised TradeRecord objects."""
        if not self._api_key or not self._user_id:
            return []
        try:
            bets = self._get("/bets", {"userId": self._user_id, "limit": limit})
            records = []
            for b in bets:
                if not b.get("amount"):
                    continue
                outcome = b.get("outcome", "YES")
                records.append(TradeRecord(
                    trade_id=b["id"],
                    market_id=b.get("contractId", ""),
                    side="buy",   # Manifold bets are always "buy" of an outcome
                    outcome=outcome,
                    price=Decimal(str(b.get("probBefore", 0.5))),
                    size=Decimal(str(abs(b.get("amount", 0)))),
                    timestamp=datetime.fromtimestamp(
                        b["createdTime"] / 1000, tz=timezone.utc
                    ) if b.get("createdTime") else datetime.now(timezone.utc),
                    metadata={"shares": b.get("shares"), "probAfter": b.get("probAfter")},
                ))
            return records
        except Exception as e:
            logger.error(f"Error fetching Manifold trades: {e}")
            return []


# ------------------------------------------------------------------ #
# Factory / singleton                                                  #
# ------------------------------------------------------------------ #

_manifold_instance: Optional[ManifoldClient] = None


def get_manifold_client(force_new: bool = False) -> ManifoldClient:
    global _manifold_instance
    if _manifold_instance is None or force_new:
        _manifold_instance = ManifoldClient()
    return _manifold_instance
