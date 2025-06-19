"""
Live Trading Execution System for Hyperion V2
Supports both Testnet and Mainnet trading
"""

import ccxt
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import json
import hmac
import hashlib
import time
from dataclasses import dataclass
import websockets
from collections import defaultdict

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading modes"""

    TESTNET = "testnet"
    MAINNET = "mainnet"
    PAPER = "paper"


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order data structure"""

    order_id: str
    symbol: str
    side: str
    type: OrderType
    size: float
    price: Optional[float]
    status: OrderStatus
    filled_size: float = 0.0
    average_price: float = 0.0
    timestamp: datetime = None
    exchange_order_id: Optional[str] = None
    metadata: Dict = None


@dataclass
class ExecutionResult:
    """Execution result"""

    success: bool
    order: Optional[Order]
    error: Optional[str]
    execution_time: float
    slippage: float = 0.0


class LiveExecutionSystem:
    """
    Advanced Live Trading Execution System

    Features:
    - Testnet and Mainnet support
    - Multiple exchange support
    - WebSocket order updates
    - Smart order routing
    - Execution algorithms (TWAP, VWAP, Iceberg)
    - Slippage protection
    - Rate limiting
    - Error recovery
    """

    def __init__(self, config: Dict, mode: TradingMode = TradingMode.TESTNET):
        self.config = config
        self.mode = mode
        self.exchanges = {}
        self.active_orders = {}
        self.position_tracker = defaultdict(lambda: {"size": 0, "avg_price": 0})
        self.websocket_connections = {}
        self.rate_limits = defaultdict(
            lambda: {"requests": 0, "reset_time": time.time()}
        )

        # Initialize exchanges
        self._initialize_exchanges()

        # Execution settings
        self.max_slippage = config.get("max_slippage", 0.002)  # 0.2%
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        self.order_timeout = config.get("order_timeout", 30)  # seconds

        logger.info(f"Live Execution System initialized in {mode.value} mode")

    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        exchange_configs = self.config.get("exchanges", {})

        for exchange_name, exchange_config in exchange_configs.items():
            try:
                self._setup_exchange(exchange_name, exchange_config)
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")

    def _setup_exchange(self, exchange_name: str, config: Dict):
        """Setup individual exchange"""
        exchange_class = getattr(ccxt, exchange_name)

        # Base configuration
        exchange_config = {
            "apiKey": config["api_key"],
            "secret": config["api_secret"],
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",  # For perpetual contracts
                "adjustForTimeDifference": True,
            },
        }

        # Add passphrase if required (e.g., for OKX)
        if "passphrase" in config:
            exchange_config["password"] = config["passphrase"]

        # Configure for testnet or mainnet
        if self.mode == TradingMode.TESTNET:
            if exchange_name == "binance":
                exchange_config["urls"] = {
                    "api": {
                        "public": "https://testnet.binance.vision/api",
                        "private": "https://testnet.binance.vision/api",
                    }
                }
            elif exchange_name == "bybit":
                exchange_config["urls"] = {
                    "api": {
                        "public": "https://api-testnet.bybit.com",
                        "private": "https://api-testnet.bybit.com",
                    }
                }

        # Create exchange instance
        exchange = exchange_class(exchange_config)

        # Load markets
        exchange.load_markets()

        # Store exchange
        self.exchanges[exchange_name] = exchange

        logger.info(f"Initialized {exchange_name} in {self.mode.value} mode")

    async def connect_websockets(self):
        """Connect to exchange WebSockets for real-time updates"""
        tasks = []

        for exchange_name in self.exchanges:
            if exchange_name == "binance":
                task = asyncio.create_task(self._connect_binance_ws())
                tasks.append(task)
            elif exchange_name == "bybit":
                task = asyncio.create_task(self._connect_bybit_ws())
                tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_binance_ws(self):
        """Connect to Binance WebSocket"""
        try:
            ws_url = (
                "wss://stream.binance.com:9443/ws"
                if self.mode == TradingMode.MAINNET
                else "wss://testnet.binance.vision/ws"
            )

            async with websockets.connect(ws_url) as ws:
                self.websocket_connections["binance"] = ws

                # Subscribe to user data stream
                listen_key = await self._get_binance_listen_key()

                subscribe_msg = {"method": "SUBSCRIBE", "params": [listen_key], "id": 1}

                await ws.send(json.dumps(subscribe_msg))

                # Listen for updates
                async for message in ws:
                    await self._handle_binance_ws_message(json.loads(message))

        except Exception as e:
            logger.error(f"Binance WebSocket error: {e}")
            # Reconnect after delay
            await asyncio.sleep(5)
            await self._connect_binance_ws()

    async def _handle_binance_ws_message(self, message: Dict):
        """Handle Binance WebSocket message"""
        if message.get("e") == "executionReport":
            # Order update
            order_id = message.get("c")  # Client order ID

            if order_id in self.active_orders:
                order = self.active_orders[order_id]

                # Update order status
                status_map = {
                    "NEW": OrderStatus.OPEN,
                    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                    "FILLED": OrderStatus.FILLED,
                    "CANCELED": OrderStatus.CANCELLED,
                    "REJECTED": OrderStatus.REJECTED,
                    "EXPIRED": OrderStatus.EXPIRED,
                }

                order.status = status_map.get(message["X"], OrderStatus.PENDING)
                order.filled_size = float(message.get("z", 0))
                order.average_price = (
                    float(message.get("Z", 0)) / order.filled_size
                    if order.filled_size > 0
                    else 0
                )

                logger.info(f"Order {order_id} updated: {order.status.value}")

    async def place_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        post_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Place order on exchange

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            order_type: Type of order
            price: Limit price
            stop_price: Stop price
            time_in_force: Order time in force
            reduce_only: Reduce only flag
            post_only: Post only flag
            client_order_id: Custom order ID

        Returns:
            ExecutionResult
        """
        start_time = time.time()

        try:
            # Validate inputs
            if exchange not in self.exchanges:
                return ExecutionResult(
                    success=False,
                    order=None,
                    error=f"Exchange {exchange} not initialized",
                    execution_time=0,
                )

            # Check rate limits
            if not self._check_rate_limit(exchange):
                await asyncio.sleep(1)  # Wait before retry

            # Generate order ID
            if not client_order_id:
                client_order_id = f"HYPERION_{int(time.time() * 1000)}"

            # Create order object
            order = Order(
                order_id=client_order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                size=size,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
                metadata={
                    "reduce_only": reduce_only,
                    "post_only": post_only,
                    "time_in_force": time_in_force,
                },
            )

            # Store order
            self.active_orders[client_order_id] = order

            # Execute based on mode
            if self.mode == TradingMode.PAPER:
                result = await self._execute_paper_order(order)
            else:
                result = await self._execute_real_order(exchange, order, stop_price)

            # Calculate execution time and slippage
            execution_time = time.time() - start_time

            if result.success and order.average_price > 0 and price:
                slippage = abs(order.average_price - price) / price
            else:
                slippage = 0

            return ExecutionResult(
                success=result.success,
                order=order,
                error=result.error,
                execution_time=execution_time,
                slippage=slippage,
            )

        except Exception as e:
            logger.error(f"Order placement error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                order=None,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _execute_real_order(
        self, exchange_name: str, order: Order, stop_price: Optional[float] = None
    ) -> ExecutionResult:
        """Execute real order on exchange"""
        exchange = self.exchanges[exchange_name]

        try:
            # Prepare order parameters
            params = {"clientOrderId": order.order_id}

            if order.metadata.get("reduce_only"):
                params["reduceOnly"] = True

            if order.metadata.get("post_only"):
                params["postOnly"] = True

            # Place order based on type
            if order.type == OrderType.MARKET:
                response = await self._async_create_order(
                    exchange,
                    order.symbol,
                    "market",
                    order.side,
                    order.size,
                    None,
                    params,
                )

            elif order.type == OrderType.LIMIT:
                response = await self._async_create_order(
                    exchange,
                    order.symbol,
                    "limit",
                    order.side,
                    order.size,
                    order.price,
                    params,
                )

            elif order.type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
                params["stopPrice"] = stop_price
                order_type = (
                    "stop_loss"
                    if order.type == OrderType.STOP_LOSS
                    else "stop_loss_limit"
                )

                response = await self._async_create_order(
                    exchange,
                    order.symbol,
                    order_type,
                    order.side,
                    order.size,
                    order.price,
                    params,
                )

            # Update order with response
            order.exchange_order_id = response.get("id")
            order.status = self._parse_order_status(response.get("status"))
            order.filled_size = float(response.get("filled", 0))
            order.average_price = float(response.get("average", 0))

            # Update position tracker
            if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                self._update_position(order)

            return ExecutionResult(
                success=True, order=order, error=None, execution_time=0
            )

        except Exception as e:
            logger.error(f"Real order execution error: {e}")
            order.status = OrderStatus.REJECTED
            return ExecutionResult(
                success=False, order=order, error=str(e), execution_time=0
            )

    async def _async_create_order(self, exchange, *args, **kwargs):
        """Async wrapper for exchange create_order"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, exchange.create_order, *args, **kwargs)

    async def _execute_paper_order(self, order: Order) -> ExecutionResult:
        """Execute paper trading order"""
        # Simulate order execution
        await asyncio.sleep(0.1)  # Simulate latency

        # Always fill market orders in paper trading
        if order.type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.average_price = order.price or 50000  # Dummy price
            order.exchange_order_id = f"PAPER_{order.order_id}"

            # Update position tracker
            self._update_position(order)

            return ExecutionResult(
                success=True, order=order, error=None, execution_time=0.1
            )

        # For limit orders, simulate partial execution
        elif order.type == OrderType.LIMIT:
            order.status = OrderStatus.OPEN
            order.exchange_order_id = f"PAPER_{order.order_id}"

            # Simulate partial fill after delay
            asyncio.create_task(self._simulate_limit_order_fill(order))

            return ExecutionResult(
                success=True, order=order, error=None, execution_time=0.1
            )

        return ExecutionResult(
            success=False, order=order, error="Unsupported order type", execution_time=0
        )

    async def _simulate_limit_order_fill(self, order: Order):
        """Simulate limit order fills for paper trading"""
        await asyncio.sleep(5)  # Wait 5 seconds

        # Simulate 50% fill
        order.filled_size = order.size * 0.5
        order.status = OrderStatus.PARTIALLY_FILLED
        order.average_price = order.price

        await asyncio.sleep(5)  # Wait another 5 seconds

        # Complete fill
        order.filled_size = order.size
        order.status = OrderStatus.FILLED

        self._update_position(order)

    def _update_position(self, order: Order):
        """Update position tracker"""
        position = self.position_tracker[order.symbol]

        if order.side == "buy":
            new_size = position["size"] + order.filled_size
            if new_size != 0:
                position["avg_price"] = (
                    position["size"] * position["avg_price"]
                    + order.filled_size * order.average_price
                ) / new_size
            position["size"] = new_size
        else:  # sell
            position["size"] -= order.filled_size
            if position["size"] <= 0:
                position["size"] = 0
                position["avg_price"] = 0

    async def cancel_order(
        self, exchange: str, order_id: str, symbol: Optional[str] = None
    ) -> bool:
        """Cancel order"""
        try:
            if self.mode == TradingMode.PAPER:
                if order_id in self.active_orders:
                    self.active_orders[order_id].status = OrderStatus.CANCELLED
                    return True
                return False

            # Real order cancellation
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                return False

            # Get order details if needed
            if not symbol and order_id in self.active_orders:
                symbol = self.active_orders[order_id].symbol

            # Cancel order
            response = await self._async_cancel_order(
                exchange_obj, self.active_orders[order_id].exchange_order_id, symbol
            )

            # Update order status
            if order_id in self.active_orders:
                self.active_orders[order_id].status = OrderStatus.CANCELLED

            return True

        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False

    async def _async_cancel_order(self, exchange, *args, **kwargs):
        """Async wrapper for exchange cancel_order"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, exchange.cancel_order, *args, **kwargs)

    async def cancel_all_orders(
        self, exchange: str, symbol: Optional[str] = None
    ) -> int:
        """Cancel all orders"""
        cancelled_count = 0

        for order_id, order in list(self.active_orders.items()):
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                if symbol and order.symbol != symbol:
                    continue

                if await self.cancel_order(exchange, order_id, order.symbol):
                    cancelled_count += 1

        return cancelled_count

    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        return dict(self.position_tracker)

    def get_open_orders(self) -> List[Order]:
        """Get open orders"""
        return [
            order
            for order in self.active_orders.values()
            if order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
        ]

    def _check_rate_limit(self, exchange: str) -> bool:
        """Check rate limits"""
        limits = self.rate_limits[exchange]
        current_time = time.time()

        # Reset counter if window passed
        if current_time > limits["reset_time"]:
            limits["requests"] = 0
            limits["reset_time"] = current_time + 60  # 1 minute window

        # Check limit (example: 1200 requests per minute for Binance)
        max_requests = self.config.get("rate_limits", {}).get(exchange, 1200)

        if limits["requests"] >= max_requests:
            return False

        limits["requests"] += 1
        return True

    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse exchange order status"""
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
        }

        return status_map.get(status.lower(), OrderStatus.PENDING)

    async def _get_binance_listen_key(self) -> str:
        """Get Binance listen key for user data stream"""
        # Implementation depends on whether using spot or futures
        # This is a placeholder
        return "dummy_listen_key"

    async def execute_twap(
        self,
        exchange: str,
        symbol: str,
        side: str,
        total_size: float,
        duration_minutes: int,
        price_limit: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Execute Time-Weighted Average Price (TWAP) order

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_size: Total size to execute
            duration_minutes: Duration in minutes
            price_limit: Price limit for orders

        Returns:
            List of execution results
        """
        results = []

        # Calculate slice parameters
        num_slices = max(duration_minutes // 5, 1)  # One order every 5 minutes
        slice_size = total_size / num_slices
        interval = (duration_minutes * 60) / num_slices

        logger.info(
            f"Starting TWAP: {total_size} {symbol} over {duration_minutes} minutes"
        )

        for i in range(num_slices):
            # Check price limit
            if price_limit:
                current_price = await self._get_current_price(exchange, symbol)
                if (side == "buy" and current_price > price_limit) or (
                    side == "sell" and current_price < price_limit
                ):
                    logger.warning(f"TWAP stopped: price limit reached")
                    break

            # Place slice order
            result = await self.place_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                size=slice_size,
                order_type=OrderType.MARKET,
            )

            results.append(result)

            # Wait for next slice (except last)
            if i < num_slices - 1:
                await asyncio.sleep(interval)

        return results

    async def execute_vwap(
        self,
        exchange: str,
        symbol: str,
        side: str,
        total_size: float,
        duration_minutes: int,
        price_limit: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Execute Volume-Weighted Average Price (VWAP) order

        Distributes orders based on historical volume patterns
        """
        results = []

        # Get volume profile
        volume_profile = await self._get_volume_profile(
            exchange, symbol, duration_minutes
        )

        # Calculate order distribution
        order_distribution = self._calculate_vwap_distribution(
            total_size, volume_profile, duration_minutes
        )

        logger.info(
            f"Starting VWAP: {total_size} {symbol} over {duration_minutes} minutes"
        )

        for order_time, order_size in order_distribution:
            # Wait until order time
            wait_time = (order_time - datetime.now()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Check price limit
            if price_limit:
                current_price = await self._get_current_price(exchange, symbol)
                if (side == "buy" and current_price > price_limit) or (
                    side == "sell" and current_price < price_limit
                ):
                    logger.warning(f"VWAP stopped: price limit reached")
                    break

            # Place order
            result = await self.place_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                size=order_size,
                order_type=OrderType.MARKET,
            )

            results.append(result)

        return results

    async def execute_iceberg(
        self,
        exchange: str,
        symbol: str,
        side: str,
        total_size: float,
        visible_size: float,
        price: float,
        price_tolerance: float = 0.001,
    ) -> List[ExecutionResult]:
        """
        Execute Iceberg order

        Shows only a portion of the total order size
        """
        results = []
        remaining_size = total_size

        logger.info(
            f"Starting Iceberg: {total_size} {symbol} with {visible_size} visible"
        )

        while remaining_size > 0:
            # Calculate next order size
            order_size = min(visible_size, remaining_size)

            # Adjust price based on market movement
            current_price = await self._get_current_price(exchange, symbol)
            adjusted_price = price

            if side == "buy":
                adjusted_price = min(price, current_price * (1 + price_tolerance))
            else:
                adjusted_price = max(price, current_price * (1 - price_tolerance))

            # Place limit order
            result = await self.place_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                size=order_size,
                order_type=OrderType.LIMIT,
                price=adjusted_price,
                post_only=True,
            )

            results.append(result)

            if result.success:
                # Wait for fill or timeout
                filled = await self._wait_for_fill(result.order.order_id, timeout=30)

                if filled:
                    remaining_size -= result.order.filled_size
                else:
                    # Cancel and retry with adjusted price
                    await self.cancel_order(exchange, result.order.order_id)

            # Small delay between orders
            await asyncio.sleep(1)

        return results

    async def _get_current_price(self, exchange: str, symbol: str) -> float:
        """Get current market price"""
        try:
            exchange_obj = self.exchanges[exchange]
            ticker = exchange_obj.fetch_ticker(symbol)
            return ticker["last"]
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0

    async def _get_volume_profile(
        self, exchange: str, symbol: str, lookback_minutes: int
    ) -> List[Tuple[datetime, float]]:
        """Get historical volume profile"""
        # Placeholder - implement based on exchange API
        return []

    def _calculate_vwap_distribution(
        self,
        total_size: float,
        volume_profile: List[Tuple[datetime, float]],
        duration_minutes: int,
    ) -> List[Tuple[datetime, float]]:
        """Calculate VWAP order distribution"""
        # Placeholder - implement VWAP distribution logic
        return []

    async def _wait_for_fill(self, order_id: str, timeout: int = 30) -> bool:
        """Wait for order to be filled"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                if order.status == OrderStatus.FILLED:
                    return True
                elif order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    return False

            await asyncio.sleep(0.5)

        return False

    async def close(self):
        """Close all connections"""
        # Cancel all open orders
        for exchange in self.exchanges:
            await self.cancel_all_orders(exchange)

        # Close WebSocket connections
        for ws in self.websocket_connections.values():
            await ws.close()

        logger.info("Live Execution System closed")


class LiveTradingBot:
    """Minimal live trading bot for unit tests."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.execution = LiveExecutionSystem(self.config)

    async def shutdown(self):
        await self.execution.close()
