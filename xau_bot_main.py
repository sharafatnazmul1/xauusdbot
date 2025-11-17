"""
XAU/USD Breakout Trading Bot - PRODUCTION VERSION
Fixed: Correct lot sizing for gold, proper error handling, drawdown protection
"""

import MetaTrader5 as mt5
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict
import json
import os
from xau_signal_generator import BreakoutSignalGenerator


class XAUTradingBot:
    def __init__(self, config_path: str = "bot_config.json"):
        self.config = self.load_config(config_path)
        
        # Core settings
        self.symbol = self.config.get("symbol", "XAUUSDm")
        self.magic_number = self.config.get("magic_number", 888888)
        self.max_spread_points = self.config.get("max_spread_points", 161)
        self.max_positions = self.config.get("max_positions", 2)
        self.risk_percent = self.config.get("risk_percent", 5.0)
        self.min_lot = self.config.get("min_lot", 0.01)
        self.max_lot = self.config.get("max_lot", 1.0)
        
        # Risk management
        self.max_daily_loss_percent = self.config.get("max_daily_loss_percent", 50.0)
        self.max_consecutive_losses = self.config.get("max_consecutive_losses", 3)
        self.max_daily_trades = self.config.get("max_daily_trades", 14)
        self.signal_cooldown_minutes = self.config.get("signal_cooldown_minutes", 20)
        
        # Trade management - Enhanced Profit Protection
        self.use_breakeven = self.config.get("use_breakeven", True)
        self.breakeven_trigger_rr = self.config.get("breakeven_trigger_rr", 1.0)  # Move to BE at 1:1
        self.use_partial_tp = self.config.get("use_partial_tp", True)  # Close 50% at 1.5:1 R:R
        self.use_trailing_stop = self.config.get("use_trailing_stop", True)  # ATR-based trailing

        # Momentum-Adaptive Trailing System
        self.use_momentum_adaptive = self.config.get("use_momentum_adaptive", True)
        self.trailing_by_momentum = self.config.get("trailing_by_momentum", {
            "strong": 0.9,    # Momentum 70-100: Wide trail for strong trends
            "medium": 0.6,    # Momentum 40-70: Normal trail
            "weak": 0.35      # Momentum 0-40: Tight trail, protect immediately
        })
        self.trailing_atr_multiplier = self.config.get("trailing_atr_multiplier", 0.6)  # Fallback if adaptive disabled

        # Early Exit Detection (Bulletproof protection against false breakouts)
        self.use_early_exit = self.config.get("use_early_exit", True)
        self.wrong_direction_threshold = self.config.get("wrong_direction_threshold", 0.5)  # 50% of SL distance
        self.early_exit_at_breakeven = self.config.get("early_exit_at_breakeven", True)  # Exit at BE after wrong move

        # Profit locking levels (R:R ratios)
        self.profit_lock_levels = self.config.get("profit_lock_levels", {
            "level_1": 1.0,   # Lock 20% of profit at 1:1
            "level_2": 1.5,   # Lock 50% of profit at 1.5:1 (also partial TP)
            "level_3": 2.0    # Lock 70% of profit at 2:1 (full TP area)
        })

        # Track which profit levels have been activated per position
        self.profit_levels_activated = {}  # {ticket: {'level_1': True, 'level_2': False, ...}}

        # Track positions flagged for early exit (wrong direction detected)
        self.early_exit_flagged = {}  # {ticket: {'flagged': True, 'max_adverse': value}}
        
        # State tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_signal_time = None
        self.last_reset_date = datetime.now().date()
        self.starting_balance = 0.0
        self.partial_closed_tickets = set()  # Track partially closed positions
        
        # Signal generator
        self.signal_gen = BreakoutSignalGenerator(
            symbol=self.symbol,
            min_score_threshold=self.config.get("min_score", 45.0),
            strong_score_threshold=self.config.get("strong_score", 60.0)
        )
        
        # MT5 info
        self.symbol_info = None
        self.account_info = None
        
        self.setup_logging()

    def load_config(self, path: str) -> Dict:
        """Load or create configuration file"""
        defaults = {
            "symbol": "XAUUSDm",
            "magic_number": 888888,
            "max_spread_points": 40,
            "max_positions": 2,
            "risk_percent": 1.0,
            "min_lot": 0.01,
            "max_lot": 5.0,
            "max_daily_loss_percent": 3.0,
            "max_consecutive_losses": 3,
            "max_daily_trades": 8,
            "signal_cooldown_minutes": 20,
            # Enhanced profit protection settings
            "use_breakeven": True,
            "breakeven_trigger_rr": 1.0,
            "use_partial_tp": True,  # Enable partial TP by default
            "use_trailing_stop": True,  # Enable trailing stop
            "trailing_atr_multiplier": 0.6,  # Fallback trail distance
            # Momentum-adaptive trailing (adapts to market conditions)
            "use_momentum_adaptive": True,
            "trailing_by_momentum": {
                "strong": 0.9,    # Wide trail for strong momentum
                "medium": 0.6,    # Normal trail for medium momentum
                "weak": 0.35      # Tight trail for weak momentum
            },
            # Early exit detection (escape false breakouts)
            "use_early_exit": True,
            "wrong_direction_threshold": 0.5,  # Flag if goes 50% of SL in wrong direction
            "early_exit_at_breakeven": True,   # Exit at BE if comes back after wrong move
            "profit_lock_levels": {
                "level_1": 1.0,   # Lock 20% profit at 1:1 R:R
                "level_2": 1.5,   # Lock 50% profit at 1.5:1 R:R
                "level_3": 2.0    # Lock 70% profit at 2:1 R:R
            },
            "min_score": 45.0,
            "strong_score": 60.0,
            "check_interval_seconds": 15,
            "trading_start_hour": 1,
            "trading_end_hour": 23
        }
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                defaults.update(loaded)
            except Exception as e:
                print(f"Config load error: {e}")
        else:
            with open(path, 'w') as f:
                json.dump(defaults, f, indent=2)
            print(f"Created config: {path}")
        
        return defaults

    def setup_logging(self):
        """Setup logging to file and console"""
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/bot_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("XAUBot")

    def initialize_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        if not mt5.initialize():
            self.logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False
        
        # Get symbol info
        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            self.logger.error(f"Symbol {self.symbol} not found")
            return False
        
        if not self.symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                self.logger.error(f"Cannot select {self.symbol}")
                return False
        
        # Get account info
        self.account_info = mt5.account_info()
        if self.account_info:
            self.starting_balance = self.account_info.balance
            self.logger.info(f"Account: {self.account_info.login}")
            self.logger.info(f"Balance: ${self.account_info.balance:.2f}")
            self.logger.info(f"Leverage: 1:{self.account_info.leverage}")
        
        # Log symbol specs
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"  Point: {self.symbol_info.point}")
        self.logger.info(f"  Digits: {self.symbol_info.digits}")
        self.logger.info(f"  Min Lot: {self.symbol_info.volume_min}")
        self.logger.info(f"  Tick Value: ${self.symbol_info.trade_tick_value}")
        self.logger.info(f"  Tick Size: {self.symbol_info.trade_tick_size}")
        
        return True

    def calculate_lot_size(self, sl_distance: float) -> float:
        """
        Calculate proper lot size for XAU/USD based on risk
        
        For Gold (XAU/USD):
        - Tick Value = Value of 1 tick (minimum price movement) per 1 lot
        - Tick Size = Minimum price movement (usually 0.01)
        - Value per price unit = Tick Value / Tick Size
        - Risk Amount = Balance * Risk%
        - Lot Size = Risk Amount / (SL Distance * Value per unit)
        """
        account = mt5.account_info()
        if account is None:
            return self.min_lot
        
        balance = account.balance
        risk_amount = balance * (self.risk_percent / 100.0)
        
        # Get actual tick value and size from broker
        tick_value = self.symbol_info.trade_tick_value  # $ per tick per lot
        tick_size = self.symbol_info.trade_tick_size    # Price movement per tick
        
        if tick_size <= 0 or tick_value <= 0:
            self.logger.warning("Invalid tick info, using minimum lot")
            return self.min_lot
        
        # Value per 1.0 price movement per 1 lot
        value_per_point = tick_value / tick_size
        
        # Total monetary risk for SL distance per 1 lot
        sl_monetary_value = sl_distance * value_per_point
        
        if sl_monetary_value <= 0:
            return self.min_lot
        
        # Calculate lot size
        lot_size = risk_amount / sl_monetary_value
        
        # Round to volume step
        step = self.symbol_info.volume_step
        lot_size = round(lot_size / step) * step
        
        # Clamp to limits
        lot_size = max(self.min_lot, min(self.max_lot, lot_size))
        lot_size = max(self.symbol_info.volume_min, min(self.symbol_info.volume_max, lot_size))
        
        self.logger.info(f"Lot calculation: Risk=${risk_amount:.2f}, SL={sl_distance:.2f}, Lots={lot_size:.2f}")
        
        return round(lot_size, 2)

    def check_spread(self) -> bool:
        """Check if current spread is acceptable"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        
        spread = (tick.ask - tick.bid) / self.symbol_info.point
        
        if spread > self.max_spread_points:
            self.logger.warning(f"Spread too wide: {spread:.1f} points (max: {self.max_spread_points})")
            return False
        return True

    def is_trading_time(self) -> bool:
        """Check if market is open and within trading hours"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        
        server_time = datetime.fromtimestamp(tick.time)
        hour = server_time.hour
        
        start_hour = self.config.get("trading_start_hour", 1)
        end_hour = self.config.get("trading_end_hour", 23)
        
        return start_hour <= hour < end_hour

    def check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        # Reset daily stats if new day
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.last_reset_date = today
            self.starting_balance = mt5.account_info().balance if mt5.account_info() else self.starting_balance
            self.logger.info("=== NEW DAY - Stats Reset ===")
        
        # Check max daily trades
        if self.daily_trades >= self.max_daily_trades:
            self.logger.info(f"Max daily trades reached: {self.daily_trades}")
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses: {self.consecutive_losses}. Pausing trades.")
            return False
        
        # Check daily loss limit
        current_balance = mt5.account_info().balance if mt5.account_info() else self.starting_balance
        daily_loss_pct = ((self.starting_balance - current_balance) / self.starting_balance) * 100
        
        if daily_loss_pct >= self.max_daily_loss_percent:
            self.logger.warning(f"Daily loss limit hit: {daily_loss_pct:.2f}%")
            return False
        
        return True

    def check_cooldown(self) -> bool:
        """Check if enough time passed since last signal"""
        if self.last_signal_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_signal_time).total_seconds() / 60
        return elapsed >= self.signal_cooldown_minutes

    def get_open_positions(self) -> List:
        """Get positions opened by this bot"""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == self.magic_number]

    def open_trade(self, signal: Dict) -> bool:
        """Execute trade based on signal"""
        direction = signal['direction']
        sl = signal['sl']
        tp = signal['tp']
        sl_distance = signal['sl_distance']
        comment = signal['comment']
        
        # Get fresh price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.logger.error("Cannot get current price")
            return False
        
        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Calculate lot size
        lot_size = self.calculate_lot_size(sl_distance)
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 30,  # Allow some slippage
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            self.logger.error(f"Order send failed: {mt5.last_error()}")
            return False
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order rejected: {result.retcode} - {result.comment}")
            return False
        
        # Success
        self.logger.info("=" * 40)
        self.logger.info(f" TRADE OPENED: {direction}")
        self.logger.info(f"   Ticket: {result.order}")
        self.logger.info(f"   Price: {result.price}")
        self.logger.info(f"   Volume: {lot_size} lots")
        self.logger.info(f"   SL: {sl} | TP: {tp}")
        self.logger.info(f"   R:R = 1:{signal['risk_reward']}")
        self.logger.info(f"   Score: {signal['score']} ({signal['quality']})")
        self.logger.info("=" * 40)
        
        # Update state
        self.daily_trades += 1
        self.last_signal_time = datetime.now()
        
        # Save to history
        self.save_trade(signal, result)
        
        return True

    def save_trade(self, signal: Dict, result):
        """Save trade details to JSON file"""
        history_file = "trade_history.json"
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "ticket": result.order,
            "direction": signal['direction'],
            "volume": result.volume,
            "entry": result.price,
            "sl": signal['sl'],
            "tp": signal['tp'],
            "score": signal['score'],
            "quality": signal['quality'],
            "session": signal['session'],
            "risk_reward": signal['risk_reward'],
            "components": signal['components']
        }
        
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                pass
        
        history.append(record)
        
        # Keep last 500 trades
        if len(history) > 500:
            history = history[-500:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def calculate_momentum_score(self, position) -> float:
        """
        Calculate momentum score (0-100) for adaptive trailing

        Combines 3 indicators:
        1. Candle body strength (vs ATR)
        2. ATR expansion (volatility building)
        3. Price velocity (speed from entry)

        Returns:
            float: Momentum score 0-100
        """
        try:
            # Get market data
            df = self.signal_gen.get_market_data(bars=30)
            if df is None or len(df) < 20:
                return 50.0  # Default medium momentum

            # Get ATR
            atr_values = self.signal_gen.calculate_atr(df)
            current_atr = float(atr_values.iloc[-1])
            avg_atr = float(atr_values.iloc[-20:].mean())

            if current_atr <= 0 or avg_atr <= 0:
                return 50.0

            # --- INDICATOR 1: Candle Body Strength (0-100) ---
            # Look at last 3 candles
            recent_candles = df.iloc[-3:]
            is_buy = position.type == mt5.ORDER_TYPE_BUY

            body_scores = []
            for idx, candle in recent_candles.iterrows():
                body = candle['close'] - candle['open']
                body_abs = abs(body)
                body_to_atr = body_abs / current_atr

                # Strong body = 1.0+ ATR, Weak = <0.3 ATR
                if body_to_atr > 1.2:
                    score = 100
                elif body_to_atr > 0.8:
                    score = 80
                elif body_to_atr > 0.5:
                    score = 60
                elif body_to_atr > 0.3:
                    score = 40
                else:
                    score = 20

                # Check if body direction matches position
                if is_buy and body < 0:  # Bearish candle on buy position
                    score *= 0.5
                elif not is_buy and body > 0:  # Bullish candle on sell position
                    score *= 0.5

                body_scores.append(score)

            body_strength = sum(body_scores) / len(body_scores)

            # --- INDICATOR 2: ATR Expansion (0-100) ---
            # Is volatility increasing (momentum building)?
            atr_ratio = current_atr / avg_atr

            if atr_ratio > 1.3:
                atr_expansion = 100  # Strong expansion
            elif atr_ratio > 1.15:
                atr_expansion = 80
            elif atr_ratio > 1.05:
                atr_expansion = 60
            elif atr_ratio > 0.95:
                atr_expansion = 50
            elif atr_ratio > 0.85:
                atr_expansion = 30
            else:
                atr_expansion = 10  # ATR contracting

            # --- INDICATOR 3: Price Velocity (0-100) ---
            # How fast is price moving from entry?
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return 50.0

            open_price = position.price_open
            current_sl = position.sl

            if is_buy:
                original_sl_distance = open_price - current_sl
                current_price = tick.bid
                distance_from_entry = current_price - open_price
            else:
                original_sl_distance = current_sl - open_price
                current_price = tick.ask
                distance_from_entry = open_price - current_price

            if original_sl_distance <= 0:
                return 50.0

            # Velocity = how far we've moved relative to risk
            velocity_ratio = distance_from_entry / original_sl_distance

            if velocity_ratio > 2.0:
                velocity = 100  # Very fast move
            elif velocity_ratio > 1.5:
                velocity = 85
            elif velocity_ratio > 1.0:
                velocity = 70
            elif velocity_ratio > 0.5:
                velocity = 55
            elif velocity_ratio > 0.0:
                velocity = 40
            else:
                velocity = 20  # Moving against us

            # --- COMBINE SCORES ---
            # Weight: Body 40%, ATR 30%, Velocity 30%
            momentum_score = (body_strength * 0.4) + (atr_expansion * 0.3) + (velocity * 0.3)

            # Clamp to 0-100
            momentum_score = max(0, min(100, momentum_score))

            return momentum_score

        except Exception as e:
            self.logger.debug(f"Momentum calculation error: {e}")
            return 50.0  # Default to medium momentum on error

    def get_adaptive_trailing_multiplier(self, momentum_score: float) -> float:
        """
        Get trailing multiplier based on momentum score

        Args:
            momentum_score: 0-100 momentum score

        Returns:
            float: ATR multiplier for trailing distance
        """
        if not self.use_momentum_adaptive:
            return self.trailing_atr_multiplier  # Use fixed multiplier

        # Map momentum to trailing distance
        if momentum_score >= 70:
            return self.trailing_by_momentum["strong"]  # Wide trail
        elif momentum_score >= 40:
            return self.trailing_by_momentum["medium"]  # Normal trail
        else:
            return self.trailing_by_momentum["weak"]  # Tight trail

    def check_early_exit(self, position) -> bool:
        """
        BULLETPROOF: Detect and exit false breakouts early

        Detects when price goes WRONG direction first, then comes back.
        This is XAU/USD's typical trap: "touch SL then reverse"

        Returns:
            bool: True if should exit position immediately
        """
        if not self.use_early_exit:
            return False

        ticket = position.ticket

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False

        open_price = position.price_open
        current_sl = position.sl
        is_buy = position.type == mt5.ORDER_TYPE_BUY

        # Calculate original SL distance
        if is_buy:
            original_sl_distance = open_price - current_sl
            current_price = tick.bid
            adverse_move = open_price - current_price  # How far it went down
            current_profit = current_price - open_price
        else:
            original_sl_distance = current_sl - open_price
            current_price = tick.ask
            adverse_move = current_price - open_price  # How far it went up
            current_profit = open_price - current_price

        if original_sl_distance <= 0:
            return False

        # Initialize tracking for this position
        if ticket not in self.early_exit_flagged:
            self.early_exit_flagged[ticket] = {
                'flagged': False,
                'max_adverse': 0.0
            }

        # Track maximum adverse move
        if adverse_move > self.early_exit_flagged[ticket]['max_adverse']:
            self.early_exit_flagged[ticket]['max_adverse'] = adverse_move

        max_adverse = self.early_exit_flagged[ticket]['max_adverse']

        # --- DETECTION PHASE: Did price go wrong direction first? ---
        threshold_distance = original_sl_distance * self.wrong_direction_threshold

        if max_adverse >= threshold_distance and not self.early_exit_flagged[ticket]['flagged']:
            # Price went WRONG direction by 50%+ of SL distance
            self.early_exit_flagged[ticket]['flagged'] = True
            self.logger.warning(f"⚠️ Wrong Direction Detected | Ticket: {ticket} | Adverse: {max_adverse:.2f}")

        # --- EXIT PHASE: If flagged and price comes back, EXIT ---
        if self.early_exit_flagged[ticket]['flagged']:
            # Check if we should exit

            if self.early_exit_at_breakeven:
                # Exit if at breakeven or any profit
                if current_profit >= 0:
                    self.logger.info(f"🛡️ Early Exit: Escaped Trap | Ticket: {ticket} | P/L: {current_profit:.2f}")
                    return True
            else:
                # Exit if at reduced loss (50% of original risk)
                max_acceptable_loss = original_sl_distance * 0.5
                if adverse_move <= max_acceptable_loss:
                    self.logger.info(f"🛡️ Early Exit: Reduced Loss | Ticket: {ticket} | Loss: {adverse_move:.2f}")
                    return True

        return False

    def close_position_immediately(self, position) -> bool:
        """
        Close entire position immediately (early exit)

        Args:
            position: MT5 position object

        Returns:
            bool: True if closed successfully
        """
        # Determine close price and order type
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = mt5.symbol_info_tick(self.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(self.symbol).ask

        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": 30,
            "magic": self.magic_number,
            "comment": "Early Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Execute close
        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        else:
            self.logger.error(f"Early exit close failed: {result.comment if result else 'Unknown error'}")
            return False

    def manage_profit_protection(self, position):
        """
        Enhanced profit protection system for XAU/USD:
        - BULLETPROOF early exit detection (escape false breakouts)
        - Momentum-adaptive trailing (wide in trends, tight in chop)
        - Multi-level profit locking at 1:1, 1.5:1, 2:1 R:R
        - Partial profit taking at 1.5:1
        - Smart profit preservation
        """
        ticket = position.ticket

        # --- PHASE 0: EARLY EXIT CHECK (Bulletproof Protection) ---
        # Check if this is a false breakout FIRST, before any other logic
        if self.check_early_exit(position):
            # Price went wrong direction, came back - EXIT NOW
            if self.close_position_immediately(position):
                # Clean up tracking
                if ticket in self.profit_levels_activated:
                    del self.profit_levels_activated[ticket]
                if ticket in self.early_exit_flagged:
                    del self.early_exit_flagged[ticket]
                if ticket in self.partial_closed_tickets:
                    self.partial_closed_tickets.remove(ticket)
            return  # Exit function, position closed

        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        # Initialize tracking for this position if needed
        if ticket not in self.profit_levels_activated:
            self.profit_levels_activated[ticket] = {
                'level_1': False,
                'level_2': False,
                'level_3': False,
                'trailing_active': False
            }

        open_price = position.price_open
        current_sl = position.sl
        tp = position.tp
        is_buy = position.type == mt5.ORDER_TYPE_BUY

        # Calculate original SL distance (risk)
        if is_buy:
            original_sl_distance = open_price - current_sl
            current_price = tick.bid
            current_profit_distance = current_price - open_price
        else:
            original_sl_distance = current_sl - open_price
            current_price = tick.ask
            current_profit_distance = open_price - current_price

        # Safety check
        if original_sl_distance <= 0:
            return

        # Calculate current R:R ratio
        current_rr = current_profit_distance / original_sl_distance

        # Get current ATR for trailing calculations
        atr = self.get_current_atr()
        if atr is None:
            atr = original_sl_distance  # Fallback to original SL distance

        # --- CALCULATE MOMENTUM SCORE (for adaptive trailing) ---
        momentum_score = self.calculate_momentum_score(position)
        trailing_multiplier = self.get_adaptive_trailing_multiplier(momentum_score)

        # ----- MULTI-LEVEL PROFIT LOCKING -----

        # Level 1: Lock 20% of profit at 1:1 R:R (Smart Breakeven)
        if (current_rr >= self.profit_lock_levels['level_1'] and
            not self.profit_levels_activated[ticket]['level_1']):

            # Lock 20% of original risk as profit
            profit_lock = original_sl_distance * 0.20

            if is_buy:
                new_sl = open_price + profit_lock
                if new_sl > current_sl:  # Only move forward
                    self.modify_position_sl(ticket, new_sl)
                    self.logger.info(f"✓ Level 1 (1:1): Locked 20% profit | Ticket: {ticket}")
                    self.profit_levels_activated[ticket]['level_1'] = True
            else:
                new_sl = open_price - profit_lock
                if new_sl < current_sl:  # Only move forward
                    self.modify_position_sl(ticket, new_sl)
                    self.logger.info(f"✓ Level 1 (1:1): Locked 20% profit | Ticket: {ticket}")
                    self.profit_levels_activated[ticket]['level_1'] = True

        # Level 2: Lock 50% of profit at 1.5:1 R:R + Partial TP
        elif (current_rr >= self.profit_lock_levels['level_2'] and
              not self.profit_levels_activated[ticket]['level_2']):

            # Lock 50% of original risk as profit
            profit_lock = original_sl_distance * 0.50

            if is_buy:
                new_sl = open_price + profit_lock
                if new_sl > current_sl:
                    self.modify_position_sl(ticket, new_sl)
                    self.logger.info(f"✓✓ Level 2 (1.5:1): Locked 50% profit | Ticket: {ticket}")
                    self.profit_levels_activated[ticket]['level_2'] = True

                    # Partial profit taking - close 50% of position
                    if self.use_partial_tp and ticket not in self.partial_closed_tickets:
                        self.close_partial_position(position, 0.5)
            else:
                new_sl = open_price - profit_lock
                if new_sl < current_sl:
                    self.modify_position_sl(ticket, new_sl)
                    self.logger.info(f"✓✓ Level 2 (1.5:1): Locked 50% profit | Ticket: {ticket}")
                    self.profit_levels_activated[ticket]['level_2'] = True

                    if self.use_partial_tp and ticket not in self.partial_closed_tickets:
                        self.close_partial_position(position, 0.5)

        # Level 3: Lock 70% of profit at 2:1 R:R (near full TP)
        elif (current_rr >= self.profit_lock_levels['level_3'] and
              not self.profit_levels_activated[ticket]['level_3']):

            # Lock 70% of original risk as profit
            profit_lock = original_sl_distance * 0.70

            if is_buy:
                new_sl = open_price + profit_lock
                if new_sl > current_sl:
                    self.modify_position_sl(ticket, new_sl)
                    self.logger.info(f"✓✓✓ Level 3 (2:1): Locked 70% profit | Ticket: {ticket}")
                    self.profit_levels_activated[ticket]['level_3'] = True
            else:
                new_sl = open_price - profit_lock
                if new_sl < current_sl:
                    self.modify_position_sl(ticket, new_sl)
                    self.logger.info(f"✓✓✓ Level 3 (2:1): Locked 70% profit | Ticket: {ticket}")
                    self.profit_levels_activated[ticket]['level_3'] = True

        # ----- MOMENTUM-ADAPTIVE TRAILING STOP -----

        # Activate trailing after Level 1 is hit
        if (self.use_trailing_stop and
            self.profit_levels_activated[ticket]['level_1'] and
            current_rr >= self.profit_lock_levels['level_1']):

            # Trailing distance based on ATR and MOMENTUM
            trailing_distance = atr * trailing_multiplier

            # Determine momentum category for logging
            if momentum_score >= 70:
                momentum_level = "STRONG"
            elif momentum_score >= 40:
                momentum_level = "MEDIUM"
            else:
                momentum_level = "WEAK"

            # Calculate what the trailing SL should be
            if is_buy:
                trailing_sl = current_price - trailing_distance

                # Only move SL forward, never backward
                if trailing_sl > current_sl:
                    self.modify_position_sl(ticket, trailing_sl)

                    # Log only first activation or significant moves
                    if not self.profit_levels_activated[ticket]['trailing_active']:
                        self.logger.info(f"⚡ Adaptive Trailing Activated | Ticket: {ticket} | Momentum: {momentum_level} ({momentum_score:.0f}) | Distance: {trailing_distance:.2f}")
                        self.profit_levels_activated[ticket]['trailing_active'] = True
            else:
                trailing_sl = current_price + trailing_distance

                if trailing_sl < current_sl:
                    self.modify_position_sl(ticket, trailing_sl)

                    if not self.profit_levels_activated[ticket]['trailing_active']:
                        self.logger.info(f"⚡ Adaptive Trailing Activated | Ticket: {ticket} | Momentum: {momentum_level} ({momentum_score:.0f}) | Distance: {trailing_distance:.2f}")
                        self.profit_levels_activated[ticket]['trailing_active'] = True

    def get_current_atr(self) -> Optional[float]:
        """Get current ATR value for trailing stop calculations"""
        try:
            df = self.signal_gen.get_market_data(bars=50)
            if df is None or len(df) < 14:
                return None

            atr_values = self.signal_gen.calculate_atr(df)
            current_atr = float(atr_values.iloc[-1])
            return current_atr
        except Exception as e:
            self.logger.debug(f"ATR calculation error: {e}")
            return None

    def close_partial_position(self, position, fraction: float = 0.5):
        """
        Close a fraction of the position for partial profit taking

        Args:
            position: MT5 position object
            fraction: Fraction to close (0.5 = 50%)
        """
        if position.ticket in self.partial_closed_tickets:
            return  # Already partially closed

        close_volume = round(position.volume * fraction, 2)

        # Make sure we have valid volume
        if close_volume < self.symbol_info.volume_min:
            self.logger.debug(f"Partial close volume too small: {close_volume}")
            return

        # Determine close price and order type
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = mt5.symbol_info_tick(self.symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(self.symbol).ask

        # Prepare partial close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": 30,
            "magic": self.magic_number,
            "comment": "Partial TP 50%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Execute partial close
        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.partial_closed_tickets.add(position.ticket)
            self.logger.info(f"💰 Partial TP: Closed {fraction*100:.0f}% at 1.5:1 | Ticket: {position.ticket}")
        else:
            self.logger.warning(f"Partial close failed: {result.comment if result else 'Unknown error'}")

    def modify_position_sl(self, ticket: int, new_sl: float):
        """Modify stop loss of a position"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return
        
        pos = positions[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": round(new_sl, self.symbol_info.digits),
            "tp": pos.tp,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.debug(f"SL modified: {ticket} -> {new_sl}")

    def check_closed_trades(self):
        """Check for recently closed trades to update stats"""
        # Get deals from today
        today_start = datetime.combine(datetime.now().date(), datetime.min.time())
        
        deals = mt5.history_deals_get(today_start, datetime.now())
        if deals is None:
            return
        
        # Filter for our bot's closing deals
        for deal in deals:
            if deal.magic == self.magic_number and deal.entry == mt5.DEAL_ENTRY_OUT:
                # This is a position close
                profit = deal.profit + deal.swap + deal.commission
                
                # Update consecutive loss counter
                if profit < 0:
                    self.consecutive_losses += 1
                    self.logger.info(f"Trade closed with loss: ${profit:.2f}. Consecutive losses: {self.consecutive_losses}")
                else:
                    self.consecutive_losses = 0  # Reset on win
                    self.logger.info(f"Trade closed with profit: ${profit:.2f}")

    def manage_positions(self):
        """Manage all open positions with enhanced profit protection"""
        positions = self.get_open_positions()

        for pos in positions:
            self.manage_profit_protection(pos)

        # Clean up tracking for closed positions
        if positions:
            open_tickets = {p.ticket for p in positions}

            # Clean up profit levels tracking
            closed_tickets = set(self.profit_levels_activated.keys()) - open_tickets
            for ticket in closed_tickets:
                del self.profit_levels_activated[ticket]
                if ticket in self.partial_closed_tickets:
                    self.partial_closed_tickets.remove(ticket)

            # Clean up early exit tracking
            closed_flagged = set(self.early_exit_flagged.keys()) - open_tickets
            for ticket in closed_flagged:
                del self.early_exit_flagged[ticket]

    def run_cycle(self):
        """Single iteration of the bot logic"""
        # Check closed trades first
        self.check_closed_trades()
        
        # Manage existing positions
        self.manage_positions()
        
        # Check if we can open new trades
        if not self.is_trading_time():
            return
        
        if not self.check_risk_limits():
            return
        
        if not self.check_spread():
            return
        
        if not self.check_cooldown():
            return
        
        # Check max positions
        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_positions:
            return
        
        # Generate signal
        signal = self.signal_gen.generate_signal()
        print(signal)
        if signal is None:
            return
        
        self.logger.info(f" SIGNAL: {signal['direction']} | Score: {signal['score']:.1f} | Session: {signal['session']}")
        
        # Execute trade
        self.open_trade(signal)

    def run(self):
        """Main bot loop"""
        self.logger.info("=" * 50)
        self.logger.info("  XAU/USD BREAKOUT BOT - STARTING")
        self.logger.info("=" * 50)
        
        if not self.initialize_mt5():
            self.logger.error("Failed to initialize. Exiting.")
            return
        
        self.logger.info(f"Risk: {self.risk_percent}% per trade")
        self.logger.info(f"Max positions: {self.max_positions}")
        self.logger.info(f"Max daily trades: {self.max_daily_trades}")
        self.logger.info(f"Max daily loss: {self.max_daily_loss_percent}%")
        self.logger.info(f"Min signal score: {self.config.get('min_score', 45)}")
        self.logger.info("=" * 50)
        self.logger.info("Bot running... Press Ctrl+C to stop")
        self.logger.info("")
        
        check_interval = self.config.get("check_interval_seconds", 15)
        reconnect_wait = 10
        max_reconnects = 10
        reconnect_count = 0
        
        try:
            while True:
                try:
                    # Check MT5 connection
                    if not mt5.terminal_info():
                        self.logger.warning("MT5 connection lost...")
                        mt5.shutdown()
                        time.sleep(reconnect_wait)
                        
                        if self.initialize_mt5():
                            self.logger.info("Reconnected successfully")
                            reconnect_count = 0
                        else:
                            reconnect_count += 1
                            if reconnect_count >= max_reconnects:
                                self.logger.error("Too many reconnect failures")
                                break
                            continue
                    
                    # Run main logic
                    self.run_cycle()
                    
                    # Wait
                    time.sleep(check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Cycle error: {e}", exc_info=True)
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            self.logger.info("\nBot stopped by user")
        finally:
            self.logger.info("Shutting down MT5...")
            mt5.shutdown()
            self.logger.info("Goodbye!")


if __name__ == "__main__":
    bot = XAUTradingBot(config_path="bot_config.json")
    bot.run()
