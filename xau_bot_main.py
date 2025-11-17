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
        
        # Trade management
        self.use_breakeven = self.config.get("use_breakeven", True)
        self.breakeven_trigger_rr = self.config.get("breakeven_trigger_rr", 1.0)  # Move to BE at 1:1
        self.use_partial_tp = self.config.get("use_partial_tp", True)  # Disabled by default - simpler
        
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
            "use_breakeven": True,
            "breakeven_trigger_rr": 1.0,
            "use_partial_tp": False,
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

    def manage_breakeven(self, position):
        """Move SL to breakeven when profit reaches trigger"""
        if not self.use_breakeven:
            return
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return
        
        open_price = position.price_open
        current_sl = position.sl
        tp = position.tp
        
        # Calculate original SL distance
        if position.type == mt5.ORDER_TYPE_BUY:
            original_sl_distance = open_price - current_sl
            current_price = tick.bid
            current_profit_distance = current_price - open_price
            
            # Check if we should move to breakeven
            # Trigger when profit >= SL distance * trigger_rr
            trigger_distance = original_sl_distance * self.breakeven_trigger_rr
            
            if current_profit_distance >= trigger_distance and current_sl < open_price:
                # Move SL to entry + small buffer
                buffer = self.symbol_info.point * 10  # 10 points buffer
                new_sl = open_price + buffer
                self.modify_position_sl(position.ticket, new_sl)
                self.logger.info(f" Breakeven set: Ticket {position.ticket}")
        else:
            original_sl_distance = current_sl - open_price
            current_price = tick.ask
            current_profit_distance = open_price - current_price
            
            trigger_distance = original_sl_distance * self.breakeven_trigger_rr
            
            if current_profit_distance >= trigger_distance and current_sl > open_price:
                buffer = self.symbol_info.point * 10
                new_sl = open_price - buffer
                self.modify_position_sl(position.ticket, new_sl)
                self.logger.info(f" Breakeven set: Ticket {position.ticket}")

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
        """Manage all open positions"""
        positions = self.get_open_positions()
        
        for pos in positions:
            self.manage_breakeven(pos)

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
