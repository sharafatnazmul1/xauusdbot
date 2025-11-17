"""
XAU/USD Dynamic Breakout Signal Generator - PRODUCTION VERSION
Fixed: Proper entry prices, realistic R:R, correct scoring logic
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import MetaTrader5 as mt5


class BreakoutSignalGenerator:
    def __init__(self,
                 symbol: str = "XAUUSDm",
                 timeframe: int = mt5.TIMEFRAME_M5,
                 atr_period: int = 14,
                 vol_ma_period: int = 20,
                 level_lookback: int = 20,
                 min_score_threshold: float = 45.0,
                 strong_score_threshold: float = 60.0):
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.atr_period = atr_period
        self.vol_ma_period = vol_ma_period
        self.level_lookback = level_lookback
        self.min_score_threshold = min_score_threshold
        self.strong_score_threshold = strong_score_threshold
        
        # Scoring weights - simplified and balanced
        self.weights = {
            'level_break': 0.25,      # Most important - actual breakout
            'body_strength': 0.25,    # Strong momentum candle
            'volume_spike': 0.20,     # Volume confirmation
            'trend_align': 0.15,      # With the trend
            'compression': 0.15       # Breakout from compression
        }

    def get_market_data(self, bars: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from MT5"""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df

    def get_current_price(self) -> Optional[Dict]:
        """Get current bid/ask prices"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        return {'bid': tick.bid, 'ask': tick.ask, 'time': tick.time}

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(self.atr_period, min_periods=1).mean()

    def get_current_session(self) -> str:
        """Determine trading session from MT5 server time"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return 'closed'
        
        # Exness MT5 uses UTC+0 or UTC+2/3 depending on DST
        server_time = datetime.fromtimestamp(tick.time)
        hour = server_time.hour
        
        # Conservative session mapping (Exness server time)
        if 1 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 12:
            return 'london_open'
        elif 12 <= hour < 15:
            return 'london_ny_overlap'  # Best session for gold
        elif 15 <= hour < 21:
            return 'ny'
        else:
            return 'off_hours'

    def compute_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all breakout quality features"""
        df = df.copy()
        
        # ATR for volatility context
        df['atr'] = self.calculate_atr(df)
        
        # Volume analysis
        df['avg_vol'] = df['volume'].rolling(self.vol_ma_period, min_periods=1).mean()
        df['vol_ratio'] = df['volume'] / (df['avg_vol'] + 1)
        
        # Define support/resistance levels
        df['resistance'] = df['high'].rolling(self.level_lookback, min_periods=5).max().shift(1)
        df['support'] = df['low'].rolling(self.level_lookback, min_periods=5).min().shift(1)
        
        # Breakout detection - close THROUGH level with some margin
        atr_margin = df['atr'] * 0.3  # Need to break by at least 30% of ATR
        df['break_up'] = ((df['close'] > df['resistance']) & 
                          (df['close'] - df['resistance'] > atr_margin)).astype(int)
        df['break_down'] = ((df['close'] < df['support']) & 
                            (df['support'] - df['close'] > atr_margin)).astype(int)
        
        # Candle body analysis
        df['body'] = df['close'] - df['open']
        df['body_abs'] = df['body'].abs()
        df['candle_range'] = df['high'] - df['low']
        
        # Body ratio - how much of candle is body (not wicks)
        df['body_ratio'] = df['body_abs'] / (df['candle_range'] + 0.001)
        
        # Trend direction (simple: are we making higher highs/lower lows)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['trend_up'] = df['higher_high'].rolling(5).sum() > 3
        df['trend_down'] = df['lower_low'].rolling(5).sum() > 3
        
        # Compression detection - narrowing range
        df['range_ma'] = df['candle_range'].rolling(10).mean()
        df['compression'] = df['range_ma'] < df['range_ma'].rolling(20).mean() * 0.8
        
        return df

    def score_breakout(self, row: pd.Series, direction: str) -> float:
        """Calculate quality score for a breakout bar"""
        scores = {}
        
        # 1. Level break strength (0-1)
        if direction == 'BUY':
            break_distance = row['close'] - row['resistance']
        else:
            break_distance = row['support'] - row['close']
        
        # Normalize by ATR - deeper break = higher score
        break_score = min(1.0, break_distance / (row['atr'] * 1.5))
        scores['level_break'] = max(0, break_score)
        
        # 2. Body strength (0-1)
        # Strong body in direction of breakout
        body_atr_ratio = row['body_abs'] / (row['atr'] + 0.001)
        body_score = min(1.0, body_atr_ratio / 1.2)  # Good if body > 1.2x ATR
        
        # Penalize if body direction doesn't match breakout
        if direction == 'BUY' and row['body'] < 0:
            body_score *= 0.3  # Bearish candle on bullish breakout = weak
        elif direction == 'SELL' and row['body'] > 0:
            body_score *= 0.3
        
        # Bonus for clean candle (high body ratio = small wicks)
        if row['body_ratio'] > 0.7:
            body_score = min(1.0, body_score * 1.2)
        
        scores['body_strength'] = body_score
        
        # 3. Volume confirmation (0-1)
        if row['vol_ratio'] > 2.0:
            vol_score = 1.0
        elif row['vol_ratio'] > 1.5:
            vol_score = 0.8
        elif row['vol_ratio'] > 1.2:
            vol_score = 0.6
        elif row['vol_ratio'] > 1.0:
            vol_score = 0.4
        else:
            vol_score = 0.2  # Below average volume = weak
        scores['volume_spike'] = vol_score
        
        # 4. Trend alignment (0-1)
        if direction == 'BUY' and row['trend_up']:
            trend_score = 1.0
        elif direction == 'SELL' and row['trend_down']:
            trend_score = 1.0
        else:
            trend_score = 0.4  # Counter-trend breakout = lower score
        scores['trend_align'] = trend_score
        
        # 5. Compression breakout (0-1)
        if row['compression']:
            comp_score = 1.0  # Breakout from tight range = high quality
        else:
            comp_score = 0.5
        scores['compression'] = comp_score
        
        # Weighted total
        total_score = sum(scores[k] * self.weights[k] for k in scores)
        percentage = (total_score / sum(self.weights.values())) * 100
        
        return percentage, scores

    def generate_signal(self) -> Optional[Dict]:
        """Generate trading signal if conditions are met"""
        
        # Get market data
        df = self.get_market_data(bars=200)
        if df is None or len(df) < 50:
            return None
        
        # Get current live price
        price_info = self.get_current_price()
        if price_info is None:
            return None
        
        # Compute features
        df = self.compute_breakout_features(df)
        
        # Check LAST COMPLETED candle for breakout (index -2, as -1 is forming)
        signal_bar = df.iloc[-2]
        
        # Determine if there's a breakout
        if signal_bar['break_up'] == 1:
            direction = 'BUY'
            entry_price = price_info['ask']  # Buy at ask
        elif signal_bar['break_down'] == 1:
            direction = 'SELL'
            entry_price = price_info['bid']  # Sell at bid
        else:
            return None  # No breakout
        
        # Score the breakout quality
        score, score_components = self.score_breakout(signal_bar, direction)
        
        # Session quality adjustment
        session = self.get_current_session()
        session_multipliers = {
            'asian': 0.85,
            'london_open': 1.0,
            'london_ny_overlap': 1.1,  # Best time for gold
            'ny': 0.95,
            'off_hours': 0.7,
            'closed': 0.0
        }
        
        adjusted_score = score * session_multipliers.get(session, 1.0)
        
        # Filter by minimum score
        if adjusted_score < self.min_score_threshold:
            return None
        
        # Calculate SL and TP with REALISTIC ratios for M5
        atr = float(signal_bar['atr'])
        
        # Conservative R:R based on score quality
        if adjusted_score >= self.strong_score_threshold:
            quality = "STRONG"
            sl_distance = atr * 1.0   # 1 ATR stop loss
            tp_distance = atr * 2.0   # 2 ATR take profit (2:1 R:R)
        elif adjusted_score >= 50:
            quality = "STANDARD"
            sl_distance = atr * 1.2   # Wider SL for standard
            tp_distance = atr * 2.0   # Still aim for ~1.7:1 R:R
        else:
            quality = "WEAK"
            sl_distance = atr * 1.5   # Even wider SL
            tp_distance = atr * 2.0   # Conservative TP (~1.3:1 R:R)
        
        # Get symbol info for proper rounding
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return None
        
        digits = symbol_info.digits
        
        # Calculate exact SL/TP levels
        if direction == 'BUY':
            sl = round(entry_price - sl_distance, digits)
            tp = round(entry_price + tp_distance, digits)
        else:
            sl = round(entry_price + sl_distance, digits)
            tp = round(entry_price - tp_distance, digits)
        
        # Create signal dictionary
        comment = f"{quality[:3]}_{session[:3].upper()}_{int(adjusted_score)}"
        
        signal = {
            'direction': direction,
            'entry_price': round(entry_price, digits),
            'sl': sl,
            'tp': tp,
            'sl_distance': round(sl_distance, digits),
            'tp_distance': round(tp_distance, digits),
            'score': round(adjusted_score, 2),
            'raw_score': round(score, 2),
            'quality': quality,
            'session': session,
            'atr': round(atr, digits),
            'comment': comment,
            'timestamp': datetime.now().isoformat(),
            'risk_reward': round(tp_distance / sl_distance, 2),
            'components': {k: round(v * 100, 1) for k, v in score_components.items()}
        }
        
        return signal


if __name__ == "__main__":
    print("Testing Signal Generator...")
    
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
    else:
        gen = BreakoutSignalGenerator(symbol="XAUUSDm")
        
        # Show market info
        info = gen.get_current_price()
        if info:
            print(f"Current Price - Bid: {info['bid']}, Ask: {info['ask']}")
        
        # Check for signal
        signal = gen.generate_signal()
        if signal:
            print("\n=== SIGNAL FOUND ===")
            for k, v in signal.items():
                print(f"{k}: {v}")
        else:
            print("\nNo signal at this time (normal - signals are rare)")
        
        # Show session
        print(f"\nCurrent Session: {gen.get_current_session()}")
        
        mt5.shutdown()
