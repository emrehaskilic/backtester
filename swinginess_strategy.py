"""
Swinginess strateji mantığı — TRS + DFS + multi-layer exit.
Tick Replay Engine'den gelen verileri kullanır.
"""

import math
from collections import deque


class TrendReversalScore:
    """TRS — Trend dönüş tespiti (swinginess orijinal mantığı)."""

    def __init__(self, confirm_ticks=90, bullish_zone=0.65, bearish_zone=0.35,
                 agreement_threshold=0.40):
        self.confirm_ticks = confirm_ticks
        self.bullish_zone = bullish_zone
        self.bearish_zone = bearish_zone
        self.agreement_threshold = agreement_threshold

        self.current_dir = 0
        self.pending_dir = 0
        self.confirm_counter = 0
        self.confidence = 0
        self.reversed = False

    def update(self, dfs, cvd_slope_z, delta_z, obi_z):
        self.reversed = False

        bullish_count = 0
        bearish_count = 0

        if dfs >= self.bullish_zone: bullish_count += 1
        if dfs <= self.bearish_zone: bearish_count += 1
        if cvd_slope_z > 0.5: bullish_count += 1
        elif cvd_slope_z < -0.5: bearish_count += 1
        if delta_z > 0.5: bullish_count += 1
        elif delta_z < -0.5: bearish_count += 1
        if obi_z > 0.3: bullish_count += 1
        elif obi_z < -0.3: bearish_count += 1

        total = 4
        bull_agreement = bullish_count / total
        bear_agreement = bearish_count / total

        new_dir = 0
        if bull_agreement >= self.agreement_threshold:
            new_dir = 1
        elif bear_agreement >= self.agreement_threshold:
            new_dir = -1

        if new_dir != 0 and new_dir != self.current_dir:
            if new_dir == self.pending_dir:
                self.confirm_counter += 1
            else:
                self.pending_dir = new_dir
                self.confirm_counter = 1

            if self.confirm_counter >= self.confirm_ticks:
                old_dir = self.current_dir
                self.current_dir = new_dir
                self.pending_dir = 0
                self.confirm_counter = 0
                self.confidence = max(bull_agreement, bear_agreement)
                if old_dir != 0:
                    self.reversed = True
        elif new_dir == self.current_dir:
            self.pending_dir = 0
            self.confirm_counter = 0
        else:
            self.confirm_counter = max(0, self.confirm_counter - 1)

        return self.reversed, self.current_dir, self.confidence


class ExitScoreCalculator:
    """Çıkış skoru — DFS reversal tespiti."""

    def __init__(self, dfs_weight=0.40, cvd_weight=0.25, delta_weight=0.20, obi_weight=0.15):
        self.dfs_w = dfs_weight
        self.cvd_w = cvd_weight
        self.delta_w = delta_weight
        self.obi_w = obi_weight

    def calculate(self, position_side, dfs, cvd_z, delta_z, obi_z):
        """
        position_side: 1=long, -1=short
        Returns: exit_score 0-1
        """
        if position_side == 1:
            dfs_flip = max(0, 1 - dfs * 2) if dfs < 0.5 else 0
            cvd_flip = max(0, -cvd_z / 3)
            delta_flip = max(0, -delta_z / 3)
            obi_flip = max(0, -obi_z / 3)
        else:
            dfs_flip = max(0, dfs * 2 - 1) if dfs > 0.5 else 0
            cvd_flip = max(0, cvd_z / 3)
            delta_flip = max(0, delta_z / 3)
            obi_flip = max(0, obi_z / 3)

        score = (self.dfs_w * dfs_flip + self.cvd_w * cvd_flip +
                 self.delta_w * delta_flip + self.obi_w * obi_flip)
        return min(score, 1.0)


class SwingingessStrategy:
    """
    Swinginess strateji — tam mantık.
    Tick Engine'den gelen verilerle çalışır.
    """

    def __init__(self, params=None):
        if params is None:
            params = {}

        self.params = params

        # DFS ağırlıkları
        self.weights = {
            "zDelta": params.get("w_delta", 0.22),
            "zCvd": params.get("w_cvd", 0.18),
            "zLogP": params.get("w_logp", 0.12),
            "zObiW": params.get("w_obi_w", 0.14),
            "zObiD": params.get("w_obi_d", 0.12),
            "sweepSigned": params.get("w_sweep", 0.08),
            "burstSigned": params.get("w_burst", 0.08),
            "oiImpulse": params.get("w_oi", 0.06),
        }
        w_sum = sum(self.weights.values())
        self.weights = {k: v / w_sum for k, v in self.weights.items()}

        # TRS
        self.trs = TrendReversalScore(
            confirm_ticks=params.get("trs_confirm_ticks", 90),
            bullish_zone=params.get("trs_bullish_zone", 0.65),
            bearish_zone=params.get("trs_bearish_zone", 0.35),
            agreement_threshold=params.get("trs_agreement", 0.40),
        )

        # Exit score
        self.exit_calc = ExitScoreCalculator()

        # Risk parametreleri
        self.sl_pct = params.get("stop_loss_pct", 1.5) / 100
        self.trailing_activation = params.get("trailing_activation_pct", 0.8) / 100
        self.trailing_distance = params.get("trailing_distance_pct", 0.5) / 100
        self.exit_score_hard = params.get("exit_score_hard", 0.85)
        self.exit_score_soft = params.get("exit_score_soft", 0.70)
        self.time_flat_sec = params.get("time_flat_sec", 14400)
        self.min_prints_per_sec = params.get("min_prints_per_sec", 1.0)
        self.entry_cooldown_sec = params.get("entry_cooldown_sec", 300)
        self.min_hold_reversal_sec = params.get("min_hold_reversal_sec", 1800)
        self.reversal_confidence = params.get("reversal_exit_confidence", 0.75)

        # Trading
        self.leverage = params.get("leverage", 50)
        self.margin = params.get("margin_per_trade", 100)
        self.maker_fee = params.get("maker_fee", 0.0002)
        self.taker_fee = params.get("taker_fee", 0.0005)

        # State
        self.position = 0
        self.entry_price = 0
        self.entry_ts = 0
        self.position_qty = 0
        self.peak_pnl_pct = 0
        self.trailing_active = False
        self.last_entry_ts = 0
        self.equity = 1000.0
        self.trades = []
        self.peak_equity = 1000.0
        self.max_drawdown = 0.0

    def _close_position(self, price, ts_sec, reason):
        """Pozisyonu kapat ve trade kaydet."""
        hold_sec = ts_sec - self.entry_ts
        if self.position == 1:
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price

        pnl = self.position_qty * self.entry_price * pnl_pct * self.leverage
        fee = self.position_qty * price * self.maker_fee * self.leverage
        self.equity += pnl - fee
        self.trades.append({"pnl": pnl - fee, "type": reason, "hold": hold_sec, "pnl_pct": pnl_pct * 100})

        self.position = 0
        self.position_qty = 0
        self.peak_pnl_pct = 0
        self.trailing_active = False

    def on_second(self, ts_sec, price, engine):
        """
        Her saniyede çağrılır.
        engine: TickReplayEngine instance
        """
        # DFS hesapla
        dfs = engine.get_dfs(self.weights)
        comps = engine.dfs_components

        # TRS güncelle
        reversed_sig, trs_dir, trs_conf = self.trs.update(
            dfs, comps["zCvd"], comps["zDelta"], comps["zObiW"]
        )

        # === POZİSYON YÖNETİMİ ===
        if self.position != 0:
            hold_sec = ts_sec - self.entry_ts

            if self.position == 1:
                current_pnl_pct = (price - self.entry_price) / self.entry_price
            else:
                current_pnl_pct = (self.entry_price - price) / self.entry_price

            if current_pnl_pct > self.peak_pnl_pct:
                self.peak_pnl_pct = current_pnl_pct

            # 1. HARD STOP LOSS
            if current_pnl_pct <= -self.sl_pct:
                self._close_position(price, ts_sec, "SL")
                return

            # 2. TRAILING STOP
            if self.peak_pnl_pct >= self.trailing_activation:
                self.trailing_active = True

            if self.trailing_active:
                giveback = self.peak_pnl_pct - current_pnl_pct
                if giveback >= self.trailing_distance:
                    self._close_position(price, ts_sec, "TRAIL")
                    return

            # 3. EXIT SCORE
            exit_score = self.exit_calc.calculate(
                self.position, dfs, comps["zCvd"], comps["zDelta"], comps["zObiW"]
            )

            if hold_sec >= 120:
                if exit_score >= self.exit_score_hard:
                    self._close_position(price, ts_sec, "EXIT_SCORE")
                    return

                # Bleed stop: kaybediyorsan ve exit score yüksekse çık
                if current_pnl_pct < -0.003 and exit_score >= 0.75:
                    self._close_position(price, ts_sec, "BLEED_STOP")
                    return

            # 4. TIME FLAT
            if hold_sec >= self.time_flat_sec and self.peak_pnl_pct < 0.003:
                self._close_position(price, ts_sec, "TIME_FLAT")
                return

            # 5. TRS REVERSAL EXIT
            if reversed_sig and hold_sec >= self.min_hold_reversal_sec and trs_conf >= self.reversal_confidence:
                if (self.position == 1 and trs_dir == -1) or (self.position == -1 and trs_dir == 1):
                    self._close_position(price, ts_sec, "TRS_REV")
                    return

        # === GİRİŞ SİNYALİ ===
        if self.position == 0 and reversed_sig and trs_conf >= 0.50:
            if ts_sec - self.last_entry_ts < self.entry_cooldown_sec:
                return

            if engine.prints_per_sec < self.min_prints_per_sec:
                return

            if self.equity <= 50:
                return

            use_margin = min(self.margin, self.equity * 0.5)
            if use_margin < 5:
                return

            # Confidence bazlı sizing
            conf_mult = min(1.0, trs_conf + 0.5)
            use_margin *= conf_mult

            if trs_dir == 1:
                self.position = 1
            elif trs_dir == -1:
                self.position = -1

            if self.position != 0:
                self.entry_price = price
                self.entry_ts = ts_sec
                self.last_entry_ts = ts_sec
                self.position_qty = use_margin / price
                self.peak_pnl_pct = 0
                self.trailing_active = False
                fee = self.position_qty * price * self.taker_fee * self.leverage
                self.equity -= fee

        # Equity tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def get_results(self):
        """Backtest sonuçlarını döndür."""
        import numpy as np

        if not self.trades:
            return {"net_pnl": 0, "profit_factor": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0}

        pnls = np.array([t["pnl"] for t in self.trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001

        hold_times = [t["hold"] for t in self.trades if t["hold"] > 0]
        avg_hold = np.mean(hold_times) if hold_times else 0

        exit_types = {}
        for t in self.trades:
            exit_types[t["type"]] = exit_types.get(t["type"], 0) + 1

        return {
            "net_pnl": round(self.equity - 1000, 2),
            "net_pnl_pct": round((self.equity - 1000) / 1000 * 100, 2),
            "profit_factor": round(gross_profit / gross_loss, 3),
            "win_rate": round(len(wins) / len(pnls) * 100, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "total_trades": len(pnls),
            "equity_final": round(self.equity, 2),
            "avg_hold_sec": round(avg_hold, 0),
            "exit_types": exit_types,
        }
