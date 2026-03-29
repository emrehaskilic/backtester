"""
PMax DCA Strategy — Limit Order Simülasyonlu.

Mimari:
  - PMax flip (mum kapanışı) → market order ile ilk giriş + reversal
  - DCA girişleri: KC band'dan hesaplanan seviyelerde limit order (tick bazlı)
  - TP çıkışları: KC karşı band'dan hesaplanan seviyelerde limit order (tick bazlı)
  - Gerçeklik filtresi: fiyat seviyeyi en az 1 tick penetre etmeli

Komisyonlar:
  - Limit order (DCA/TP): maker %0.02
  - Market order (PMax flip): taker %0.05
"""

import math


class LimitOrder:
    """Bekleyen limit emir."""
    __slots__ = ["side", "price", "margin", "qty", "order_type", "tp_pct", "level"]

    def __init__(self, side, price, margin, qty, order_type, tp_pct=0.0, level=0):
        self.side = side          # "BUY" veya "SELL"
        self.price = price        # hedef fiyat
        self.margin = margin      # kullanılacak margin
        self.qty = qty            # kontrat miktarı
        self.order_type = order_type  # "DCA" veya "TP"
        self.tp_pct = tp_pct      # TP ise: pozisyonun yüzde kaçı kapatılacak
        self.level = level        # kademe numarası (1-4)


class Position:
    """Açık pozisyon takibi."""

    def __init__(self):
        self.side = 0             # 1=LONG, -1=SHORT, 0=yok
        self.entries = []         # [(price, qty, margin), ...]
        self.avg_price = 0.0
        self.total_qty = 0.0
        self.total_margin = 0.0
        self.dca_count = 0        # kaç DCA doldu (0=sadece ilk giriş)
        self.tp_count = 0         # kaç TP doldu
        self.initial_qty = 0.0    # ilk giriş qty (TP yüzde hesabı için)

    def add_entry(self, price, qty, margin):
        """Yeni giriş ekle (ilk giriş veya DCA)."""
        self.entries.append((price, qty, margin))
        new_cost = self.avg_price * self.total_qty + price * qty
        self.total_qty += qty
        self.total_margin += margin
        self.avg_price = new_cost / self.total_qty if self.total_qty > 0 else price

    def remove_qty(self, qty):
        """TP ile miktar azalt."""
        self.total_qty -= qty
        if self.total_qty < 1e-12:
            self.total_qty = 0.0

    def unrealized_pnl(self, current_price, leverage):
        """Anlık PnL."""
        if self.side == 0 or self.total_qty == 0:
            return 0.0
        if self.side == 1:
            return self.total_qty * (current_price - self.avg_price) * leverage
        else:
            return self.total_qty * (self.avg_price - current_price) * leverage

    def unrealized_pnl_pct(self, current_price):
        """Anlık PnL yüzdesi."""
        if self.avg_price == 0:
            return 0.0
        if self.side == 1:
            return (current_price - self.avg_price) / self.avg_price * 100
        else:
            return (self.avg_price - current_price) / self.avg_price * 100


class PMaxDCAStrategy:
    """
    Adaptive PMax + Keltner Channel DCA/TP Stratejisi.
    Dual-loop: tick bazlı limit order + mum kapanışı PMax/KC.
    """

    def __init__(self, params=None):
        p = params or {}

        # === Sabit parametreler ===
        self.initial_balance = float(p.get("initial_balance", 1000))
        self.leverage = int(p.get("leverage", 25))
        self.maker_fee = float(p.get("maker_fee", 0.0002))
        self.taker_fee = float(p.get("taker_fee", 0.0005))
        self.margin_ratio = float(p.get("margin_ratio", 1.0 / 40.0))  # kasa/40 = %2.5
        self.max_dca = int(p.get("max_dca", 4))

        # === DCA multipliers (KC band mesafesi katları) ===
        self.dca_multipliers = [
            float(p.get("dca_m1", 0.50)),
            float(p.get("dca_m2", 3.00)),
            float(p.get("dca_m3", 3.75)),
            float(p.get("dca_m4", 3.75)),
        ]

        # === TP seviyeleri (KC karşı band mesafesi katları) ===
        # tp değerleri: pozisyonun yüzde kaçının kapatılacağı (kümülatif değil)
        self.tp_levels = [
            float(p.get("tp1", 0.50)),
            float(p.get("tp2", 0.80)),
            float(p.get("tp3", 0.85)),
            float(p.get("tp4", 0.90)),
        ]

        # === State ===
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

        self.position = Position()
        self.pending_dca_orders = []   # List[LimitOrder]
        self.pending_tp_orders = []    # List[LimitOrder]

        # Trade log
        self.trades = []
        self.trade_groups = []  # her pozisyon açılış-kapanış grubu

        # Mevcut KC bandları (son mum kapanışından)
        self.kc_upper = 0.0
        self.kc_lower = 0.0
        self.kc_middle = 0.0
        self.kc_atr = 0.0

        # PMax state
        self.pmax_direction = 0
        self.pmax_ready = False

        # Aktif trade grubu
        self._current_group = None

        # İstatistikler
        self.total_fees_paid = 0.0

    def _calc_margin(self):
        """Dinamik margin: equity / 40."""
        return max(0.0, self.equity * self.margin_ratio)

    def _calc_qty(self, price, margin):
        """Margin'dan kontrat miktarı hesapla."""
        if price <= 0:
            return 0.0
        return margin / price

    # ================================================================
    # MUM KAPANIŞI EVENTI
    # ================================================================

    def on_candle_close(self, pmax_flipped, pmax_direction, pmax_stop,
                        kc_upper, kc_middle, kc_lower, kc_atr,
                        candle_close, candle_ts):
        """
        Her 3dk mum kapanışında çağrılır.
        PMax ve KC güncellemesi yapılmış olarak gelir.
        """
        self.kc_upper = kc_upper
        self.kc_lower = kc_lower
        self.kc_middle = kc_middle
        self.kc_atr = kc_atr
        self.pmax_direction = pmax_direction

        if pmax_direction != 0:
            self.pmax_ready = True

        if not self.pmax_ready:
            return

        # PMax flip → hard stop + reversal
        if pmax_flipped and pmax_direction != 0:
            self._handle_pmax_flip(pmax_direction, candle_close, candle_ts)

        # Açık pozisyon varsa ve KC güncellendiyse → TP emirlerini güncelle
        if self.position.side != 0:
            self._update_tp_orders(candle_close)
            # DCA emirlerini de KC'ye göre güncelle
            self._update_dca_orders(candle_close)

    def _handle_pmax_flip(self, new_direction, price, ts):
        """PMax yön değişimi: tüm pozisyonu kapat + ters yöne aç."""
        # 1. Mevcut pozisyonu kapat (market order — taker fee)
        if self.position.side != 0:
            self._close_entire_position(price, ts, "PMAX_FLIP", use_taker=True)

        # 2. Tüm bekleyen emirleri iptal et
        self.pending_dca_orders.clear()
        self.pending_tp_orders.clear()

        # 3. Ters yöne yeni pozisyon aç (market order — taker fee)
        margin = self._calc_margin()
        if margin < 0.1 or self.equity < 1:
            return

        qty = self._calc_qty(price, margin)
        if qty <= 0:
            return

        side = new_direction  # 1=LONG, -1=SHORT
        fee = qty * price * self.taker_fee * self.leverage
        self.equity -= fee
        self.total_fees_paid += fee

        self.position = Position()
        self.position.side = side
        self.position.add_entry(price, qty, margin)
        self.position.initial_qty = self.position.total_qty

        # Trade grubu başlat
        self._current_group = {
            "entry_ts": ts,
            "entry_price": price,
            "side": "LONG" if side == 1 else "SHORT",
            "entries": [{"price": price, "qty": qty, "margin": margin, "type": "ENTRY", "fee": fee}],
            "exits": [],
            "dca_count": 0,
            "tp_count": 0,
        }

        self.trades.append({
            "ts": ts, "type": "ENTRY", "side": "LONG" if side == 1 else "SHORT",
            "price": price, "qty": qty, "margin": margin, "fee": fee,
            "equity_after": self.equity,
        })

        # 4. DCA ve TP emirlerini kur
        self._setup_dca_orders(price)
        self._setup_tp_orders(price)

    def _setup_dca_orders(self, entry_price):
        """KC band'a göre DCA limit emirlerini kur."""
        self.pending_dca_orders.clear()

        if self.kc_atr == 0:
            return

        side = self.position.side
        kc_band_width = self.kc_atr * 0.5  # KC multiplier = 0.5

        for i, mult in enumerate(self.dca_multipliers):
            if i >= self.max_dca:
                break

            # DCA seviyeleri: pozisyon yönünün tersine, KC band mesafesinin katları
            if side == 1:  # LONG — fiyat düştükçe DCA
                dca_price = self.kc_lower - kc_band_width * mult
                order_side = "BUY"
            else:  # SHORT — fiyat yükseldikçe DCA
                dca_price = self.kc_upper + kc_band_width * mult
                order_side = "SELL"

            # Margin henüz belli değil — dolduğunda hesaplanacak
            order = LimitOrder(
                side=order_side,
                price=dca_price,
                margin=0,  # dolduğunda equity/40 hesaplanacak
                qty=0,
                order_type="DCA",
                level=i + 1,
            )
            self.pending_dca_orders.append(order)

    def _setup_tp_orders(self, entry_price):
        """KC karşı band'a göre TP limit emirlerini kur."""
        self.pending_tp_orders.clear()

        if self.kc_atr == 0:
            return

        side = self.position.side
        kc_band_width = self.kc_atr * 0.5  # KC multiplier = 0.5

        for i, tp_ratio in enumerate(self.tp_levels):
            # TP seviyeleri: KC karşı band mesafesinin katları
            if side == 1:  # LONG — fiyat yükseldikçe TP (üst band)
                tp_price = self.kc_upper + kc_band_width * tp_ratio
                order_side = "SELL"
            else:  # SHORT — fiyat düştükçe TP (alt band)
                tp_price = self.kc_lower - kc_band_width * tp_ratio
                order_side = "BUY"

            # TP yüzdeleri: kümülatif değil, her kademe toplam pozisyonun o yüzdesidir
            # TP1=%50 → %50 kapat
            # TP2=%80 → %80 kapat (kalan pozisyonun tamamının %80'i değil, orijinalin)
            # Ama kümülatif mantıkla: TP1'de %50, TP2'de kalanın bir kısmı...
            # Aslında bu yüzdeler toplam pozisyon üzerinden kümülatif:
            # TP1: %50 kapat (kalan %50)
            # TP2: %80 kapat (önceden %50 kapatıldı, şimdi %30 daha = toplamda %80)
            # TP3: %85 kapat (%5 daha)
            # TP4: %90 kapat (%5 daha, kalan %10 runner)
            order = LimitOrder(
                side=order_side,
                price=tp_price,
                margin=0,
                qty=0,
                order_type="TP",
                tp_pct=tp_ratio,
                level=i + 1,
            )
            self.pending_tp_orders.append(order)

    def _update_tp_orders(self, current_price):
        """KC güncellendiyse TP seviyelerini yeniden hesapla."""
        if not self.pending_tp_orders or self.kc_atr == 0:
            return

        side = self.position.side
        kc_band_width = self.kc_atr * 0.5

        for order in self.pending_tp_orders:
            if side == 1:
                order.price = self.kc_upper + kc_band_width * order.tp_pct
            else:
                order.price = self.kc_lower - kc_band_width * order.tp_pct

    def _update_dca_orders(self, current_price):
        """KC güncellendiyse DCA seviyelerini yeniden hesapla."""
        if not self.pending_dca_orders or self.kc_atr == 0:
            return

        side = self.position.side
        kc_band_width = self.kc_atr * 0.5

        for order in self.pending_dca_orders:
            mult = self.dca_multipliers[order.level - 1]
            if side == 1:
                order.price = self.kc_lower - kc_band_width * mult
            else:
                order.price = self.kc_upper + kc_band_width * mult

    # ================================================================
    # TICK LOOP — LİMİT ORDER KONTROL
    # ================================================================

    def on_tick(self, ts_ms, price, quantity, is_buyer_maker):
        """
        Her tick'te çağrılır.
        Bekleyen limit emirleri kontrol eder.
        Gerçeklik filtresi: fiyat seviyeyi en az 1 tick penetre etmeli.
        """
        if self.position.side == 0:
            return

        ts_sec = ts_ms // 1000

        # DCA emirlerini kontrol et
        filled_dca = []
        for order in self.pending_dca_orders:
            if self._is_limit_filled(order, price):
                filled_dca.append(order)

        for order in filled_dca:
            self._fill_dca_order(order, ts_sec)
            self.pending_dca_orders.remove(order)

        # TP emirlerini kontrol et
        filled_tp = []
        for order in self.pending_tp_orders:
            if self._is_limit_filled(order, price):
                filled_tp.append(order)

        for order in filled_tp:
            self._fill_tp_order(order, ts_sec)
            self.pending_tp_orders.remove(order)

        # Equity tracking
        self._update_equity_tracking()

    def _is_limit_filled(self, order, current_price):
        """
        Limit emir doldu mu? Gerçeklik filtresi:
        Fiyat, emrin bulunduğu seviyeyi en az 1 tick penetre etmeli.
        Sadece dokunup dönmesi yetmez.
        """
        if order.side == "BUY":
            # Alım emri: fiyat emrin altına düşmeli (penetrasyon)
            return current_price < order.price
        else:
            # Satım emri: fiyat emrin üstüne çıkmalı (penetrasyon)
            return current_price > order.price

    def _fill_dca_order(self, order, ts_sec):
        """DCA limit emri doldu."""
        margin = self._calc_margin()
        if margin < 0.1 or self.equity < 1:
            return

        qty = self._calc_qty(order.price, margin)
        if qty <= 0:
            return

        # Maker fee
        fee = qty * order.price * self.maker_fee * self.leverage
        self.equity -= fee
        self.total_fees_paid += fee

        self.position.add_entry(order.price, qty, margin)
        self.position.dca_count += 1
        # initial_qty'yi güncelle (TP hesabı tüm pozisyon üzerinden)
        self.position.initial_qty = self.position.total_qty

        if self._current_group:
            self._current_group["dca_count"] += 1
            self._current_group["entries"].append({
                "price": order.price, "qty": qty, "margin": margin,
                "type": f"DCA{order.level}", "fee": fee,
            })

        self.trades.append({
            "ts": ts_sec, "type": f"DCA{order.level}",
            "side": "LONG" if self.position.side == 1 else "SHORT",
            "price": order.price, "qty": qty, "margin": margin, "fee": fee,
            "avg_price": self.position.avg_price,
            "equity_after": self.equity,
        })

        # DCA sonrası TP emirlerini yeniden hesapla (pozisyon büyüdü)
        self._recalc_tp_after_dca()

    def _recalc_tp_after_dca(self):
        """DCA sonrası TP emirlerinin qty'lerini güncelle."""
        # TP yüzdeleri toplam pozisyon üzerinden kümülatif
        # initial_qty güncellendi, TP qty'ler buna göre hesaplanacak
        pass  # qty, fill anında hesaplanıyor

    def _fill_tp_order(self, order, ts_sec):
        """TP limit emri doldu."""
        if self.position.total_qty <= 0:
            return

        # Kümülatif TP hesabı:
        # tp_pct = toplam pozisyonun bu noktaya kadar kapatılması gereken yüzdesi
        # Daha önce kapatılan miktarı çıkar
        already_closed = self.position.initial_qty - self.position.total_qty
        target_closed = self.position.initial_qty * order.tp_pct
        close_qty = target_closed - already_closed

        if close_qty <= 0:
            return

        close_qty = min(close_qty, self.position.total_qty)

        # PnL hesapla
        if self.position.side == 1:
            pnl = close_qty * (order.price - self.position.avg_price) * self.leverage
        else:
            pnl = close_qty * (self.position.avg_price - order.price) * self.leverage

        # Maker fee
        fee = close_qty * order.price * self.maker_fee * self.leverage
        net_pnl = pnl - fee
        self.equity += net_pnl
        self.total_fees_paid += fee

        self.position.remove_qty(close_qty)
        self.position.tp_count += 1

        if self._current_group:
            self._current_group["tp_count"] += 1
            self._current_group["exits"].append({
                "price": order.price, "qty": close_qty,
                "type": f"TP{order.level}", "pnl": net_pnl, "fee": fee,
            })

        self.trades.append({
            "ts": ts_sec, "type": f"TP{order.level}",
            "side": "CLOSE",
            "price": order.price, "qty": close_qty, "pnl": net_pnl, "fee": fee,
            "remaining_qty": self.position.total_qty,
            "equity_after": self.equity,
        })

        # Pozisyon tamamen kapandıysa
        if self.position.total_qty <= 1e-12:
            self._finalize_group(ts_sec)

    def _close_entire_position(self, price, ts_sec, reason, use_taker=True):
        """Tüm pozisyonu kapat."""
        if self.position.side == 0 or self.position.total_qty <= 0:
            return

        qty = self.position.total_qty

        if self.position.side == 1:
            pnl = qty * (price - self.position.avg_price) * self.leverage
        else:
            pnl = qty * (self.position.avg_price - price) * self.leverage

        fee_rate = self.taker_fee if use_taker else self.maker_fee
        fee = qty * price * fee_rate * self.leverage
        net_pnl = pnl - fee
        self.equity += net_pnl
        self.total_fees_paid += fee

        if self._current_group:
            self._current_group["exits"].append({
                "price": price, "qty": qty,
                "type": reason, "pnl": net_pnl, "fee": fee,
            })

        self.trades.append({
            "ts": ts_sec, "type": reason, "side": "CLOSE",
            "price": price, "qty": qty, "pnl": net_pnl, "fee": fee,
            "equity_after": self.equity,
        })

        self._finalize_group(ts_sec)

        self.position = Position()
        self.pending_dca_orders.clear()
        self.pending_tp_orders.clear()

    def _finalize_group(self, ts_sec):
        """Trade grubunu tamamla ve kaydet."""
        if self._current_group:
            self._current_group["exit_ts"] = ts_sec
            total_pnl = sum(e.get("pnl", 0) for e in self._current_group["exits"])
            self._current_group["total_pnl"] = total_pnl
            self._current_group["equity_after"] = self.equity
            self.trade_groups.append(self._current_group)
            self._current_group = None

    def _update_equity_tracking(self):
        """Max drawdown tracking."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - self.equity) / self.peak_equity * 100
            if dd_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = dd_pct

    # ================================================================
    # SONUÇLAR
    # ================================================================

    def get_results(self):
        """Backtest sonuçlarını döndür."""
        if not self.trade_groups:
            return {
                "net_pnl": 0, "net_pnl_pct": 0, "total_groups": 0,
                "win_rate": 0, "profit_factor": 0, "max_drawdown_pct": 0,
                "equity_final": self.equity, "total_fees": self.total_fees_paid,
            }

        pnls = [g["total_pnl"] for g in self.trade_groups]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001

        return {
            "net_pnl": round(self.equity - self.initial_balance, 2),
            "net_pnl_pct": round((self.equity - self.initial_balance) / self.initial_balance * 100, 2),
            "equity_final": round(self.equity, 2),
            "total_groups": len(self.trade_groups),
            "total_trades": len(self.trades),
            "win_rate": round(len(wins) / len(pnls) * 100, 2) if pnls else 0,
            "profit_factor": round(gross_profit / gross_loss, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_fees": round(self.total_fees_paid, 2),
            "avg_dca_per_group": round(sum(g["dca_count"] for g in self.trade_groups) / len(self.trade_groups), 2),
            "avg_tp_per_group": round(sum(g["tp_count"] for g in self.trade_groups) / len(self.trade_groups), 2),
        }

    def print_trade_log(self):
        """Trade logunu yazdır."""
        print(f"\n{'='*80}")
        print(f"  TRADE LOG")
        print(f"{'='*80}")
        for i, g in enumerate(self.trade_groups):
            pnl = g["total_pnl"]
            icon = "+" if pnl > 0 else ""
            print(f"  #{i+1:3d} | {g['side']:5s} | Entry: {g['entry_price']:.2f} | "
                  f"DCA: {g['dca_count']} | TP: {g['tp_count']} | "
                  f"PnL: {icon}{pnl:.2f}")
