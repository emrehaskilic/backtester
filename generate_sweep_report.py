"""Weekly Daily 8H 4H 1H Sweep Report -PDF"""
import json, sys
from fpdf import FPDF

class SweepReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Weekly Daily 8H 4H 1H Sweep Pattern Discovery Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9)
        self.cell(0, 5, 'ETHUSDT 5m | 11 Ay | Parametrik Sweep Pattern Discovery v3', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(40, 40, 60)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f'  {title}', fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(40, 40, 120)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def table_header(self, cols, widths):
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(220, 220, 240)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, fill=True, align='C')
        self.ln()

    def table_row(self, vals, widths, bold=False):
        self.set_font('Helvetica', 'B' if bold else '', 8)
        for val, w in zip(vals, widths):
            self.cell(w, 5, str(val), border=1, align='C')
        self.ln()

pdf = SweepReport()
pdf.set_auto_page_break(auto=True, margin=20)

# === SAYFA 1: PLAN ===
pdf.add_page()
pdf.section_title('1. PLAN OZETI')

pdf.sub_title('Amac')
pdf.body_text('Haftalik ve gunluk mum high/low seviyeleri sweep edildiginde, saf order flow ve istatistik ile continuation vs reversal\'i ayiran kosullari bulmak. Tum parametreler veriden turetilir.')

pdf.sub_title('Veri')
pdf.body_text('5m ETHUSDT 11 ay (~96K bar) - native OI dahil, resampling yok\n5m\'den turetilen: gunluk mumlar (UTC 00:00), haftalik mumlar (Pazartesi UTC 00:00)\nTum order flow verileri (fiyat, hacim, CVD, OI) %100 native ve senkronize')

pdf.sub_title('Sweep Tespiti (Adim 1)')
pdf.body_text('10 sweep tipi: Haftalik high/low, Gunluk high/low, 8H high/low, 4H high/low, 1H high/low\nEntry fiyati = kirilan seviyenin kendisi (bar kapanisi degil)\nFeature\'lar sweep bar\'indan bir onceki bar\'in degerleriyle hesaplanir (look-ahead korumasi)\nReset logic: Fiyat seviyeden 0.5 x Prev_Day_ATR geri donerse veya 12 bar (1 saat) ters tarafta kapatirsa seviye resetlenir')

pdf.sub_title('Feature\'lar (Adim 2) - 7 adet, saf order flow')
features_text = '''1. CVD z-score (mikro) - 12-bar rolling (1 saat)
2. CVD z-score (makro) - 288-bar rolling (24 saat)
3. OI change - 288-bar, Previous Day ATR normalize
4. Volume z-score (mikro) - 12-bar
5. Volume z-score (makro) - 288-bar
6. Imbalance (smoothed) - EMA_12 kumulatif
7. ATR percentile - 288-bar rolling rank'''
pdf.body_text(features_text)

pdf.sub_title('Etiketleme (Adim 3) - Path-Dependent Triple Barrier')
pdf.body_text('ATR referansi: Previous Day ATR (bir onceki kapanmis gunun 14-bar Daily ATR\'si)\nSimetrik grid: tp_mult == sl_mult (random walk yanilgisi korumasi)\nHigh sweep: TP=yukari (cont), SL=asagi (rev) | Low sweep: TP=asagi (cont), SL=yukari (rev)\nEtiket: TP\'ye ONCE dokundu = Continuation | SL\'ye ONCE dokundu = Reversal | Timeout = analiz disi\nTimeout: Weekly 1-5 gun, Daily 2-24h, 8H 4-24h, 4H 1-12h, 1H 30m-6h')

pdf.sub_title('Parametre Secimi (Adim 4)')
pdf.body_text('Tek kriter: Ayrisma gucu = |continuation - reversal| orani\n"En yuksek WR" aranmaz (curve fitting korumasi)\nGunluk sweep: N >= 100 | Timeout < %25')

pdf.sub_title('Feature Analizi (Adim 5)')
pdf.body_text('Grup karsilastirma: Median (mean degil, fat-tail korumasi) + Mann-Whitney U testi\nQuantile analizi: Her feature 5 bucket (Q1-Q5), piyasa kendi uclariyla olculur\nKombinatoryal arama: k=2 (tum Q) + k=3 (Q1/Q3/Q5), min sample >= 30')

pdf.sub_title('Dogrulama (Adim 6)')
pdf.body_text('Walk-Forward: 3 ay train -> 1 ay test, 8 pencere (statik train/test bolunme iptal)\nAylik tutarlilik: Min %70 pencerede pozitif ayrisma\nFDR Correction: Benjamini-Hochberg alpha=0.05')

# === SAYFA 2: GENEL SONUCLAR ===
pdf.add_page()
pdf.section_title('2. GENEL SONUCLAR')

with open('results/sweep_miner_results.json') as f:
    data = json.load(f)

pdf.body_text(f'Toplam sure: {data["elapsed"]}s | Toplam sweep eventi: {data["total_events"]} | Grid testleri: {data["total_grid"]:,}')
pdf.ln(2)

# Ozet tablo
pdf.sub_title('Sweep Tipi Ozet')
cols = ['Sweep Tipi', 'Event', 'Cont%', 'Rev%', 'Ayrisma%', 'Mult', 'Timeout']
widths = [40, 20, 20, 20, 25, 20, 25]
pdf.table_header(cols, widths)

for sr in data['results']:
    if sr['type'] in ['15m High Sweep', '15m Low Sweep']:
        continue
    bg = sr['best_grid']
    timeout_h = f"{bg['timeout_bars']*5/60:.0f}h"
    pdf.table_row([
        sr['type'], sr['events'],
        f"{sr['base_cont']:.1f}", f"{sr['base_rev']:.1f}",
        f"{bg['separation']:.1f}", f"{bg['mult']:.1f}", timeout_h
    ], widths)

pdf.ln(3)
pdf.sub_title('Kritik Bulgu: OI Change Her Yerde Anlamli')
pdf.body_text('Tum 4 sweep tipinde OI change, continuation vs reversal\'i ayiran en guclu feature.\nOI artiyorsa sweep devam ediyor (yeni pozisyonlar aciliyor).\nOI dusuyorsa reversal geliyor (pozisyonlar kapaniyor, likidite cekiliyor).')

# === SAYFA 3+: HER SWEEP TIPI DETAY (15m haric) ===
skip_types = ['15m High Sweep', '15m Low Sweep']
for sr in data['results']:
    if sr['type'] in skip_types:
        continue
    pdf.add_page()
    pdf.section_title(f'3. {sr["type"]}')

    bg = sr['best_grid']
    pdf.sub_title('Optimal Parametreler')
    timeout_h = f"{bg['timeout_bars']*5/60:.0f}h"
    pdf.body_text(f'mult: {bg["mult"]:.1f} | timeout: {bg["timeout_bars"]} bar ({timeout_h})\nN: {bg["n_total"]} | Cont: {bg["n_continuation"]} ({bg["continuation_rate"]:.1f}%) | Rev: {bg["n_reversal"]} ({bg["reversal_rate"]:.1f}%) | Timeout: {bg["n_timeout"]} ({bg["timeout_rate"]:.1f}%)\nAyrisma gucu: {bg["separation"]:.1f}%')

    pdf.sub_title('Feature Karsilastirma (Continuation vs Reversal)')
    cols = ['Feature', 'Cont Med', 'Rev Med', 'p-value', 'Sig']
    widths = [45, 30, 30, 30, 15]
    pdf.table_header(cols, widths)
    for fc in sr['features']:
        sig = "***" if fc['significant'] else ""
        pdf.table_row([
            fc['feature'], f"{fc['cont_median']:+.3f}", f"{fc['rev_median']:+.3f}",
            f"{fc['p_value']:.6f}", sig
        ], widths, bold=fc['significant'])

    pdf.ln(3)

    if sr['patterns']:
        pdf.sub_title(f'Top Pattern\'ler (FDR + Walk-Forward filtered)')
        cols = ['#', 'Cont%', 'N', 'p-value', 'WF', 'Cons%', 'Conditions']
        widths = [8, 18, 15, 25, 15, 18, 91]
        pdf.table_header(cols, widths)
        for i, p in enumerate(sr['patterns']):
            wf = f"{p['wf_positive']}/{p['wf_total']}" if p['wf_total'] > 0 else "N/A"
            pdf.table_row([
                str(i+1), f"{p['continuation_rate']:.1f}", str(p['n']),
                f"{p['p_value']:.6f}", wf, f"{p['wf_consistency']:.0f}",
                p['conditions']
            ], widths)
    else:
        pdf.body_text('Pattern bulunamadi (FDR veya walk-forward filtresi gecemedi)')

# === SON SAYFA: META-PATTERN ===
pdf.add_page()
pdf.section_title('4. META-PATTERN OZETI')

pdf.sub_title('Sweep Sonrasi Continuation Olasligini Artiran Kosullar')
pdf.body_text('''1. OI ARTIYOR (en guclu sinyal)
   Yeni pozisyonlar aciliyor, hareket gercek. Tum sweep tiplerinde istatistiksel olarak anlamli (p < 0.05).
   Weekly Low: OI cont median +160 vs rev median -113 (p=0.001)
   Daily Low: OI cont median +35 vs rev median -109 (p=0.000000)

2. ATR YUKSEK
   Volatil piyasada sweep\'ler continuation egilimli.
   Daily High ATR Q5: %69.9 continuation (base %53.9)
   Daily Low ATR Q4-Q5: %63-67 continuation (base %59.7)

3. CVD UCLARDA
   Guclu tek yonlu akis varken sweep devam ediyor.
   Daily High CVD_macro Q5: %64.9 continuation (base %53.9)''')

pdf.sub_title('Reversal Sinyali')
pdf.body_text('''OI dusuyorsa pozisyonlar kapaniyor, likidite cekiliyor, hareket sahte.
Daily Low OI Q1: sadece %43.2 continuation (base %59.7) = %57 reversal olasiligi''')

pdf.sub_title('En Guclu Trade-Edilebilir Pattern\'ler')

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, '1H SWEEP (En buyuk sample + en yuksek WR)', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 9)
pdf.body_text('''1H Low Sweep + CVD_micro Q1 + Imbalance Q4:
  Continuation %97.2 | N=607 | Walk-Forward 4/4 (%100)
  Yorum: 1H low kirildiktan sonra, mikro CVD cok negatif (satis devam) + alis baskisi
  yuksekse (imbalance Q4), %97 olasilikla asagi devam. 607 ornek -en guvenilir pattern.

1H Low Sweep + CVD_macro Q3 + Imbalance Q4:
  Continuation %88.3 | N=857 | Walk-Forward 6/7 (%86)
  Yorum: En buyuk sample (857). Daha gevsek CVD kosulu ama hala cok guclu.

1H High Sweep + OI_change Q4 + Vol_micro Q3:
  Continuation %88.1 | N=695 | Walk-Forward 6/7 (%86)
  Yorum: 1H high kirildiktan sonra, OI artiyor + orta volume = gercek breakout. 695 ornek.

1H High Sweep + OI_change Q4 + ATR Q5:
  Continuation %85.3 | N=761 | Walk-Forward 3/4 (%75)
  Yorum: OI artiyor + yuksek volatilite. En buyuk sample (761).''')

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, '8H SWEEP (Sweet spot -anlamli seviyeler + iyi WF)', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 9)
pdf.body_text('''8H High Sweep + Imbalance Q4 + ATR Q4:
  Continuation %86.4 | N=88 | Walk-Forward 5/5 (%100)
  Yorum: 8H high kirildiktan sonra, alis baskisi yuksek + volatilite yuksek = gercek breakout.

8H Low Sweep + OI_change Q5 + Imbalance Q4:
  Continuation %84.8 | N=132 | Walk-Forward 3/3 (%100)
  Yorum: OI artiyor + alis/satis dengesizligi = hareket gercek.

8H Low Sweep + OI_change Q5 + Vol_micro Q1:
  Continuation %86.7 | N=105 | Walk-Forward 4/4 (%100)
  Yorum: OI artiyor + dusuk volume = sessiz likidasyon, hareket devam edecek.

8H Low Sweep + OI Q5 + Imbalance Q5 + ATR Q3:
  Continuation %96.8 | N=63 | Walk-Forward 4/4 (%100)
  Yorum: En yuksek cont rate. Kucuk sample ama WF cok guclu.''')

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, '4H SWEEP (Guclu sample + yuksek WF tutarliligi)', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 9)
pdf.body_text('''4H Low Sweep + CVD_micro Q1 + ATR Q4:
  Continuation %84.5 | N=207 | Walk-Forward 7/7 (%100)
  Yorum: 4H low kirildiktan sonra, mikro CVD cok negatif (satis baskisi devam) + volatilite yuksekse,
  %85 olasilikla asagi devam ediyor. 207 ornek ve 7/7 WF tutarliligi -en guvenilir pattern.

4H Low Sweep + CVD_micro Q2 + ATR Q5:
  Continuation %77.8 | N=297 | Walk-Forward 5/6 (%83)
  Yorum: En buyuk sample (297). Daha gevsek kosullar ama hala guclu.

4H High Sweep + CVD_macro Q5 + Vol_micro Q3:
  Continuation %85.8 | N=226 | Walk-Forward 4/4 (%100)
  Yorum: 4H high kirildiktan sonra, 24 saatlik CVD cok guclu + orta volume ise,
  %86 olasilikla yukari devam. 226 ornek.

4H High Sweep + Imbalance Q4 + ATR Q4:
  Continuation %85.2 | N=162 | Walk-Forward 5/5 (%100)
  Yorum: Alis baskisi yuksek + volatilite yuksek = gercek breakout.''')

pdf.set_font('Helvetica', 'B', 9)
pdf.cell(0, 6, 'DAILY SWEEP', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 9)
pdf.body_text('''Daily High Sweep + CVD_macro Q5 + ATR Q5:
  Continuation %82.7 | N=52 | Walk-Forward 3/3 (%100)

Daily Low Sweep + OI_change Q5 + ATR Q3:
  Continuation %88.9 | N=45 | Walk-Forward 3/3 (%100)

Daily Low Sweep + Vol_micro Q4 + ATR Q4:
  Continuation %96.9 | N=32 | Walk-Forward 3/3 (%100)
  Not: Sample kucuk (32-52), 4H pattern\'lerden daha az guvenilir.''')

pdf.sub_title('Uyarilar')
pdf.body_text('''1. Daily sweep sample\'lari kucuk (32-53). 1H, 4H ve 8H cok daha guclu (88-900).
2. Haftalik sweep\'lerde pattern bulunamadi (sample yetersiz, 89-126 event).
3. 1H sweep\'ler en buyuk sample ve en yuksek WR verdi (607 ornek, %97 cont rate).
4. 4H sweep\'ler en yuksek WF tutarliligi verdi (7/7 pencere).
5. 8H sweep\'ler sweet spot -anlamli seviyeler + guclu WF (5/5, 4/4).
6. Fee ve slippage modele dahil degil. Gercek trade\'de edge daralacaktir.
7. Bu sonuclar ETHUSDT\'ye ozeldir. Diger coinlerde (BTC, SOL) test edilmeden genelleme yapilamaz.
8. OI change tum timeframe\'lerde en guclu ayirici -bu yapisal bir edge isaretcidir.
9. 1H pattern\'lerinin bir kismi WF sonucu N/A -daha uzun veri ile dogrulanmali.''')

pdf.sub_title('Karsilastirma: Hangi Timeframe Daha Guvenilir?')
pdf.body_text('''1H > 4H > 8H > Daily > Weekly

1H Avantajlari:
- En buyuk sample (22K-29K event, pattern basina 500-900)
- En yuksek filtered cont rate (%88-97)
- OI + CVD + Imbalance kombinasyonlari cok guclu

8H Avantajlari (sweet spot):
- Anlamli seviyeler (gunluk yapiyi yansitir)
- Cok yuksek WF tutarliligi (5/5, 4/4, 6/7)
- Imbalance + OI kombinasyonlari dominant

4H Avantajlari:
- En yuksek WF tutarliligi (7/7 pencere)
- Tum 7 feature istatistiksel olarak anlamli (p < 0.05)
- Buyuk sample (200-477 per pattern)

Daily Avantajlari:
- Daha yuksek base ayrisma gucu (%19 vs %8.5)
- Gunluk seviyeler institutional akis ile uyumlu

Weekly:
- En yuksek base continuation rate (%71-74) ama sample cok kucuk
- Pattern bulunamadi -daha uzun veri gerekli (2-3 yil)

ONERILEN STRATEJI:
1H veya 4H sweep tespiti + 8H veya Daily dogrulamasi (multi-timeframe)
Birden fazla TF\'de ayni yonde sweep = en guclu sinyal
OI artisi + yuksek ATR = continuation icin en guvenilir filtre''')

# Kaydet
output_path = 'C:/Users/emrehaskilic/Desktop/weekly_daily_8h_4h_1h_sweep.pdf'
pdf.output(output_path)
print(f'PDF kaydedildi: {output_path}')
