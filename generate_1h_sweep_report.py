"""1H Candle Sweep Analysis Report - PDF"""
import json, sys
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, '1H Candle Sweep Pattern Discovery Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9)
        self.cell(0, 5, 'ETHUSDT 5m Native | 11 Ay | Mum Kapanisi Bazli Analiz', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', align='C')

    def section(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(40, 40, 60)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f'  {title}', fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def sub(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(40, 40, 120)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def txt(self, text):
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def th(self, cols, widths):
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(220, 220, 240)
        for c, w in zip(cols, widths):
            self.cell(w, 6, c, border=1, fill=True, align='C')
        self.ln()

    def tr(self, vals, widths, bold=False):
        self.set_font('Helvetica', 'B' if bold else '', 8)
        for v, w in zip(vals, widths):
            self.cell(w, 5, str(v), border=1, align='C')
        self.ln()

pdf = Report()
pdf.set_auto_page_break(auto=True, margin=20)

# === SAYFA 1: YONTEM ===
pdf.add_page()
pdf.section('1. YONTEM')

pdf.sub('Amac')
pdf.txt('1H mum kapanisi bazli sweep continuation/reversal pattern discovery. Parametre optimizasyonu yok, piyasa yapisina dayali.')

pdf.sub('Veri')
pdf.txt('5m ETHUSDT 11 ay (96K bar), native OI dahil, resampling yok.')

pdf.sub('Sweep + Mum Kapanisi Tanimi')
pdf.txt('''HIGH SWEEP sonrasi:
- Close > prev_high = Continuation (LONG sinyal)
- Close < prev_high + kirmizi mum = Reversal (SHORT sinyal)
- Close < prev_high + yesil mum = Belirsiz (pas)

LOW SWEEP sonrasi:
- Close < prev_low = Continuation (SHORT sinyal)
- Close > prev_low + yesil mum = Reversal (LONG sinyal)
- Close > prev_low + kirmizi mum = Belirsiz (pas)

Inside bar (ne high ne low sweep) = Pas''')

pdf.sub('Feature\'lar (7 adet, saf order flow)')
pdf.txt('''1. CVD z-score micro (12-bar, 1 saat)
2. CVD z-score macro (288-bar, 24 saat)
3. OI change (288-bar, Daily ATR normalized)
4. Volume z-score micro (12-bar)
5. Volume z-score macro (288-bar)
6. Imbalance smoothed (EMA_12)
7. ATR percentile (288-bar rolling rank)''')

pdf.sub('Pattern Mining')
pdf.txt('''Quantile encoding: Her feature 5 bucket (Q1-Q5)
Kombinasyon: k=2 (tum Q) + k=3 (Q1/Q3/Q5)
Min sample: 30 per kesisim
FDR Correction: Benjamini-Hochberg alpha=0.05
Walk-Forward: 8 pencere (3 ay train / 1 ay test)''')

# === SAYFA 2: DAGILIM ===
pdf.add_page()
pdf.section('2. SINYAL DAGILIMI')

pdf.txt('Toplam 8,015 adet 1H mum analiz edildi.')
pdf.ln(2)

cols = ['Sinyal', 'Sayi', 'Oran', 'Anlam']
widths = [40, 20, 20, 90]
pdf.th(cols, widths)
pdf.tr(['High Continuation', '1,707', '21.3%', 'LONG - gercek breakout'], widths)
pdf.tr(['High Reversal', '843', '10.5%', 'SHORT - fake breakout, stop hunt'], widths)
pdf.tr(['High Belirsiz', '624', '7.8%', 'Pas - celiski'], widths)
pdf.tr(['Low Continuation', '1,573', '19.6%', 'SHORT - gercek kirilim'], widths)
pdf.tr(['Low Reversal', '799', '10.0%', 'LONG - fake breakdown'], widths)
pdf.tr(['Low Belirsiz', '621', '7.7%', 'Pas - celiski'], widths)
pdf.tr(['Inside Bar', '1,465', '18.3%', 'Pas - sikisma'], widths)

pdf.ln(3)
pdf.sub('Onemli Gozlem')
pdf.txt('''High sweep toplam: 3,174 event -> Cont 53.8% | Rev 26.6% | Belirsiz 19.7%
Low sweep toplam: 2,993 event -> Cont 52.6% | Rev 26.7% | Belirsiz 20.7%

Base rate yaklasik %53 continuation. Feature filtreleriyle %95+ cikarildi.''')

# === SAYFA 3: QUANTILE ANALIZI ===
pdf.add_page()
pdf.section('3. QUANTILE ANALIZI - Monoton Trendler')

pdf.sub('HIGH SWEEP - CVD Micro Continuation Rate')
pdf.txt('''Q1: %36.4 | Q2: %47.5 | Q3: %70.2 | Q4: %85.9 | Q5: %94.9

Mukemmel monoton artis! CVD ne kadar pozitifse (alici baskisi),
high sweep sonrasi continuation (yukari devam) olasiligi o kadar yuksek.''')

pdf.sub('LOW SWEEP - CVD Micro Continuation Rate')
pdf.txt('''Q1: %95.2 | Q2: %86.7 | Q3: %69.9 | Q4: %46.6 | Q5: %33.1

Mukemmel monoton dusus! CVD ne kadar negatifse (satici baskisi),
low sweep sonrasi continuation (asagi devam) olasiligi o kadar yuksek.''')

pdf.sub('Diger Onemli Feature\'lar')
pdf.txt('''Imbalance (HIGH): Q1 %35.2 -> Q5 %93.3 (monoton)
Vol_micro (HIGH): Q1 %47.9 -> Q5 %83.5 (monoton)
ATR_pctile (HIGH): Q1 %60.5 -> Q5 %76.6 (artis)

Imbalance (LOW): Q1 %88.2 -> Q5 %35.9 (ters monoton)
Vol_micro (LOW): Q1 %33.7 -> Q5 %87.1 (monoton)
ATR_pctile (LOW): Q1 %57.4 -> Q5 %78.9 (artis)''')

# === SAYFA 4: TOP PATTERNS ===
pdf.add_page()
pdf.section('4. EN GUCLU PATTERN\'LER')

pdf.sub('HIGH CONTINUATION - LONG Sinyalleri')
cols = ['#', 'Rate%', 'N', 'WF', 'Conditions']
widths = [8, 18, 15, 15, 114]
pdf.th(cols, widths)
pdf.tr(['1', '97.8', '270', '8/8', 'CVD_micro Q5 AND Imbalance Q5'], widths, bold=True)
pdf.tr(['2', '98.4', '243', '8/8', 'CVD_micro Q5 AND Vol_micro Q5'], widths, bold=True)
pdf.tr(['3', '98.2', '171', '8/8', 'Vol_micro Q5 AND Imbalance Q5'], widths)
pdf.tr(['4', '98.7', '150', '8/8', 'CVD_micro Q5 AND Vol_micro Q5 AND Imbalance Q5'], widths)
pdf.tr(['5', '99.2', '131', '8/8', 'CVD_micro Q5 AND CVD_macro Q5'], widths)
pdf.tr(['6', '92.2', '153', '8/8', 'CVD_micro Q4 AND Imbalance Q4'], widths)
pdf.tr(['7', '99.1', '107', '8/8', 'CVD_micro Q5 AND ATR_pctile Q5'], widths)

pdf.ln(3)
pdf.sub('LOW CONTINUATION - SHORT Sinyalleri')
pdf.th(cols, widths)
pdf.tr(['1', '99.6', '268', '8/8', 'CVD_micro Q1 AND Vol_micro Q5'], widths, bold=True)
pdf.tr(['2', '97.5', '239', '8/8', 'CVD_micro Q1 AND Imbalance Q1'], widths, bold=True)
pdf.tr(['3', '98.8', '170', '8/8', 'Vol_micro Q5 AND Imbalance Q1'], widths)
pdf.tr(['4', '99.3', '146', '8/8', 'CVD_micro Q1 AND Vol_micro Q5 AND Imbalance Q1'], widths)
pdf.tr(['5', '96.5', '115', '8/8', 'CVD_micro Q1 AND ATR_pctile Q5'], widths)
pdf.tr(['6', '90.8', '163', '8/8', 'CVD_micro Q2 AND Vol_micro Q4'], widths)
pdf.tr(['7', '100.0', '86', '7/7', 'CVD_micro Q1 AND Vol_macro Q4'], widths)

pdf.ln(3)
pdf.sub('HIGH REVERSAL - SHORT Sinyalleri')
pdf.th(cols, widths)
pdf.tr(['1', '74.3', '249', '8/8', 'CVD_micro Q1 AND Imbalance Q1'], widths, bold=True)
pdf.tr(['2', '75.2', '133', '8/8', 'Vol_micro Q1 AND Imbalance Q1'], widths)
pdf.tr(['3', '71.7', '159', '8/8', 'CVD_micro Q1 AND Vol_micro Q1'], widths)
pdf.tr(['4', '85.2', '54', '3/3', 'CVD_micro Q1 AND Imbalance Q1 AND ATR Q1'], widths)

pdf.ln(3)
pdf.sub('LOW REVERSAL - LONG Sinyalleri')
pdf.th(cols, widths)
pdf.tr(['1', '88.0', '142', '8/8', 'Vol_micro Q1 AND Imbalance Q5'], widths, bold=True)
pdf.tr(['2', '77.0', '217', '8/8', 'CVD_micro Q5 AND Imbalance Q5'], widths)
pdf.tr(['3', '75.8', '194', '8/8', 'CVD_micro Q5 AND Vol_micro Q1'], widths)
pdf.tr(['4', '89.3', '75', '7/7', 'CVD_micro Q5 AND Vol_micro Q1 AND Imbalance Q5'], widths)

# === SAYFA 5: FEATURE COMPARISON ===
pdf.add_page()
pdf.section('5. FEATURE KARSILASTIRMA')

pdf.sub('HIGH SWEEP: Continuation vs Reversal (Mann-Whitney U)')
cols = ['Feature', 'Cont Med', 'Rev Med', 'p-value', 'Sig']
widths = [45, 25, 25, 25, 15]
pdf.th(cols, widths)
pdf.tr(['CVD_micro', '+0.639', '-1.114', '0.000000', '***'], widths, bold=True)
pdf.tr(['Vol_micro', '+0.975', '-0.443', '0.000000', '***'], widths, bold=True)
pdf.tr(['Imbalance', '+0.023', '-0.033', '0.000000', '***'], widths, bold=True)
pdf.tr(['ATR_pctile', '+51.0', '+39.6', '0.000000', '***'], widths)
pdf.tr(['CVD_macro', '+0.355', '+0.027', '0.000000', '***'], widths)
pdf.tr(['Vol_macro', '+0.171', '-0.062', '0.001586', '***'], widths)
pdf.tr(['OI_change', '+12.19', '-2.26', '0.059914', ''], widths)

pdf.ln(3)
pdf.sub('LOW SWEEP: Continuation vs Reversal')
pdf.th(cols, widths)
pdf.tr(['CVD_micro', '-0.700', '+1.132', '0.000000', '***'], widths, bold=True)
pdf.tr(['Vol_micro', '+1.161', '-0.614', '0.000000', '***'], widths, bold=True)
pdf.tr(['Imbalance', '-0.040', '+0.014', '0.000000', '***'], widths, bold=True)
pdf.tr(['ATR_pctile', '+57.6', '+42.7', '0.000000', '***'], widths)
pdf.tr(['CVD_macro', '-0.482', '-0.093', '0.000000', '***'], widths)
pdf.tr(['Vol_macro', '+0.113', '-0.205', '0.005913', '***'], widths)
pdf.tr(['OI_change', '-22.64', '-18.65', '0.973561', ''], widths)

# === SAYFA 6: META PATTERN ===
pdf.add_page()
pdf.section('6. META-PATTERN OZETI')

pdf.sub('Continuation Sinyali (yonde devam)')
pdf.txt('''1H mum onceki mum'un high/low'unu sweep etti VE sweep yonunde kapatti.
Bu anlamli bir breakout. Eger ayni anda:
- CVD_micro ucta (Q4-Q5 long icin, Q1-Q2 short icin)
- Volume yuksek (Q4-Q5)
- Imbalance yone uyumlu
ise, %95-99 olasilikla yon devam ediyor. (N=150-270, WF 8/8)''')

pdf.sub('Reversal Sinyali (yon donusu)')
pdf.txt('''1H mum onceki mum'un high/low'unu sweep etti AMA ters yonde kapatti (mum rengi ters).
Bu fake breakout / stop hunt. Eger ayni anda:
- CVD_micro ters ucta
- Volume dusuk (Q1)
- Imbalance ters yonde
ise, %74-88 olasilikla yon donuyor. (N=142-249, WF 8/8)''')

pdf.sub('Dominant Feature: CVD_micro')
pdf.txt('''CVD_micro (12-bar / 1 saatlik kumulatif delta) en guclu ayirici:
- High sweep: Q1 %36 cont -> Q5 %95 cont (monoton)
- Low sweep: Q1 %95 cont -> Q5 %33 cont (ters monoton)

Tek basina bile cok guclu: CVD_micro Q5'te high sweep %95 continuation,
CVD_micro Q1'de low sweep %95 continuation.''')

pdf.sub('Strateji Notu')
pdf.txt('''Giris sinyali cok guclu (%95-99 dogruluk). Ancak strateji backtestlerinde
pozisyon yonetimi (cikis zamanlama, fee kontrolu) sorun yaratti.
Cozum olarak KC DCA/TP, trailing stop veya zaman bazli cikis test edilmeli.
Fee etkisini azaltmak icin maker order veya daha az trade gerekli.''')

pdf.sub('TradingView Indikatoru')
pdf.txt('1h_sweep_indicator.pine dosyasi masaustunde mevcut. Canli sinyalleri gosterir.')

out = 'C:/Users/emrehaskilic/Desktop/1h_candle_sweep_report.pdf'
pdf.output(out)
print(f'PDF kaydedildi: {out}')
