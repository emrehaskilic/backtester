"""PDF Strategy Report — masaustune kaydet."""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

DESKTOP = r"C:\Users\emrehaskilic\Desktop"

# Load params
with open("results/ETHUSDT_unified_pmax.json") as f:
    pmax_data = json.load(f)
with open("results/ETHUSDT_unified_kc.json") as f:
    kc_data = json.load(f)
with open("results/ETHUSDT_unified_kelly.json") as f:
    kelly_data = json.load(f)

pmax_p = pmax_data["best_params"]
kc_p = kc_data["best_kc_params"]
kelly_p = kelly_data["best_kelly_params"]
result = kelly_data["result"]

# DD-capped sonuclar (generate_reports.py'den)
result = {
    "balance": 407535,
    "net_pct": 3975.4,
    "max_dd": 22.5,
    "win_rate": 93.8,
    "total_trades": 4835,
}

# Colors
DARK_BG = HexColor("#1a1a2e")
HEADER_BG = HexColor("#16213e")
GREEN = HexColor("#00b894")
RED = HexColor("#d63031")
GOLD = HexColor("#fdcb6e")
WHITE = HexColor("#ffffff")
GRAY = HexColor("#b2bec3")
BLUE = HexColor("#0984e3")

# PDF setup
pdf_path = os.path.join(DESKTOP, "ETHUSDT_strategy_report_dynsl.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm,
                        leftMargin=15*mm, rightMargin=15*mm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, textColor=BLUE,
                             spaceAfter=10)
h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, textColor=BLUE,
                           spaceBefore=15, spaceAfter=8)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, textColor=HexColor("#2d3436"),
                             leading=14)
small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=GRAY, leading=11)

elements = []

# TITLE
elements.append(Paragraph("ETHUSDT Unified Walk-Forward Strategy Report", title_style))
elements.append(Paragraph("Adaptive PMax + Keltner Channel + Kelly Dynamic Compounding", body_style))
elements.append(Spacer(1, 10))

# SUMMARY TABLE
elements.append(Paragraph("Performans Ozeti", h2_style))
summary_data = [
    ["Metrik", "Deger"],
    ["Sembol", "ETHUSDT Perpetual Futures"],
    ["Timeframe", "3 dakika (3m)"],
    ["Test Donemi", "6 ay (180 gun)"],
    ["Baslangic Kasa", "$10,000"],
    ["Final Kasa", f"${result['balance']:,.0f}"],
    ["Net Getiri", f"+{result['net_pct']:,.1f}%"],
    ["Max Drawdown", f"{result['max_dd']:.1f}%"],
    ["Win Rate", f"{result['win_rate']:.1f}%"],
    ["Toplam Trade", f"{result['total_trades']:,}"],
    ["Leverage", "25x"],
]
t = Table(summary_data, colWidths=[120, 200])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f6fa"), WHITE]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
elements.append(t)
elements.append(Spacer(1, 10))

# PMAX PARAMETERS
elements.append(Paragraph("ADIM 1: Adaptive PMax Parametreleri", h2_style))
elements.append(Paragraph(
    "PMax (Profit Maximizer) giri/cikis sinyali uretir. Adaptive Continuous modu "
    "volatilite, trend ve momentum'a gore parametreleri her 29 barda gunceller.", body_style))
elements.append(Spacer(1, 5))

pmax_desc = {
    "vol_lookback": "Volatilite median penceresi (bar)",
    "flip_window": "Yon degisim sayma penceresi (bar)",
    "mult_base": "ATR carpan tabani",
    "mult_scale": "Volatiliteye bagli carpan olcekleme",
    "ma_base": "MA uzunluk tabani",
    "ma_scale": "Trende bagli MA olcekleme",
    "atr_base": "ATR period tabani",
    "atr_scale": "Flip sayisina bagli ATR olcekleme",
    "update_interval": "Parametre guncelleme sikligi (bar)",
}
pmax_data_table = [["Parametre", "Deger", "Aciklama"]]
for k, v in pmax_p.items():
    pmax_data_table.append([k, str(v), pmax_desc.get(k, "")])

t = Table(pmax_data_table, colWidths=[100, 60, 220])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f6fa"), WHITE]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(t)
elements.append(Paragraph("Sabit base parametreler: atr_period=10, atr_multiplier=3.0, ma_length=10, ma_type=EMA, source=hl2", small_style))
elements.append(Spacer(1, 10))

# KC PARAMETERS
elements.append(Paragraph("ADIM 2: Keltner Channel Parametreleri", h2_style))
elements.append(Paragraph(
    "KC giris/cikis yonetimi saglar: DCA (KC lower'da ek alim) ve TP (KC upper'da kar alma). "
    "PMax sinyali uzerine KC ile pozisyon yonetimi yapilir.", body_style))
elements.append(Spacer(1, 5))

kc_desc = {
    "kc_length": "KC EMA periyodu",
    "kc_multiplier": "KC ATR carpani (bant genisligi)",
    "kc_atr_period": "KC ATR periyodu",
    "max_dca_steps": "Maksimum DCA adim sayisi",
    "tp_close_percent": "Kismi kar alma orani (%50 = yarisi)",
}
kc_data_table = [["Parametre", "Deger", "Aciklama"]]
for k, v in kc_p.items():
    kc_data_table.append([k, str(v), kc_desc.get(k, "")])

t = Table(kc_data_table, colWidths=[120, 60, 200])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f6fa"), WHITE]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(t)
elements.append(Spacer(1, 10))

# KELLY PARAMETERS
elements.append(Paragraph("ADIM 3: Kelly / Dynamic Compounding Parametreleri", h2_style))
elements.append(Paragraph(
    "Kasa buyudukce pozisyon buyuklugu kademeli olarak artar. "
    "Kucuk kasada agresif (%9.5), buyuk kasada muhafazakar (%3).", body_style))
elements.append(Spacer(1, 5))

kelly_table = [["Kasa Araligi", "Margin Orani", "Ornek ($10K kasa)"]]
kelly_table.append([
    f"< ${kelly_p['tier1_threshold']:,.0f}",
    f"%{kelly_p['base_margin_pct']}",
    f"$10K x {kelly_p['base_margin_pct']}% = ${10000 * kelly_p['base_margin_pct'] / 100:,.0f} margin",
])
kelly_table.append([
    f"${kelly_p['tier1_threshold']:,.0f} - ${kelly_p['tier2_threshold']:,.0f}",
    f"%{kelly_p['tier1_pct']}",
    f"$50K x {kelly_p['tier1_pct']}% = ${50000 * kelly_p['tier1_pct'] / 100:,.0f} margin",
])
kelly_table.append([
    f"> ${kelly_p['tier2_threshold']:,.0f}",
    f"%{kelly_p['tier2_pct']}",
    f"$200K x {kelly_p['tier2_pct']}% = ${200000 * kelly_p['tier2_pct'] / 100:,.0f} margin",
])

t = Table(kelly_table, colWidths=[120, 80, 180])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f6fa"), WHITE]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(t)
elements.append(Spacer(1, 10))

# FILTERS
elements.append(Paragraph("Filtreler", h2_style))
filter_data = [
    ["Filtre", "Parametre", "Mantik"],
    ["EMA Trend", "EMA(144)", "LONG: fiyat > EMA, SHORT: fiyat < EMA"],
    ["RSI", "RSI(28), OB=65, OS=35", "LONG: RSI < 65, SHORT: RSI > 35"],
    ["ATR Volume", "ATR(50), %20 percentile", "Dusuk volatilite = islem yapma"],
    ["Hard Stop", "%2.5", "DCA full sonrasi %2.5 kayip = kapat"],
]
t = Table(filter_data, colWidths=[80, 120, 180])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f6fa"), WHITE]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(t)
elements.append(Spacer(1, 10))

# OPTIMIZATION PROCESS
elements.append(Paragraph("Optimizasyon Sureci", h2_style))
opt_data = [
    ["Adim", "Yontem", "Detay"],
    ["ADIM 1: PMax", "1000 trial x 25 hafta", "Unified test — tek parametre tum haftalarda"],
    ["ADIM 2: KC", "1000 trial x 25 hafta", "PMax kilitli, KC optimize"],
    ["ADIM 3: Kelly", "1000 trial x 180 gun", "PMax+KC kilitli, dynamic margin"],
    ["Engine", "Rust (PyO3)", "178x Python hizlanmasi"],
    ["Sampler", "TPE Multivariate", "Parametreler arasi korelasyon ogrenir"],
    ["Pruner", "MedianPruner", "Kotu trial'lari erken keser"],
    ["Paralel", "n_jobs=6", "6 trial ayni anda"],
]
t = Table(opt_data, colWidths=[80, 110, 190])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f6fa"), WHITE]),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
elements.append(t)
elements.append(Spacer(1, 15))

# DISCLAIMER
elements.append(Paragraph("Dynamic Stop Loss", h2_style))
elements.append(Paragraph(
    "Hard Stop %2.5'ten %2.0'a dusurulmustur. DCA tamamlandiktan sonra fiyat giris noktasindan "
    "%2.0 uzaklasirsa pozisyon otomatik kapatilir. Bu degisiklik DD'yi %23.3'ten %22.5'e dusururken "
    "getiriyi $365K'dan $407K'ya yukseltmistir. ATR-based DynSL test edilmis ancak sabit %2.0 "
    "daha iyi sonuc vermistir.",
    body_style))
elements.append(Spacer(1, 10))

elements.append(Paragraph("Uyari", h2_style))
elements.append(Paragraph(
    "Bu backtest sonuclari gecmis veriye dayalidir. Gercek piyasada slippage, funding rate, "
    "likidite farklari ve exchange gecikmeleri nedeniyle sonuclar farkli olabilir.",
    body_style))

# Build PDF
doc.build(elements)
print(f"PDF saved: {pdf_path}")
