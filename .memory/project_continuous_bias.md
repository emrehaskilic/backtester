---
name: project_continuous_bias
description: Kullanicinin surekli yon gosteren bias sistemi ihtiyaci — sweep sinyalleri sadece %6 kapsiyor, geri kalan %94 icin cozum bulunamadi
type: project
---

## Surekli Bias Gostericisi Ihtiyaci

Kullanici surekli (her an) yon gosteren bir bias sistemi istiyor. Bu bias'in altinda KC ile islem yapilacak.

### Mevcut Durum (2026-03-26)
- **Sweep 4H + 3D grid + PMAX 3m catisma filtresi**: %84.3 WR, 886 trade/5yil, ama zamanin sadece %6'sinda aktif
- **PMAX 3m tek basina**: %48 WR, coin flip, 25x kaldiracla zarar
- **KAMA**: %48 WR, ayni sekilde yetersiz
- **Son sinyali tasima**: Edge'i olduruyor (%57→%52)
- **Sweep + PMAX doldurucu**: PMAX bolumleri zarar, sweep bolumleri kar → net faydasiz

### Denenmis ve Basarisiz Olan Yontemler
1. Son sinyali tasima (carry) — edge kayboluyor
2. PMAX 5m bosluk doldurucu — coin flip
3. KAMA bosluk doldurucu — coin flip
4. PMAX 3m surekli pozisyon (25x) — komisyon + whipsaw kasayi yiyor
5. Decay (zamanla zayiflayan) tasima — iyilestirmiyor

### Cozulmesi Gereken Problem
Zamanin %94'unde yon bilgisi olmayan boşluklari dolduracak, en az %55+ WR ile calisacak, 25x kaldiracli surekli pozisyonu kaldiracak bir sistem.

**Why:** Kullanici KC entry sistemi ile calisacak, KC'nin dogru calismasi icin surekli bir bias/yon gostericisi sart.

**How to apply:** Bir sonraki session'da bu problemi cozmek oncelikli. Farkli yaklasimlar denenebilir: market regime detection, price structure bazli trend, multi-asset korelasyon, veya tamamen farkli bir indicator seti.
