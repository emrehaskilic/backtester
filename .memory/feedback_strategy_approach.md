---
name: feedback_strategy_approach
description: Strateji gelistirmede parametre optimizasyonu vs domain bilgisi - kullanicinin yaklasimi cok daha etkili
type: feedback
---

Parametre optimizasyonu (Optuna, WF) ile %97+ WR bulunan sonuclar sabit parametrelerle -%134 cikti. Kullanicinin mum kapanisi bazli sweep tanimlari ise gercek edge verdi.

**Why:** Domain bilgisi (mum kapanisi + mum rengi + sweep) piyasanin gercek mekanigini yakaladı. Parametre optimizasyonu gecmise fit etti.

**How to apply:** Her zaman once kullanicinin trading bilgisini sor, sonra koda dok. Makine ogrenmesi / optimizasyon kullanicinin hipotezini TEST etmek icin kullan, hipotez URETMEK icin degil.
