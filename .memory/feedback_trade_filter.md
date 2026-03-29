---
name: Trade count filter rule
description: WF optimizasyonda train set'te minimum trade sayısı yüksek tutulmalı (>=500), düşük trade'li parametreler reddedilmeli
type: feedback
---

Train set'te minimum trade filtresi yüksek tutulmalı (en az 500 trade). Düşük trade sayılı parametreler (3-5 trade) istatistiksel olarak anlamsız — şansa dayalı sonuçlar üretir.

**Why:** Kullanıcı "total trade < 500 olanları ele" dedi. Ben yanlış yorumlayıp filtreyi 3'ten 1'e düşürdüm. Bu, 1 trade ile WR %100 gösteren anlamsız sonuçlar üretti. Kullanıcı haklı olarak kızdı.

**How to apply:**
- WF optimizasyonda `if tr["total_trades"] < 500: return -999` kullan (train set)
- Test set'te daha düşük filtre olabilir (7 günlük pencerede az trade normal)
- Aceleye getirmeden kullanıcının talebini doğru anla. "Ele" = "elemine et" = reddet
- Hızlı fix yapmak yerine sorunun kökenine in
