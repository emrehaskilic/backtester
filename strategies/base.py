"""Strategy base interface — tüm stratejiler bu ABC'yi implement eder."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class RoundConfig:
    """Tek bir optimizasyon round'unun tanımı."""
    round_num: int
    name: str          # "PMAX KESIF", "KC OPTIMIZE" vb.
    n_trials: int      # Default trial sayısı
    description: str = ""
    skippable: bool = False  # Bu round atlanabilir mi?


@dataclass
class TrialResult:
    """Tek bir trial'ın sonucu."""
    tid: int
    score: float
    is_net: float       # In-sample net %
    oos_net: float      # Out-of-sample net %
    wr: float           # Win rate %
    dd: float           # Max drawdown %
    trades: int
    params: dict = field(default_factory=dict)
    equity_curve: list = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """Net/DD oranı — auto-selection için birincil metrik."""
        if self.dd <= 0:
            return 0.0
        return round(self.oos_net / self.dd, 2)


@dataclass
class RoundResult:
    """Bir round'un tüm sonuçları."""
    round_num: int
    round_name: str
    trials: list[TrialResult]
    selected: TrialResult
    selected_params: dict


@dataclass
class PipelineResult:
    """Tam pipeline sonucu (tüm round'lar)."""
    symbol: str
    strategy: str
    timeframe: str
    rounds: list[RoundResult]
    final_params: dict
    final_metrics: dict
    equity_curve: list = field(default_factory=list)


class StrategyOptimizer(ABC):
    """Tüm stratejilerin implement etmesi gereken abstract base class.

    Her strateji:
    1. Round tanımlarını döndürür (get_rounds)
    2. Veri tipini belirtir (get_data_type)
    3. Veriyi yükler ve hazırlar (prepare_data)
    4. Her round için objective function oluşturur (create_objective)
    5. Round sonuçlarından parametre çıkarır (extract_params)
    """

    def __init__(self, symbol: str, timeframe: str = "3m", days: int = 180,
                 leverage: int = 25, event_callback: Optional[Callable] = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
        self.leverage = leverage
        self.event_callback = event_callback  # WebSocket broadcast için

        # Pipeline boyunca korunan state
        self.selected_params: dict[str, dict] = {}   # round_key -> params
        self.selected_metrics: dict[str, dict] = {}   # round_key -> metrics
        self.data_cache: dict[str, Any] = {}          # pre-computed data

    def emit(self, event_type: str, data: dict):
        """WebSocket event gönder (varsa)."""
        if self.event_callback:
            self.event_callback(event_type, data)

    @abstractmethod
    def get_rounds(self) -> list[RoundConfig]:
        """Bu strateji için optimizasyon round'larının listesi."""
        ...

    @abstractmethod
    def get_data_type(self) -> str:
        """Gereken veri tipi: 'klines' veya 'aggtrades'."""
        ...

    @abstractmethod
    def prepare_data(self, data_path: str) -> dict:
        """Veriyi yükle, IS/OOS böl, pre-compute yap.

        Returns:
            dict with keys like 'df_is', 'df_oos', 'src_is', 'src_oos',
            'indicators_is', 'indicators_oos', etc.
        """
        ...

    @abstractmethod
    def create_objective(self, round_num: int, prepared_data: dict) -> Callable:
        """Belirli bir round için Optuna objective function oluştur.

        Args:
            round_num: Hangi round (1-based)
            prepared_data: prepare_data() çıktısı

        Returns:
            Callable(trial) -> float (Optuna objective)
        """
        ...

    @abstractmethod
    def extract_params(self, round_num: int, trial_params: dict) -> dict:
        """Trial parametrelerinden strateji parametrelerini çıkar.

        Her round farklı parametre setleri optimize edebilir.
        Bu method, seçilen trial'ın parametrelerini bir sonraki round'a
        aktarılacak şekilde normalize eder.
        """
        ...

    @abstractmethod
    def run_final_backtest(self, prepared_data: dict, params: dict) -> dict:
        """Final parametrelerle tam backtest çalıştır.

        Returns:
            dict with 'net_pct', 'dd', 'wr', 'trades', 'equity_curve', 'trades_list'
        """
        ...

    def get_available_timeframes(self) -> list[str]:
        """Bu strateji için desteklenen timeframe'ler."""
        return ["3m"]  # Default, override edilebilir

    def get_default_trial_counts(self) -> dict[int, int]:
        """Round bazlı default trial sayıları."""
        return {r.round_num: r.n_trials for r in self.get_rounds()}
