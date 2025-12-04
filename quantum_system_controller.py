"""System orchestrator connecting BioImmortalityAI with MLPerf-like metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mlperf_brion_quantum_integration import MLPerfBrionQuantumIntegration
from quantum_research_engine import BioImmortalityAI, CONFIG


class QuantumLongevitySystem:
    """
    High-level orchestrator connecting:
    - BioImmortalityAI (longevity / age reversal reasoning)
    - MLPerf v5.1-inspired training benchmarks (system performance + training data)
    """

    def __init__(
        self,
        output_dir: str = "outputs/mlperf",
        quantum_hardware: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.mlperf = MLPerfBrionQuantumIntegration(
            mlperf_root=self.output_dir / "mlperf_training",
            output_dir=self.output_dir,
            quantum_hardware=quantum_hardware,
        )
        self.quantum_hardware = quantum_hardware
        self.bio_ai = BioImmortalityAI(CONFIG)
        self.bio_ai.load_domain_knowledge()

        self.training_env_metrics: Dict[str, Any] = {}

    def run_mlperf_and_collect(self, dry_run: bool = True) -> Dict[str, Any]:
        """Run the suite of benchmarks (or a dry-run) and store summary metrics."""
        results = self.mlperf.run_all_benchmarks(dry_run=dry_run)
        summary: Dict[str, Any] = {}
        for name, res in results.items():
            summary[name] = {
                "benchmark": res.benchmark_name,
                "time_to_train": res.time_to_train,
                "final_accuracy": res.final_accuracy,
                "hardware_config": res.hardware_config,
                "quantum_enhanced": res.quantum_enhanced,
            }
        self.training_env_metrics = summary
        return summary

    def register_env_metrics_with_bio_ai(self):
        """Allow the core AI to reason about system capabilities from benchmarks."""
        self.bio_ai.memory.append(
            {
                "type": "system_benchmark",
                "source": "MLPerf Training v5.1",
                "metrics": self.training_env_metrics,
                "quantum_hardware": self.quantum_hardware,
            }
        )
