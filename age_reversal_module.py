"""Age reversal and rejuvenation research structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RejuvenationIntervention:
    """Representation of a single rejuvenation intervention."""

    name: str
    modality: str  # "drug", "gene_therapy", "lifestyle", etc.
    targets: List[str]  # hallmarks / pathways
    evidence_level: str  # "preclinical", "early_clinical", ...
    risk_flags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class RejuvenationHypothesis:
    """Structured rejuvenation hypothesis composed of interventions."""

    id: str
    interventions: List[RejuvenationIntervention]
    rationale: str
    predicted_benefits: List[str] = field(default_factory=list)
    predicted_risks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgeReversalResearchEngine:
    """Reason about rejuvenation hypotheses and integrate with a knowledge graph."""

    def __init__(self, knowledge_graph: Optional[Any] = None):
        self.knowledge_graph = knowledge_graph
        self.hypotheses: Dict[str, RejuvenationHypothesis] = {}

    def register_hypothesis(self, hypothesis: RejuvenationHypothesis):
        """Store a rejuvenation hypothesis for further evaluation."""

        self.hypotheses[hypothesis.id] = hypothesis

    def create_hypothesis_from_spec(
        self,
        spec: Dict[str, Any]
    ) -> RejuvenationHypothesis:
        """
        Build and register a RejuvenationHypothesis from a structured spec.

        Expected spec schema:
        {
          "id": "hypo_001",
          "rationale": "...",
          "predicted_benefits": [...],
          "predicted_risks": [...],
          "metadata": {...},
          "interventions": [
            {
              "name": "...",
              "modality": "drug|gene_therapy|lifestyle|cell_therapy|other",
              "targets": ["epigenetic alterations", "cellular senescence"],
              "evidence_level": "preclinical|early_clinical|observational|theoretical",
              "risk_flags": ["oncogenic_risk", "unknown_long_term_safety"],
              "notes": "..."
            },
            ...
          ]
        }
        """
        interventions = []
        for iv in spec.get("interventions", []):
            interventions.append(
                RejuvenationIntervention(
                    name=iv.get("name", "unknown"),
                    modality=iv.get("modality", "unknown"),
                    targets=iv.get("targets", []),
                    evidence_level=iv.get("evidence_level", "unknown"),
                    risk_flags=iv.get("risk_flags", []),
                    notes=iv.get("notes", ""),
                )
            )

        hypothesis = RejuvenationHypothesis(
            id=spec.get("id", f"h_{len(self.hypotheses) + 1}"),
            interventions=interventions,
            rationale=spec.get("rationale", ""),
            predicted_benefits=spec.get("predicted_benefits", []),
            predicted_risks=spec.get("predicted_risks", []),
            metadata=spec.get("metadata", {}),
        )

        self.register_hypothesis(hypothesis)
        return hypothesis

    def simple_consistency_check(self, hypothesis_id: str) -> Dict[str, Any]:
        """Perform basic validation on targets and risk signals."""

        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return {"status": "not_found", "message": "Hypothesis ID not registered"}

        hallmarks_targeted = set()
        risk_flags: List[str] = []

        for intervention in hypothesis.interventions:
            hallmarks_targeted.update(intervention.targets)
            risk_flags.extend(intervention.risk_flags)

            if self.knowledge_graph:
                for target in intervention.targets:
                    # If target is in graph, we consider it grounded
                    if target not in self.knowledge_graph.graph:
                        risk_flags.append(f"target_unlinked:{target}")

        return {
            "status": "ok",
            "hypothesis_id": hypothesis_id,
            "hallmarks_targeted": sorted(hallmarks_targeted),
            "risk_flags": sorted(set(risk_flags)),
            "n_interventions": len(hypothesis.interventions),
        }

    def score_hypothesis_for_optimization(
        self,
        hypothesis_id: str,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Produce a scalar score for optimization.
        NOTE: This is a research meta-score, not a prediction of clinical outcome.

        Components:
        - hallmarks_coverage: coverage over hallmarks of aging
        - penalty_risk_flags: penalty for many / severe risk flags
        - complexity_penalty: penalize too many interventions
        """
        if weights is None:
            weights = {
                "hallmarks_coverage": 1.0,
                "risk_penalty": 1.0,
                "complexity_penalty": 0.2,
            }

        summary = self.simple_consistency_check(hypothesis_id)
        if summary.get("status") != "ok":
            return -1e6

        n_interventions = summary["n_interventions"]
        n_hallmarks = len(summary["hallmarks_targeted"])
        n_risks = len(summary["risk_flags"])

        score = 0.0
        score += weights["hallmarks_coverage"] * float(n_hallmarks)
        score -= weights["risk_penalty"] * float(n_risks)
        score -= weights["complexity_penalty"] * float(max(0, n_interventions - 3))

        return score
