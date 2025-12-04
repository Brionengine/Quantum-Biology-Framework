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

    def simple_consistency_check(self, hypothesis_id: str) -> Dict[str, Any]:
        """Perform basic validation on targets and risk signals."""

        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            return {"status": "not_found", "message": "Hypothesis ID not registered"}

        covered_targets = set()
        risk_flags = []

        for intervention in hypothesis.interventions:
            covered_targets.update(intervention.targets)
            risk_flags.extend(intervention.risk_flags)

            if self.knowledge_graph:
                for target in intervention.targets:
                    # If target is in graph, we consider it grounded
                    if target not in self.knowledge_graph.graph:
                        risk_flags.append(f"target_unlinked:{target}")

        return {
            "status": "ok",
            "hypothesis_id": hypothesis_id,
            "covered_targets": sorted(covered_targets),
            "risk_flags": sorted(set(risk_flags)),
            "intervention_count": len(hypothesis.interventions),
        }
