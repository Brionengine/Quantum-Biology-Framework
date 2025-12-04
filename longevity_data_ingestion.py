import csv
import json
from typing import List, Dict


class LongevityDataLoader:
    def __init__(self, base_path: str = "data/longevity"):
        self.base_path = base_path

    def load_genage(self) -> List[Dict]:
        """Load GenAge gene data (after you export/convert to CSV/JSON)."""
        path = f"{self.base_path}/genage_human.csv"
        genes = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                genes.append(row)
        return genes

    def load_drugage(self) -> List[Dict]:
        """Load DrugAge compound data."""
        path = f"{self.base_path}/drugage_compounds.csv"
        compounds = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                compounds.append(row)
        return compounds

    def load_aging_trials(self) -> List[Dict]:
        """Load curated list of aging-targeted clinical trials (from agingdb / ClinicalTrials.gov exports)."""
        path = f"{self.base_path}/aging_trials.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

