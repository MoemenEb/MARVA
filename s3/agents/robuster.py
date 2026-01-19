from collections import Counter
from typing import List, Dict


class MajorityArbitrator:

    @staticmethod
    def arbitrate(results: List[Dict]) -> Dict:

        decisions = [r["decision"] for r in results]
        counts = Counter(decisions)

        decision, freq = counts.most_common(1)[0]

        final_decision = decision       

        # --- Merge issues (flatten, not nest) ---
        aggregated_issues = []
        for r in results:
            aggregated_issues.append(r.get("issues", []))

        return {
            "final_decision": final_decision,
            "runs": results,
            "issues": aggregated_issues,
        }
