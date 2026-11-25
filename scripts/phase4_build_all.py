#!/usr/bin/env python
"""Phase 4: build everything the API needs, in order (about 3 minutes on CPU).

    uv run python scripts/phase4_build_all.py

1. phase4_select_xg.py   selection table + reliability plot (val only)        -> results/phase4/
2. phase4_build_xg.py    retrain ANN x5 seeds, verify vs Phase 2, score shots -> models/xg_ann/, data/processed/xg_scores.csv
3. phase4_build_index.py embed 10,233 docs, val sanity check + threshold      -> models/index/, results/phase4/retrieval_check.json
4. phase4_select_scope.py scope / low-confidence rule (val queries only)      -> results/phase4/scope_rule.json (needs 2 and 3)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("phase4_select_xg.py", "phase4_build_xg.py", "phase4_build_index.py", "phase4_select_scope.py"):
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True)
