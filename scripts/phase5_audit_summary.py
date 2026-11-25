"""Summarise the filled audit sheet (results/phase5/audit_sheet.csv) against audit_key.csv.
Prints: checker miss rate (unfaithful among the checker-passed rows), real-flag rate (unfaithful among the checker-flagged
rows) and mean clarity. Optional argument: path to the filled sheet (default: results/phase5/audit_sheet.csv)."""
import csv, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results/phase5"


def main():
    sheet = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "audit_sheet.csv"
    key = {r["id"]: int(r["checker_passed"]) for r in csv.DictReader(open(OUT / "audit_key.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    lab = [(r["id"], r["faithful_YN"].strip().upper()[:1], r["clarity_1to5"].strip()) for r in rows]
    missing = [i for i, y, _ in lab if y not in ("Y", "N")]
    if missing:
        print(f"{len(missing)} rows without a Y/N label, e.g. {missing[:5]}"); sys.exit(1)

    def share_unfaithful(passed_flag):
        sub = [y for i, y, _ in lab if key[i] == passed_flag]
        return (sum(y == "N" for y in sub), len(sub))

    k, n = share_unfaithful(1)
    print(f"checker miss rate (unfaithful among checker-passed): {k}/{n} = {k / n:.1%}")
    k2, n2 = share_unfaithful(0)
    print(f"real-flag rate (unfaithful among checker-flagged):   {k2}/{n2} = {k2 / n2:.1%}")
    cl = [float(c) for _, _, c in lab if c]
    print(f"mean clarity: {sum(cl) / len(cl):.2f} (n={len(cl)})" if cl else "no clarity scores")


if __name__ == "__main__":
    main()
