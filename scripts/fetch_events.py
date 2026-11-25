import subprocess
from pathlib import Path

REPO = Path("open-data")
DATA = REPO / "data"

# Every match with 360 data has a file named <match_id>.json in this folder
match_ids = sorted(p.stem for p in (DATA / "three-sixty").glob("*.json"))
print(len(match_ids), "matches with 360 data")

# Files git should keep in the working folder
keep = [
    "/data/competitions.json",
    "/data/matches/",
    "/data/three-sixty/",
]
keep += [f"/data/events/{match_id}.json" for match_id in match_ids]


subprocess.run(
    ["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
    input="\n".join(keep) + "\n",
    text=True,
    cwd=REPO,
    check=True,
)


n_events = len(list((DATA / "events").glob("*.json")))
print(n_events, "event files")
assert n_events == len(match_ids), "events and 360 file counts differ"