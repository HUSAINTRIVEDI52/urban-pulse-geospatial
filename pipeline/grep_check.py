import time
import re
from pathlib import Path

def grep_files():
    start = time.time()
    targets = [
        "417.82", "431.05", "458.74", "478.11", 
        "416.82", "453.90", "477.15", "61.18", "60.29"
    ]
    files = [
        Path("README.md"),
        Path("docs/report/index.html"),
        Path("web/index.html"),
        Path("web/data/ahmedabad/stats.json"),
        Path("web/data/pune/stats.json")
    ]
    
    pattern = re.compile(r"|".join(re.escape(t) for t in targets))
    matches = []
    
    for fpath in files:
        if not fpath.exists():
            print(f"File not found: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                m = pattern.search(line)
                if m:
                    matches.append((str(fpath), line_no, m.group(0), line.strip()))
                    
    print(f"Total matches found: {len(matches)}")
    for fpath, line_no, matched_txt, line_str in matches:
        print(f"{fpath}:{line_no} [Matched {matched_txt}]: {line_str}")
    
    elapsed = time.time() - start
    print(f"Grep completed in {elapsed:.4f}s")

if __name__ == "__main__":
    grep_files()
