import json
from pathlib import Path

# list of package directories to search for report.json
PKGS = ["multilayer_perceptron"]
REPORT_FILENAME = "report.json"
MERGED_REPORT_FILENAME = "merged_report.json"


all_test = []
summary = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "error": 0,
    "xfailed": 0,
    "xpassed": 0,
    "total": 0,
}

for pkg in PKGS:
    pkg_path = Path(pkg)
    report_path = pkg_path / REPORT_FILENAME
    if not report_path.exists():
        print(f"Report file not found in {pkg_path}")
        continue

    with open(report_path) as f:
        data = json.load(f)
        # Collet tests
        tests = data.get("tests", [])
        all_test.extend(tests)
        # Merge summary
        s = data.get("summary", {})
        for k in summary:
            summary[k] += s.get(k, 0)

# Compose merged report
merged_report = {
    "tests": all_test,
    "summary": summary,
    "collectors": PKGS,
}

# Write merged report to file
with open(MERGED_REPORT_FILENAME, "w") as f:
    json.dump(merged_report, f, indent=2)

print(f"Merged {len(all_test)} tests from {len(PKGS)} packages into {MERGED_REPORT_FILENAME}")
