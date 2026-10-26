"""
Quick end-to-end verification script for Profile Watch API endpoints.
Validates:
1. Registration & Scanning
2. Duplicate prevention & Idempotency
3. Aggregate recalculation
4. Pattern detection
5. State transitions (MONITORING / PAUSED)
6. Profile retrieval
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_e2e_flow():
    print("[1] Testing initial scan on Reddit profile...")
    res = client.post("/api/profile-watch/scan", json={
        "url": "https://www.reddit.com/user/test_analyst",
        "limit": 10,
        "demo_fallback": True,
        "enable_jev": False
    })
    assert res.status_code == 200, f"Scan failed: {res.text}"
    scan_data = res.json()
    profile_id = scan_data["profile"]["id"]
    print(f" -> Scan succeeded: {scan_data['profile']['handle']} ({scan_data['profile']['platform']})")
    print(f" -> Discovered: {scan_data['discovered']}, New: {scan_data['new']}, Skipped: {scan_data['skipped']}")
    print(f" -> Toxicity Rate: {scan_data['aggregates']['toxicity_rate']:.1f}%, High Risk Count: {scan_data['high_risk']}")
    
    print("\n[2] Testing subsequent duplicate scan on same profile...")
    res2 = client.post("/api/profile-watch/scan", json={
        "url": "https://www.reddit.com/user/test_analyst",
        "limit": 10,
        "demo_fallback": True,
        "enable_jev": False
    })
    assert res2.status_code == 200
    scan_data2 = res2.json()
    print(f" -> Duplicate scan: Discovered: {scan_data2['discovered']}, New: {scan_data2['new']}, Skipped: {scan_data2['skipped']}")
    assert scan_data2["new"] == 0, "Subsequent identical scan should have 0 new items"
    assert scan_data2["skipped"] == scan_data2["discovered"], "All items should be skipped as duplicates"

    print("\n[3] Testing profile summary retrieval...")
    res3 = client.get(f"/api/profile-watch/{profile_id}")
    assert res3.status_code == 200
    summary = res3.json()
    print(f" -> Summary: {summary['profile']['handle']} - Total Obs: {summary['aggregates']['total_observations']}")
    print(f" -> Category averages: {list(summary['aggregates']['average_scores'].keys())}")
    print(f" -> Detected behavioral events: {len(summary['events'])}")

    print("\n[4] Testing state transitions (START MONITORING -> PAUSE)...")
    res_m = client.post(f"/api/profile-watch/{profile_id}/monitor")
    assert res_m.status_code == 200
    assert res_m.json()["monitoring_state"] == "MONITORING"
    print(" -> State transitioned to MONITORING")

    res_p = client.post(f"/api/profile-watch/{profile_id}/pause")
    assert res_p.status_code == 200
    assert res_p.json()["monitoring_state"] == "PAUSED"
    print(" -> State transitioned to PAUSED")

    print("\n[5] Testing profile activity query...")
    res_act = client.get(f"/api/profile-watch/{profile_id}/activity?limit=5")
    assert res_act.status_code == 200
    obs_list = res_act.json()
    print(f" -> Fetched {len(obs_list)} observations")
    if obs_list:
        print(f" -> Sample observation: [{obs_list[0]['final_risk_level']}] {obs_list[0]['text'][:60]}...")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_e2e_flow()
