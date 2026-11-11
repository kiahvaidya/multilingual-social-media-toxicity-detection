import urllib.request
import json

phrases = [
    "HEY MAN U ARE THE FUCKING GOAT",
    "you are a fucking wizard",
    "this song is fucking fire",
    "she is bloody brilliant",
    "you are a fucking clown",
    "what the fuck is going on",
    "bhai tu katai bawaal hai",
]

for p in phrases:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/analyze/text",
        data=json.dumps({"text": p}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    d = json.loads(res.read().decode("utf-8"))
    icon = d["badge_icon"]
    risk = d["risk_level"]
    is_tox = d["is_toxic"]
    expl = d["explanation"]
    print(f"[{icon} {risk}] (Toxic={is_tox}) {p}\n   -> {expl}\n")

