import re, difflib

_WAKE_ALIASES = {
    "jarvis", "jarvas", "jarvus", "jarviz", "jarviss", "jarvis's", "jarvez",
    "jarvys", "jarwis", "jarvice", "jervis", "jerviz", "jervas", "javis",
    "chavez", "charvis", "darvis", "travis", "garvis", "harvest", "harvis",
    "jardis", "java", "javas", "tardis"
}

def _word_is_wake(w: str) -> bool:
    raw = w.strip()
    if "جارف" in raw:
        return True
    latin = re.sub(r"[^a-z]", "", raw.lower())
    if not latin:
        return False
    if latin in _WAKE_ALIASES or latin.startswith(("jarv", "jerv")):
        return True
    skel = re.sub(r"[aeiouy]", "", latin)
    return skel.startswith("j") and difflib.SequenceMatcher(None, skel, "jrvs").ratio() >= 0.7

print("jarvis:", _word_is_wake("jarvis"))
print("J.A.R.V.I.S.:", _word_is_wake("J.A.R.V.I.S."))
