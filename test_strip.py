import re

_WAKE_ALIASES = {
    "jarvis", "jarvas", "jarvus", "jarviz", "jarviss", "jarvis's", "jarvez",
    "jarvys", "jarwis", "jarvice", "jervis", "jerviz", "jervas", "javis",
    "chavez", "charvis", "darvis", "travis", "garvis", "harvest", "harvis",
    "jardis", "java", "javas", "tardis"
}

def _word_is_wake(w: str) -> bool:
    import difflib
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

def _strip_wake_words(text: str) -> str:
    words = text.split()
    for i, w in enumerate(words[:6]):
        if _word_is_wake(w):
            after = words[i + 1:]
            if not after and i > 0:
                return " ".join(words[:i]).strip().lstrip(" ,.:;-!?")
            return " ".join(after).strip().lstrip(" ,.:;-!?")
    i = 0
    while i < len(words) and re.sub(r"[^a-z]", "", words[i].lower()) in ("hey", "ok", "okay", "hi", "hello"):
        i += 1
    return " ".join(words[i:]).strip().lstrip(" ,.:;-!?")

print(_strip_wake_words("turn on the lights"))
