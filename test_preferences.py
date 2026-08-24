from recommender import GameSearchEngine

_extract_preferences = GameSearchEngine._extract_preferences

test_cases = [
    ("free roam open world",  "free",    False),
    ("freedom fighter game",  "free",    False),
    ("I want a free game",    "free",    True),
    ("no cost horror game",   "free",    True),
    ("cheap indie horror",    "cheap",   True),
    ("low price RPG",         "cheap",   True),
    ("space combat game",     "windows", False),
    ("compact city builder",  "windows", False),
    ("PC strategy game",      "windows", True),
    ("macabre puzzle game",   "mac",     False),
    ("mac exclusive game",    "mac",     True),
    ("new action RPG",        "recent",  True),
    ("classic retro RPG",     "classic", True),
    ("linux native game",     "linux",   True),
    ("steam deck compatible", "linux",   True),
]

passed = 0
failed = 0
for query, key, expected in test_cases:
    result = _extract_preferences(query)
    actual = result[key]
    status = "✅" if actual == expected else "❌"
    if actual == expected:
        passed += 1
    else:
        failed += 1
    print(f"{status} '{query}' → {key}={actual} (expected {expected})")

print(f"\n{passed}/{passed + failed} passed")
