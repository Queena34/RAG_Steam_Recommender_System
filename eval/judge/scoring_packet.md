# Blind scoring packet

Five rounds, three anonymous recommendation sets each. Please score the whole packet **3 times**, in **separate conversations**, without referring back to earlier passes.

The sets come from different recommender systems. Which is which is not disclosed, and the ordering changes between rounds. Please judge only what each set contains.

## Why several passes

An LLM judge is not deterministic. A single pass cannot separate a real difference between systems from variation in the scoring itself. With 3 passes the spread across passes can be reported next to the means, so a claimed improvement can be checked against the noise in the measurement.

## Rubric

```
Score each recommendation set out of 100, as the sum of three dimensions:

  Relevance to the prompt                                        30 points
  Verifiability on the Steam store                               30 points
  Projected player satisfaction, from an experienced gamer's
  perspective                                                    40 points

Judge only what is shown. Do not attempt to identify which system produced
which set. Give the three sub-scores for every set, plus one sentence of
justification.
```

## Rounds

### Round 1 — Simple / Vague

**Prompt:** Recommend some relaxing games

**Set A**
- Stardew Valley
- Dorfromantik
- Unpacking
- A Short Hike
- Tiny Glade

**Set B**
- Unpacking
- Dorfromantik
- Virtual Cottage
- Hello Kitty Island Adventure
- SUMMERHOUSE

**Set C**
- Monument Valley
- Ouros
- Aery - Calm Mind
- Solo
- Areia: Pathway to Dawn


### Round 2 — Detailed Descriptive

**Prompt:** I want games that are intellectually stimulating and require some gaming experience

**Set A**
- Chants of Sennaar
- Storyteller
- SteamWorld Dig
- Minds Beneath Us
- Fantasy Map Simulator

**Set B**
- Ghostory
- Halver
- Escape: The Brother's Saloon
- The Lab - Escape Room
- Agent Brain: Tricky Puzzles

**Set C**
- Outer Wilds
- Return of the Obra Dinn
- The Witness
- Slay the Spire
- EXAPUNKS


### Round 3 — Negation-Constrained

**Prompt:** I don't want cozy or cute games - give me some combat games to play with friends

**Set A**
- Circle Empires Rivals
- Battle Islands
- Tom Clancy's Rainbow Six Siege
- Bodycam
- Super Animal Royale

**Set B**
- Helldivers 2
- Deep Rock Galactic
- Warhammer 40,000: Darktide
- GTFO
- Remnant II

**Set C**
- ShellShock Live
- CRSED: Cuisine Royale
- Super Animal Royale
- Bopl Battle
- We Who Are About To Die


### Round 4 — Hard-Constrained

**Prompt:** Free co-op multiplayer games

**Set A**
- Sven Co-op
- PICO PARK:Classic Edition
- Operation: Tango - Friend Pass
- Wolvesville
- Sonder

**Set B**
- Warframe
- Path of Exile
- We Were Here
- Unturned
- SCP: Secret Laboratory

**Set C**
- Sonder
- PICO PARK:Classic Edition
- Never Split the Party
- Counter Agents
- Handy Dandy


### Round 5 — Analogy

**Prompt:** Like Stardew Valley but more challenging

**Set A**
- Graveyard Keeper
- Kynseed
- Rune Factory 4 Special
- Core Keeper
- Atomicrops

**Set B**
- Diablo® IV
- TUNIC
- ICARUS
- Life is Feudal
- A Difficult Game About Climbing

**Set C**
- Roots of Pacha
- Land Of Idyllic Beauty
- Everdream Valley
- Cornucopia®
- Stardew Valley


---

## How to answer

Reply with **one JSON object and nothing else** — no commentary before or after it. Every round and every set must appear, with all three sub-scores as numbers.

```json
{
  "1": {
    "A": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"},
    "B": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"},
    "C": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"}
  },
  "2": { "A": {...}, "B": {...}, "C": {...} }
}
```

Totals are computed from the sub-scores, so do not include them.
Save the reply verbatim as `scores_pass1.json` (then `2`, `3`) in `eval/judge/`, and run `aggregate_scores.py`.
