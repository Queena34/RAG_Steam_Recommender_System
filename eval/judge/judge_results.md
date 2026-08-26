# Judge results

- Scoring passes: 3
- Values are mean ± standard deviation across passes

| Round | Original system | Revised system | ChatGPT |
|---|---|---|---|
| 1 | 89.0 ± 1.0 | 68.3 ± 0.6 | 96.0 ± 0.0 |
| 2 | 73.3 ± 2.5 | 56.3 ± 0.6 | 98.7 ± 0.6 |
| 3 | 62.7 ± 2.5 | 75.3 ± 1.2 | 98.7 ± 0.6 |
| 4 | 73.0 ± 1.7 | 56.0 ± 5.3 | 94.7 ± 1.2 |
| 5 | 58.0 ± 1.0 | 56.7 ± 4.9 | 95.7 ± 0.6 |
| **Mean** | **71.2** | **62.5** | **96.7** |

## By dimension

| System | Relevance /30 | Verifiability /30 | Satisfaction /40 |
|---|---|---|---|
| Original system | 19.9 ± 5.9 | 26.3 ± 2.0 | 24.9 ± 4.8 |
| Revised system | 19.5 ± 4.9 | 21.6 ± 2.8 | 21.4 ± 3.6 |
| ChatGPT | 29.4 ± 0.6 | 29.6 ± 0.5 | 37.7 ± 1.0 |

## Measurement noise

Median standard deviation across passes for a single round: **1.0 points**. Differences between systems smaller than this should not be read as improvements.

---

## Comparison with the previous scoring round

| | Round 1 (weight 0.25) | Round 2 (weight 0.60) |
|---|---|---|
| Original system | 68.7 | **71.2** |
| Revised system | 60.7 | 62.5 |
| ChatGPT | 96.7 | 96.7 |
| Gap, original minus revised | 8.0 | **8.7** |
| Within-round judge noise | 2.9 | 1.0 |

**The gap did not narrow.** The revised system gained 1.8 points; the gap
against the original widened slightly.

**A fixed input moved 2.5 points between rounds.** The original system's
recommendations are the text recorded in report Table 14 and were identical in
both packets. It scored 68.7 in the first scoring round and 71.2 in the second.

That number is the more useful result here. Within-round spread across three
passes was 2.9 points and then 1.0, but the same content re-scored in a
separate batch moved 2.5. Repeating passes inside one batch measures only part
of the variance; a second source sits between batches and this protocol does
not capture it. The revised system's 1.8-point gain is smaller than that drift
and cannot be called an improvement.

## By dimension

| System | Relevance /30 | Verifiability /30 | Satisfaction /40 |
|---|---|---|---|
| Original | 19.9 ± 5.9 | 26.3 ± 2.0 | 24.9 ± 4.8 |
| Revised | 19.5 ± 4.9 | **21.6 ± 2.8** | 21.4 ± 3.6 |
| ChatGPT | 29.4 ± 0.6 | 29.6 ± 0.5 | 37.7 ± 1.0 |

Relevance is level, at -0.4. The deficit is concentrated in verifiability
(-4.7) and projected satisfaction (-3.5), both of which turn on whether the
judge recognises a title. Median positive reviews rose from 468 to 2,593 but
remain far below the original's 8,935 and ChatGPT's 37,721.

## Round 3

The revised system scores 75.3 against the original's 62.7, a 12.6-point lead
well outside the noise, and its only round above the original. Round 3 is the
negation prompt asking for combat games to play with friends, where the
original recommended a single-player title. Game modes entered the indexed
documents in PRD-001; this is that change showing up end to end.
