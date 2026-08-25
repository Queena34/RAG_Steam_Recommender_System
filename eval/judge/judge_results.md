# Judge results

- Scoring passes: 3
- Values are mean ± standard deviation across passes

| Round | Original system | Revised system | ChatGPT |
|---|---|---|---|
| 1 | 83.3 ± 4.5 | 74.3 ± 2.9 | 96.0 ± 0.0 |
| 2 | 73.0 ± 1.7 | 45.3 ± 9.3 | 99.0 ± 0.0 |
| 3 | 61.0 ± 6.6 | 67.7 ± 5.5 | 98.7 ± 0.6 |
| 4 | 68.0 ± 2.0 | 57.7 ± 5.9 | 94.7 ± 0.6 |
| 5 | 58.0 ± 3.5 | 58.7 ± 4.0 | 95.0 ± 1.0 |
| **Mean** | **68.7** | **60.7** | **96.7** |

## By dimension

| System | Relevance /30 | Verifiability /30 | Satisfaction /40 |
|---|---|---|---|
| Original system | 20.0 ± 5.7 | 25.5 ± 2.5 | 23.2 ± 4.4 |
| Revised system | 18.7 ± 5.0 | 23.1 ± 3.9 | 19.0 ± 5.1 |
| ChatGPT | 29.4 ± 0.6 | 29.9 ± 0.4 | 37.4 ± 1.3 |

## Measurement noise

Median standard deviation across passes for a single round: **2.9 points**. Differences between systems smaller than this should not be read as improvements.

---

## Popularity of what each system recommended

Median positive reviews across the five recommended titles per round.

| Round | Original | Revised | ChatGPT |
|---|---|---|---|
| 1 | 10,173 | 85 | 26,639 |
| 2 | 6,518 | 60 | 28,093 |
| 3 | 30,411 | 25,297 | 80,268 |
| 4 | 1,095 | 468 | 209,822 |
| 5 | 23,445 | 596 | 5,218 |
| **All** | **8,935** | **468** | **37,721** |

The revised system recommends titles with roughly one nineteenth the review
count of the ones the original returned. Round 2, its worst round at 45.3
against 73.0, has a median of 60 positive reviews.

## Reading

**The revised system scores 8 points below the original**, and the gap exceeds
the 2.9-point measurement noise on rounds 2 and 4. Only round 3 improves.

**Re-scoring the baseline was what made this visible.** The original system's
published figure is 60.2 from a single scoring pass; re-scored over three
passes it is 68.7. Comparing the revised system's 60.7 against the published
60.2 would have read as no change, when measured against a like-for-like
baseline it is a regression.

**The cause is a ranking weight, not the retrieval work.** C7 lowered the index
threshold from 100 positive reviews to 10, admitting 12,244 long-tail titles.
C5's quality signal carries a weight of 0.25 against relevance at 0.75, which
is not enough to stop a semantically close title with 60 reviews from
outranking a well-regarded one. The retrieval evaluation could not have caught
this: its ground truth asks whether tags match, or whether a specific game was
found, and neither asks whether a game is worth recommending.

**Obscurity is not the same as poor quality, but it is not free either.** Round
4's Sonder is free, tagged co-op, and 90% positive -- a correct answer a judge
cannot easily verify, which is the familiarity bias argued in
OPTIMIZED_DESIGN §11.3. Round 2's picks, at a median of 60 reviews, are a
different case: for a request wanting something intellectually demanding, a
title almost nobody has played gives the user nothing to go on. Both effects
are present; the second is the larger one here.
