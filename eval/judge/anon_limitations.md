# Anonymised re-scoring — known limitations

## One prompt names a game that also appears as an entry

Round 5 asks for something "Like Stardew Valley but more challenging". That
title cannot be removed without changing the task. Stardew Valley is also the
first entry in the revised system's round 5 set and in ChatGPT's round 1 set,
so a judge could plausibly infer its identity in round 5 from the prompt plus
the attributes shown -- 98% positive over 886,195 reviews, tagged Farming Sim
and Pixel Graphics.

Anonymisation is therefore incomplete for that one entry. Round 5 should be
checked separately when reading the result: if the effect holds with round 5
excluded, the inference is not driving it.

## One entry has no attributes

Remnant II is not in the catalogue snapshot, so ChatGPT's round 3 set carries
one entry marked as unavailable rather than described. Nothing was invented for
it. A judge scoring verifiability may reasonably penalise that entry, which is
a data limitation rather than a property of the recommendation, so round 3's
ChatGPT figure should be read with that in mind.

## Review counts are retained

The claim under test is that the judge scores by *recognising* titles, not that
it responds to popularity. Popularity is legitimate evidence for projected
satisfaction, so review counts stay visible. This makes the test conservative:
scores converging once names are hidden, while popularity remains visible, is
stronger evidence for a recognition effect than it would be with both removed.

A separate variant hiding review counts as well would test recognition and
popularity together. It is not run here.

## What a null result would mean

If the scores do not move, that is evidence the gap between systems reflects
the attributes rather than familiarity -- which would undercut the argument in
OPTIMIZED_DESIGN §11.3 rather than support it. The experiment is set up so
either outcome is informative.
