# End-to-end judge protocol

Re-running the LLM-as-judge evaluation from report Section 3.3 against the
revised system, with two changes to the protocol itself.

## Who may score

**Whoever implemented the changes must not score them.** Report Section 3.3.1
states the reason for using a third-party judge:

> To reduce evaluator subjectivity, we used a third-party LLM (Claude) as an
> independent judge ... This setup avoids the subjectivity inherent in
> self-evaluation.

If the party that made the changes also scores them, that property is gone and
the claim in the report becomes untrue. The scoring passes therefore have to be
run by someone with no stake in the outcome, in a session that has not seen the
implementation work — the same way the original run was done.

Everything except scoring can be automated, and is:

| Step | Who |
|---|---|
| Run the revised system, capture outputs | `run_system.py` |
| Anonymise and assemble the packet | `build_packet.py` |
| **Score the packet** | **an independent judge** |
| Join scores to the key, summarise | `aggregate_scores.py` |

## Two changes to the protocol

**Several scoring passes.** An LLM judge is not deterministic. A single pass
cannot separate a difference between systems from variation in the scoring. The
packet asks for three passes in separate conversations, and
`aggregate_scores.py` reports the spread across them next to the means. A
difference smaller than that spread is not evidence of an improvement.

**The baseline is re-scored.** The figures in Table 14 come from one scoring
pass. Comparing a fresh multi-pass mean against them would confound a change in
the system with a change in the measurement, so the recorded outputs of the
original system and of ChatGPT are re-scored in the same packet, blind,
alongside the revised system. Their published scores are kept in
`baseline_outputs.json` for reference only.

## Running it

```bash
uv run python eval/judge/run_system.py --runs 3     # about 27 minutes
uv run python eval/judge/build_packet.py --passes 3
```

Hand `scoring_packet.md` to the judge. **Do not include `scoring_key.json`** —
it maps the anonymous labels back to their systems.

Save each returned pass as `scores_pass1.json`, `scores_pass2.json`,
`scores_pass3.json`, then:

```bash
uv run python eval/judge/aggregate_scores.py
```

## Files

| File | Contents |
|---|---|
| `prompts.jsonl` | The five prompts from Table 14, unchanged |
| `baseline_outputs.json` | Recorded outputs and original scores for the other two systems |
| `system_outputs.jsonl` | Revised system's responses, including PRD-004 telemetry |
| `scoring_packet.md` | Blind packet for the judge |
| `scoring_key.json` | Label to system mapping — withhold from the judge |
| `judge_results.md` | Written by `aggregate_scores.py` |

## What this measures, and what it does not

The judge protocol scores complete recommendations, which is what a user
actually receives, but its composite score cannot say which stage produced a
poor result. The retrieval evaluation in `eval/` answers that and says nothing
about the generated text. Neither replaces the other.

Five prompts also remain few. The revised protocol addresses judge noise
through repeated passes; it does not address the sample size, which would need
more prompts and, for a like-for-like comparison, ChatGPT responses for them.

---

## Choosing a judge

The judge must not be one of the systems being compared, and must not be
whoever implemented the changes. Both constraints follow from the same
principle: the protocol should exclude the conflict structurally rather than
rely on the scorer's restraint.

That rules out two candidates for this packet:

| Candidate | Problem |
|---|---|
| Whoever implemented the changes | Scores its own work; report §3.3.1's independence claim would become untrue |
| ChatGPT | Its own recommendations are one of the three sets being scored |

LLM judges show a documented preference for outputs resembling their own, and
anonymisation reduces but does not remove it. If ChatGPT is used regardless,
drop its own sets from the packet — losing the commercial comparison is a
smaller cost than having a competitor judge itself.

## Running a scoring pass

Whichever model is used:

1. **Start a fresh conversation.** Not a continuation of anything related to
   this project.
2. **Disable persistent memory.** ChatGPT carries memory across chats;
   use a Temporary Chat. The original run did the same, for the same reason —
   report §3.3.1 notes each session was temporary "to prevent user memory and
   reference".
3. **Paste `scoring_packet.md` in full**, and nothing else. No context about
   the project, the systems, or what any change was meant to achieve.
4. **Save the reply verbatim** as `scores_pass1.json`.
5. **Repeat twice more**, each in a new temporary conversation. Do not paste
   earlier answers or ask the model to reconsider — the passes are only
   independent if the model has not seen its own previous scores.

Never paste `scoring_key.json`, and do not answer questions about which set is
which, even after scoring — a follow-up pass in the same conversation would be
contaminated.
