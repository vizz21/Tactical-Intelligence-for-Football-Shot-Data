# Error analysis (counts; context in claude/phase6_testing_notes.md)

| # | Error type | Where measured | Count | Example |
|---|---|---|---|---|
| 1 | Colloquial words not understood by retrieval | 50 questions, 6 colloquial | 4 of 6 with nP@10 <= 0.2 | "Strikes that were smothered before they got near the goal" (blocked) |
| 2 | Scope false accept (off-topic treated as in scope) | 50 questions | 3 of 11 out-of-scope, plus 1 minute-only query as partial | "When does Barcelona play next?" returns Barcelona shots |
| 3 | Scope false refusal | 50 questions | 2 of 34 in scope | "Screamers from miles out" |
| 4 | Wrong ranking ("highest xG ...") | eval arm C, rule diagnostic | 55 of 130 evidence answers | "[1] has the highest xG at 0.06, followed by [2] at 0.16" |
| 5 | Wrong "all three ..." (goals / foot / area) | eval arm C, rule diagnostic | 30 instances (area 18, foot 6, goals 4, saved 2) | "All three were goals" when two were blocked |
| 6 | Wrong team/player attribution | audit | several of 45 | "[3] by X of Bayern Munich" (Bayer Leverkusen) |
| 7 | Garbled or repeated text | audit clarity | mean 2.67 / 5 | dangling "Result [2]." |
| 8 | Checker false alarm (typo in a name or number) | audit, 15 flagged | 4 | "Demirovi", "xGen" |
| 9 | Checker blind spot (unfaithful but passed) | audit, 30 passed | 19 | types 4 to 6 |

Rows 4 and 5 can overlap (one answer can have several errors); the diagnostic's per-answer total is 69 of 130. No model or rule was changed after seeing these results. The options (verification plus fallback to the rule template, or system-stated group facts) need a decision.
