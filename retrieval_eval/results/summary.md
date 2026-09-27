# Retrieval experiment results

27 queries over 28 history rows (20 with history, 7 without).

## Variants at threshold 0.6

| Variant | Family recall | Latest-status hit | Precision | False-hit rate (no history) | Other-patient leaks |
|---|---|---|---|---|---|
| spec_baseline | 0.77 | 0.60 | 1.00 | 0 | 0 |
| +watch_for | 0.98 | 0.95 | 1.00 | 0 | 0 |
| +history | 1.00 | 1 | 1.00 | 0 | 0 |
| full | 1.00 | 1 | 1.00 | 0 | 0 |

## Threshold sweep

| Threshold | Baseline recall | Baseline latest | Baseline false-hit | Full recall | Full latest | Full precision | Full false-hit |
|---|---|---|---|---|---|---|---|
| 0.400 | 0.97 | 0.95 | 0.57 | 1.00 | 1 | 0.64 | 0.57 |
| 0.425 | 0.96 | 0.95 | 0.43 | 1.00 | 1 | 0.78 | 0.43 |
| 0.450 | 0.94 | 0.95 | 0.43 | 1.00 | 1 | 0.87 | 0.43 |
| 0.475 | 0.94 | 0.95 | 0.43 | 1.00 | 1 | 0.88 | 0.43 |
| 0.500 | 0.87 | 0.80 | 0.29 | 1.00 | 1 | 0.93 | 0.29 |
| 0.525 | 0.87 | 0.80 | 0.14 | 1.00 | 1 | 0.95 | 0.14 |
| 0.550 | 0.84 | 0.75 | 0 | 1.00 | 1 | 1.00 | 0 |
| 0.575 | 0.80 | 0.65 | 0 | 1.00 | 1 | 1.00 | 0 |
| 0.600 | 0.77 | 0.60 | 0 | 1.00 | 1 | 1.00 | 0 |
| 0.625 | 0.71 | 0.60 | 0 | 1.00 | 1 | 1.00 | 0 |
| 0.650 | 0.64 | 0.50 | 0 | 0.97 | 0.95 | 1.00 | 0 |
| 0.675 | 0.54 | 0.35 | 0 | 0.93 | 0.90 | 1.00 | 0 |
| 0.700 | 0.48 | 0.30 | 0 | 0.90 | 0.90 | 1.00 | 0 |
| 0.725 | 0.42 | 0.30 | 0 | 0.90 | 0.90 | 1.00 | 0 |
| 0.750 | 0.38 | 0.30 | 0 | 0.90 | 0.90 | 1.00 | 0 |
| 0.775 | 0.18 | 0.20 | 0 | 0.80 | 0.80 | 1.00 | 0 |
| 0.800 | 0.13 | 0.10 | 0 | 0.80 | 0.80 | 1.00 | 0 |

## Weight sweep (spec baseline, threshold 0.6)

| Symptom weight | Family recall | Latest-status hit | Precision | False-hit rate (no history) |
|---|---|---|---|---|
| 0.5 | 0.64 | 0.40 | 1.00 | 0 |
| 0.6 | 0.75 | 0.60 | 1.00 | 0 |
| 0.7 | 0.77 | 0.60 | 1.00 | 0 |
| 0.8 | 0.78 | 0.65 | 1.00 | 0 |
| 0.9 | 0.83 | 0.75 | 1.00 | 0 |
| 1.0 | 0.86 | 0.90 | 1.00 | 0 |

## Score separation

- Relevant pairs: n=63, median 0.657, 67% above 0.6.
- Irrelevant pairs: n=617, median 0.276, max 0.543, 0% above 0.6.

## Per-query results (full variant)

| Query | Recall | Latest hit | Returned |
|---|---|---|---|
| swelling_paraphrase | 1.00 | True | sw1(swelling), sw2(swelling), sw3(swelling), sw4(swelling), sw5(swelling) |
| swelling_legs_puffy | 1.00 | True | sw1(swelling), sw2(swelling), sw3(swelling), sw4(swelling), sw5(swelling) |
| swelling_checkin_item | 1.00 | True | sw1(swelling), sw2(swelling), sw3(swelling), sw4(swelling), sw5(swelling) |
| swelling_followup_weak_context | 1.00 | True | sw1(swelling), sw2(swelling), sw3(swelling), sw4(swelling), sw5(swelling) |
| weight_paraphrase | 1.00 | True | wt1(weight), wt2(weight), wt3(weight) |
| weight_checkin_item | 1.00 | True | wt1(weight), wt2(weight), wt3(weight) |
| breathing_exertion | 1.00 | True | sob1(breathing), sob2(breathing), sob3(breathing), sob4(breathing) |
| breathing_orthopnea | 1.00 | True | sob1(breathing), sob2(breathing), sob3(breathing), sob4(breathing) |
| breathing_checkin_item | 1.00 | True | sob1(breathing), sob2(breathing), sob3(breathing), sob4(breathing) |
| wheeze_paraphrase | 1.00 | True | wz1(wheeze), wz2(wheeze) |
| cough_mucus | 1.00 | True | cg1(cough), cg2(cough) |
| chest_pain_new | 1.00 | True | cp1(chest_pain), cp2(chest_pain) |
| dizzy_standing | 1.00 | True | dz1(dizziness), dz2(dizziness) |
| dizzy_checkin_item | 1.00 | True | dz1(dizziness), dz2(dizziness) |
| rash_back | 1.00 | True | rash1(rash), rash2(rash) |
| knee_again | 1.00 | True | knee1(knee) |
| sleep_trouble | 1.00 | True | slp1(sleep) |
| nausea_pills | 1.00 | True | nau1(nausea) |
| throat_scratchy | 1.00 | True | thr1(throat) |
| two_symptoms | 1.00 | True | sw1(swelling), sw2(swelling), sob1(breathing), sob2(breathing), sob3(breathing), sw3(swelling), sw4(swelling), sw5(swelling), sob4(breathing) |
| no_history_confusion | n/a | n/a | none |
| no_history_confusion_checkin | n/a | n/a | none |
| distractor_headache | n/a | n/a | none |
| distractor_itchy_eyes | n/a | n/a | none |
| distractor_back_pain | n/a | n/a | none |
| distractor_chest_congestion | n/a | n/a | none |
| distractor_eye_swelling | n/a | n/a | none |

## Per-query results (spec baseline)

| Query | Recall | Latest hit | Returned |
|---|---|---|---|
| swelling_paraphrase | 0.80 | False | sw1(swelling), sw2(swelling), sw3(swelling), sw4(swelling) |
| swelling_legs_puffy | 0.40 | False | sw3(swelling), sw4(swelling) |
| swelling_checkin_item | 0.60 | True | sw1(swelling), sw2(swelling), sw5(swelling) |
| swelling_followup_weak_context | 0.20 | False | sw2(swelling) |
| weight_paraphrase | 1.00 | True | wt1(weight), wt2(weight), wt3(weight) |
| weight_checkin_item | 1.00 | True | wt1(weight), wt2(weight), wt3(weight) |
| breathing_exertion | 0.50 | False | sob1(breathing), sob3(breathing) |
| breathing_orthopnea | 0.25 | False | sob2(breathing) |
| breathing_checkin_item | 0.50 | False | sob1(breathing), sob3(breathing) |
| wheeze_paraphrase | 1.00 | True | wz1(wheeze), wz2(wheeze) |
| cough_mucus | 0.50 | False | cg1(cough) |
| chest_pain_new | 1.00 | True | cp1(chest_pain), cp2(chest_pain) |
| dizzy_standing | 1.00 | True | dz1(dizziness), dz2(dizziness) |
| dizzy_checkin_item | 1.00 | True | dz1(dizziness), dz2(dizziness) |
| rash_back | 1.00 | True | rash1(rash), rash2(rash) |
| knee_again | 1.00 | True | knee1(knee) |
| sleep_trouble | 1.00 | True | slp1(sleep) |
| nausea_pills | 1.00 | True | nau1(nausea) |
| throat_scratchy | 1.00 | True | thr1(throat) |
| two_symptoms | 0.68 | False | sw1(swelling), sw2(swelling), sob1(breathing), sob3(breathing), sw3(swelling), sob4(breathing) |
| no_history_confusion | n/a | n/a | none |
| no_history_confusion_checkin | n/a | n/a | none |
| distractor_headache | n/a | n/a | none |
| distractor_itchy_eyes | n/a | n/a | none |
| distractor_back_pain | n/a | n/a | none |
| distractor_chest_congestion | n/a | n/a | none |
| distractor_eye_swelling | n/a | n/a | none |
