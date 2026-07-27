# Next 4 moves — from seed-42 draft to submittable paper

*Companion to `paper_draft_v3.md`. This is the operational checklist that
takes the paper from a "single-seed, one-axis-cross-cut-missing" state
to a "camera-ready, multi-seed, fully-answered" state.*

*Ordered by priority. Do 1 before 2 (both are background so they can
overlap once 1 is running). Do 3 after 1 finishes. Do 4 anytime — it's
a 30-second sanity check.*

---

## Move 1 — Multi-seed RQ7 sweep (**highest priority**)

### Why
The current RQ7 matrix (Table 10 in the draft) is single-seed (seed 42).
No IEEE TII / Reliability reviewer will accept single-seed adversarial-
robustness numbers. Multi-seed aggregation replaces every point estimate
in Table 10 with `mean ± std ± 95 %-CI`, which is the standard bar for
a security/robustness paper.

### What it delivers
- `results/rq7_poisoning_seeds/seed_43/metrics.json` through `seed_46`
- Aggregated `results/rq7_poisoning_seeds/aggregated_metrics.json` with
  the 25-cell matrix as mean ± std ± CI across 5 seeds
- Rebuilt Table 10 with statistical rigor

### Command
```powershell
cd "C:\Program Files\Project\AirCraft\FL-for-Aircraft"
# One command, sequential:
python scripts/run_rq7_multiseed.py --seeds 43 44 45 46
# Then aggregate all 5 seeds (42 + the four new ones):
python scripts/aggregate_rq7_seeds.py --seeds 42 43 44 45 46 --out-file results/rq7_poisoning_seeds/aggregated_metrics.json
```

### Time & risk
- **Compute:** ~45 min per seed × 4 new seeds = **~3 hours total**,
  unattended.
- **Failure risk:** low. The `_eval_backdoor_success` `WindowedArrays`
  bug is already fixed. Resume mode is on by default, so if a seed
  crashes mid-run, re-running the same command picks up from the
  per-round CSVs.
- **Watch for:** log the terminal output to a file
  (`| Tee-Object -FilePath run.log`) so we can inspect it later without
  re-running.

### Success criterion
`aggregated_metrics.json` exists and contains `cells` with fields
`best_rmse.mean`, `best_rmse.std`, `best_rmse.ci95_low`,
`best_rmse.ci95_high` for all 25 cells (or 24 if we accept the
mathematically-undefined `D54_coord_krum_f2` as skipped).

### Paper impact
- **Table 10** gets rebuilt with `mean ± CI`.
- **Section 9.2 observations** are re-checked: does the "Krum → 0 % ASR"
  claim survive across seeds? Does the "coord ×−10 breaks
  trimmed/median" claim survive?
- If any observation weakens under multi-seed, we soften the prose;
  if it strengthens (e.g. always 0 % ASR under Krum), we harden it.

---

## Move 2 — Multi-seed RQ2 sweep (**second priority, can overlap with Move 1**)

### Why
Tables 5–8 in the draft (FedRep, FedCCFA, FedProx μ-sweep, imbalance-
aware) are also single-seed. The Axis 1 findings (personalization is
~10× more effective than proximal / reweighting) rest on these numbers.
Multi-seed makes them defensible.

### What it delivers
- 2-3 new seeds' worth of `metrics.json` for the four Axis 1 methods
- A short aggregation script (~30 lines) that averages the per-subset
  RMSE / F1 across seeds and updates Tables 5-8 with mean ± std

### Commands
```powershell
cd "C:\Program Files\Project\AirCraft\FL-for-Aircraft"

# FedRep on seeds 43, 44 (h1, e1 as in v3 draft):
python scripts/run_rq2_fedrep.py --seed 43 --out-dir results/rq2_fedrep_seeds/seed_43
python scripts/run_rq2_fedrep.py --seed 44 --out-dir results/rq2_fedrep_seeds/seed_44

# FedCCFA on seeds 43, 44:
python scripts/run_rq2_fedccfa.py --seed 43 --out-dir results/rq2_fedccfa_seeds/seed_43
python scripts/run_rq2_fedccfa.py --seed 44 --out-dir results/rq2_fedccfa_seeds/seed_44

# FedProx (just mu=0.1, the winner):
python scripts/run_rq2_fedprox.py --seed 43 --mus 0.1 --out-dir results/rq2_fedprox_seeds/seed_43
python scripts/run_rq2_fedprox.py --seed 44 --mus 0.1 --out-dir results/rq2_fedprox_seeds/seed_44

# Imbalance-aware (just validation_f1, the winner):
python scripts/run_rq2_imbalance_aware.py --seed 43 --schemes validation_f1 --out-dir results/rq2_imbalance_aware_seeds/seed_43
python scripts/run_rq2_imbalance_aware.py --seed 44 --schemes validation_f1 --out-dir results/rq2_imbalance_aware_seeds/seed_44
```

> **NOTE on script names:** the exact scripts are `scripts/run_rq2_*.py`
> — verify they accept `--seed` and `--out-dir` flags before kicking
> off. If any script uses a different flag convention (e.g. `--out`),
> adjust accordingly.

### Time & risk
- **Compute:** FedRep ≈ 4.5 min/seed, FedCCFA ≈ 4 min/seed,
  FedProx (1 μ) ≈ 6 min/seed, Imbalance-aware (1 scheme) ≈ 5 min/seed
  → ~40 min for two seeds across all four methods.
- **Aggregation script:** I need to write a small `aggregate_rq2_seeds.py`
  (~30 lines). Not yet on disk — flag as a mini-todo.

### Success criterion
Tables 5, 6, 7, 8 in `paper_draft_v3.md` gain a `± std` column across
2 or 3 seeds. Key claim to preserve: **FedRep's gap-closed % stays
≥ 65 %** across seeds; **FedProx-best stays ≤ 10 %**; the ~10 × ratio
in Table 9 stays intact.

### Paper impact
Table 9's "**~ 10 × ratio**" claim gets a proper statistical footing.

---

## Move 3 — The bridge experiment (**COMPLETE — 2026-07-24**)

### Result (5 seeds: 42, 43, 44, 45, 46)

| Seed | best_round | Attacker ASR | Honest mean ASR | Delta |
|---:|---:|---:|---:|---:|
| 42 | 47 | 0.800 | 0.814 | −0.014 |
| 43 | 11 | 0.182 | 0.141 | +0.041 |
| 44 | 45 | 1.000 | 1.000 |  0.000 |
| 45 | 35 | 0.617 | 0.612 | +0.005 |
| 46 | 21 | 0.481 | 0.599 | −0.118 |
| **mean ± std** | 31.8 ± 15.5 | **0.616 ± 0.311** | **0.633 ± 0.320** | **−0.017 ± 0.060** |

### Outcome bucket (from the "possible outcomes" table below)
**"Attack partially succeeds" (mean shift) + "Attack fully succeeds"
(worst-case seed)** — a bimodal distribution. Mean honest ASR (0.633)
is 30 pp below vanilla FedAvg (0.949) but with catastrophic std 0.320.
Attacker−honest delta = −0.017 ± 0.060 (indistinguishable from zero)
confirms personalization does *not* shield honest clients.

### Paper story
"Personalization alone is not a reliable backdoor defense; the
apparent mean reduction is an early-stopping artifact. Krum remains
required. Krum + FedRep stacked is now the recommended two-axis
defense (Table 11 row 7). Confirms the two-axis orthogonality
argument."

### Artifacts
- `scripts/run_rq2_fedrep_under_backdoor.py` — new runner
- `scripts/aggregate_bridge_seeds.py` — new aggregator
- `results/rq_bridge/seed_{42..46}/metrics.json` — per-seed runs
- `results/rq_bridge/metrics_aggregated.json` — 5-seed statistics
- `paper_draft_v3.md` § 9.4 — new results section (Table 12)
- Library changes: `client_hook` in `run_fedrep_from_bundle`,
  public `make_backdoor_poisoned_loader` helper

### Original planning notes below (kept for reference)

---

**Original description:**

## ~Move 3~ — The bridge experiment (**highest research value**)

### Why
Section 10.4 of the draft currently lists FedRep-under-backdoor as
"future work." Running even one seed of it turns that limitation into
a *result* and makes the paper genuinely a unified two-axis study
instead of two half-studies stapled together. This is the single
experiment that closes the paper's open question.

### What it delivers
- A new experiment cell: FedRep training loop wrapping `client_3` with
  the `BackdoorAttacker`
- Metrics: does FedRep also reduce Attack Success Rate below the
  vanilla-FedAvg 98 % baseline, or does the shared encoder still absorb
  the backdoor pattern?
- A new results section (proposed § 9.4 in the paper) discussing the
  three-way interaction between personalization, backdoor attack, and
  aggregator choice.

### Steps
1. **Write the wrapper** (`scripts/run_rq2_fedrep_under_backdoor.py`,
   ~150 lines):
   - Reuse `run_rq2_fedrep.py`'s FedRep loop.
   - Replace `client_3`'s `FederatedClient` with a `BackdoorAttacker`
     from `src/fl_aircraft/fl/poisoning.py` (drop-in compatible).
   - Add triggered-eval hook after the last round (reuse
     `_eval_backdoor_success` from `scripts/run_rq7.py`, refactored
     into a shared helper).
   - Expected new metrics: `clean_macro_rmse`, `clean_macro_f1`,
     `triggered_fault_positive_rate`, `attack_success_rate` — all
     computed on the personalized global model.
2. **Smoke test** on seed 42:
   ```powershell
   python scripts/run_rq2_fedrep_under_backdoor.py --seed 42 --out-dir results/rq_bridge/seed_42
   ```
3. **If it works**, run seeds 43-44 in the background as with Move 2.
4. **Report** as a new subsection § 9.4 titled *"Bridge: does
   personalization also defend?"* with a small 3-row table:
   `FedAvg-clean, FedAvg-backdoor, FedRep-backdoor` → RMSE, F1, ASR.

### Time & risk
- **Development:** ~2 hours (writing the wrapper + smoke test).
- **Compute:** ~5 min per seed for the run.
- **Risk:** medium — this is new code, so there may be integration bugs
  between FedRep's per-client head persistence and the backdoor
  wrapper. Budget for one debug cycle.

### Possible outcomes and paper implications

| Outcome | ASR value | Paper story |
|---|---:|---|
| Attack fully succeeds | ~ 98 % | "Personalization alone is not a backdoor defense; still need Krum on the encoder." Adds strong evidence for the *stacking* recommendation in Table 11. |
| Attack partially succeeds | 30 – 90 % | "Per-client heads absorb some of the trigger effect but the shared encoder still learns the backdoor." Bridge-cell adds a middle row. |
| Attack fully neutralized | ~ 0 % | "Personalization is a natural backdoor defense on prognostics — the per-client heads decouple from the shared encoder's poisoned features." A big result that would justify a longer follow-up. |

**Any** of these three outcomes is a publishable result. That's why
this experiment is the paper's highest research-value next step.

### Success criterion
`results/rq_bridge/seed_42/metrics.json` exists with a
`backdoor_evaluation` section including a numerical `attack_success_rate`
field. New subsection § 9.4 written into `paper_draft_v3.md` with the
three-row comparison table.

---

## Move 4 — Sanity-check the Mermaid diagrams (**30 seconds, do anytime**)

### Why
Five diagrams were added to v3. GitHub, VS Code, Obsidian, and Typora all
render Mermaid slightly differently. A rendering failure would leave a
"grey blob of code" where a figure should be — embarrassing in a
supervisor review or PDF export.

### Steps
1. Open `research_Paper/paper_draft_v3.md` in VS Code.
2. `Ctrl+Shift+V` (Open Preview) or click the "preview" icon top-right.
3. Scroll to each of the five figures:
   - Fig 1 — Federation topology (§ 3.1)
   - Fig 2 — Multi-task 1-D CNN (§ 3.2)
   - Fig 3 — Two-axis frame (§ 3.3)
   - Fig 4 — One communication round (§ 3.4)
   - Fig 5 — Backdoor trigger construction (§ 5.2.3)
4. Confirm each renders as an actual diagram, not raw Mermaid source.
5. If VS Code's built-in preview doesn't render Mermaid, install the
   [Markdown Preview Mermaid Support](vscode:extension/bierner.markdown-mermaid)
   extension (open the link, click "Install").
6. If any diagram fails to render, note which one and I'll patch it.

### Time
- 30 seconds if all five render on first try.
- 5-10 minutes if the extension needs to be installed.

### Success criterion
All five figures visible as actual diagrams in the VS Code Markdown
preview. Screenshot each and file them under
`research_Paper/figs_preview/` for the supervisor review.

---

## Timeline summary

| Day | Session | Move | Compute | Human-time |
|---|---|---|---|---|
| Today evening | Kick off & sleep | Move 1 (seeds 43-46 RQ7) | 3 hrs unattended | 5 min to launch |
| Tomorrow morning | Check & kick off | Move 2 (Axis-1 seeds 43-44) | 40 min unattended | 5 min to launch |
| Tomorrow morning | Sanity check | Move 4 (diagram preview) | — | 5 min |
| Tomorrow afternoon | Write & test | Move 3 dev — bridge wrapper | ~5 min | 2 hrs |
| Tomorrow evening | Smoke test & sleep | Move 3 seed 42 + seeds 43-44 background | ~15 min unattended | 5 min |
| Day 3 morning | Aggregate | Both aggregation scripts | ~1 min | 30 min to update tables |
| Day 3 afternoon | Update draft | v4 with mean±CI, bridge result | — | 2 hrs |
| Day 3 evening | Ready for supervisor review | — | — | — |

**Total elapsed:** ~ 2 days from now to a supervisor-review-ready v4
draft.

---

## Post-move-4 checklist (for the v4 draft)

Once all four moves complete, do these small updates to `paper_draft_v3.md`
(or write `paper_draft_v4.md`):

- [x] Replace point estimates in Table 10 with `mean ± CI` from
      `aggregated_metrics.json`. *(done during Move 1)*
- [x] Replace point estimates in Tables 5–8 with `mean ± std` from the
      Axis 1 multi-seed aggregation. *(done during Move 2)*
- [x] Delete "single seed" from the abstract Limitation and § 10.3
      Limitations. *(done during Moves 1 + 2)*
- [x] Add § 9.4 "Bridge: does personalization also defend?" with the
      three-row comparison table. *(done during Move 3 — became a
      more detailed Table 12 with per-seed + aggregate)*
- [x] Move the bridge experiment out of § 10.4 (future work) into the
      results (§ 9.4) — leave a shorter "future work" bullet for
      *combining FedRep + Krum on the encoder*. *(done during Move 3;
      § 10.4 now titled "Further extensions")*
- [ ] Screenshot each figure for the supervisor / journal submission.
- [ ] Convert informal references into BibTeX.
- [ ] Optional: add a Fig 16 visualizing the bridge results (bar
      chart with error bars comparing vanilla / FedRep-bridge / Krum,
      + scatter of per-seed honest ASR vs best_round).

---

*Document created 2026-07-23. Update as moves complete.*
