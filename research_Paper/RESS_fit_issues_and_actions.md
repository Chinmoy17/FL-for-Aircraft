# RESS Submission — Scope-Fit Issues, Claim Softening & Lit-Study Gaps

> Audit created after reading the **official RESS Aims & Scope**. This is the
> "what must change before we submit" list, ordered by desk-reject risk.
> Companion to [`RESULTS_CONSOLIDATED.md`](../RESULTS_CONSOLIDATED.md).

---

## 1. Where we legitimately hook into the RESS scope

RESS is about the **safety and reliability of complex technological systems**
(it names *transportation systems* explicitly — aircraft fleets qualify). The
in-scope topics we can map onto:

| RESS scope topic (verbatim) | Our mapping |
| --- | --- |
| "methods and applications of automatic **fault detection and diagnosis**" | PHM / fault-state classification |
| "**test and maintenance policies**" | RUL-driven maintenance scheduling |
| "models for **ageing and life extension**" | Remaining-useful-life estimation |
| "**operator decision support systems**" | Prognostics feeding maintenance decisions |
| "**design innovation for safety and reliability**" | Robust federated training for trustworthy fleet prognostics |
| "software reliability" | (tangential) trustworthiness of the learned model |

**So there is a real hook** — but only through *prognostics-for-maintenance*,
**not** through federated learning or ML security.

---

## 2. The core fit problem (be honest with ourselves)

The scope names RAMS, PSA, uncertainty, human reliability, maintenance policies,
fault diagnosis. It **does not mention machine learning, federated learning,
cybersecurity, adversarial attacks, or data poisoning — at all.**

Our paper's *machinery* (FL algorithms, Byzantine-robust aggregation, backdoor
attacks) is exactly the vocabulary the scope is silent on. **Risk:** a strict
handling editor reads it as *"an ML/security paper — redirect to an ML journal."*

> **This is fixable but not cosmetic.** It requires reframing so the
> **reliability/safety problem is the paper** and FL/robustness is the *method*.
> Softening wording alone will not do it — see §4.

---

## 3. Issues, ranked by desk-reject / reviewer-fit risk

**Issue 1 — ML-first framing (highest risk).**
As written, the story is "we benchmark FL methods on C-MAPSS." For RESS it must
be "we quantify and mitigate a **safety hazard** in collaborative fleet
prognostics." Method is the vehicle, not the subject.

**Issue 2 — ML-only metrics, no reliability consequence (highest *substantive* gap).**
We report RMSE and attack-success-rate (ASR). RESS reviewers think in
**reliability/safety consequences**: missed failures, risk increase, effect on
maintenance decisions. A paper that never translates ASR into a maintenance/
safety impact reads as an ML experiment. **This is the single most important
enhancement — see §4.**

**Issue 3 — simulated federation / academic setup.**
The scope prizes "a balance between academic material and **practical
applications**." Our federation is simulated by splitting C-MAPSS. We must
motivate the *real* scenario (OEMs / airlines pooling fleet knowledge without
sharing raw data — a genuine industry trend) and mark real-data validation as
future work.

**Issue 4 — novelty over-claims (must soften — see §5).**
Post lit-check, "SHARP is a novel defense" and "personalization is vulnerable"
are **not** method-novel (SARS 2024; Fan & Chen 2026 already do these in general
FL). Only the *failure-masking-backdoor-on-RUL + RMSE-blindness* claim survives
as a first.

**Issue 5 — accuracy is not SOTA.**
14–20 RMSE vs ~11–12 SOTA on FD001. Must pre-empt in one sentence: the objective
is robustness/safety, not centralized accuracy; a federated multi-task model is
not comparable to a tuned centralized regressor.

**Issue 6 — threat-model realism.**
For a *reliability* venue the hazard must be credible: a compromised/insider
client, supply-chain tampering, or a malicious operator — not "an attacker for
attack's sake." Motivate it in RAMS terms.

**Issue 7 — C-MAPSS saturation.**
Ubiquitous benchmark. The federated + adversarial + safety differentiator must
be loud in the abstract or it risks "yet another C-MAPSS study."

---

## 4. Highest-leverage fix: translate ML metrics → reliability consequences

This is what turns an ML paper into a RESS paper. Concretely, add an analysis
that maps our attack/defense numbers onto **reliability/safety terms**, e.g.:

- ASR → **expected missed near-failure detections** on the test fleet (we already
  compute fault-positive rates on triggered vs clean windows — this is a short
  step).
- Backdoor success → **effect on the maintenance decision** (a masked failure =
  a skipped inspection = elevated in-service failure risk).
- Defense recovery → **reduction in that risk** (Krum/SHARP restore the
  decision quality).
- Optionally frame as degradation/restoration of a **fleet reliability metric**.

> **Recommendation:** add one results subsection ("Reliability consequences of
> the attack and defense") + one figure. This is likely the difference between
> desk-reject and review.

---

## 5. Claim-softening plan (before → after)

| Current / draft claim | Softened, defensible version |
| --- | --- |
| "First backdoor attack on federated RUL." | "**To our knowledge, the first systematic study** of stealthy, failure-masking backdoors on federated aeroengine prognostics" (Tallat '26, Li '26 study *untargeted/scaling* poisoning, not backdoors). |
| "We propose SHARP, a novel defense." | "We **adapt** personalized + Byzantine-robust stacking (cf. SARS '24; Fan & Chen '26) **to the prognostics setting**." — domain-first, not method-novel. |
| "Personalization is vulnerable to backdoors." | "We **confirm, in the prognostics domain**, the general-FL finding that personalization alone does not confer robustness (Fan & Chen '26)." |
| "Krum is the best defense." | "Among the aggregators tested, Krum is the most robust **on this benchmark**." |
| "The stacked defense beats Krum." | "Stacking **preserves** Krum-level robustness **while retaining** the personalization accuracy gain." (Statistically ≈ Krum-alone.) |
| "Our method generalizes." | "Results **hold across the client counts and operating regimes we tested**." (No claim beyond C-MAPSS.) |
| Any implied SOTA accuracy. | Drop entirely; state accuracy is not the objective. |

---

## 6. Literature study — enrichment REQUIRED

The current [`paper_list.md`](paper_list.md) is excellent but **ML-venue-heavy**
(NeurIPS / ICML / USENIX / AISTATS). For RESS we need a **reliability-literature
backbone** so the paper speaks the venue's language. Targeted additions to find:

1. **RESS-native PHM / RUL / prognostics papers** (last ~3 yrs) — to cite,
   position against, and show we know the reliability literature (Zio, Baraldi,
   Lei, Si, et al.).
2. **ML / deep learning for reliability & maintenance in RESS specifically** —
   demonstrates ML is an accepted RESS method and gives us "home-venue" anchors.
3. **Trustworthy / secure / adversarial ML in safety-critical or reliability
   contexts** — the bridge between our security angle and RESS.
4. **Reliability-consequence / risk-of-prognostic-error / cost-of-missed-
   maintenance** framing — to support §4.
5. **Collaborative / cross-fleet prognostics in industry** — to support the
   practical-motivation requirement (Issue 3).

> **Action:** run a RESS-focused literature sweep (ScienceDirect/RESS + Google
> Scholar) and fold ~8–15 reliability-venue references into a revised
> `paper_list.md` (new section: "Reliability-venue anchors").

---

## 7. Action checklist (pre-draft)

- [ ] Reframe abstract + intro around the **safety hazard of fleet prognostics**; method as vehicle.
- [ ] Add a **reliability-consequence analysis** (ASR → missed failures / risk) + figure (§4).
- [ ] **Soften** all novelty claims per §5; cite SARS '24 and Fan & Chen '26.
- [ ] **Motivate** the threat model (compromised/insider client, supply chain) and the practical multi-operator FL scenario.
- [ ] **Pre-empt** the accuracy question in one sentence.
- [ ] **Enrich** the literature with RESS-native reliability references (§6).
- [ ] Redraw the overview as a **formal system diagram** (Fig 1), not the analogy version.
- [ ] Decide venue confidence: RESS is defensible **with** the above; without §1/§4 it is a stretch.

---

## 8. Bottom line

The work is solid and the topic (fleet prognostics safety) is in RESS scope via
fault-diagnosis / maintenance / life-extension. **But the paper currently speaks
"ML"; RESS speaks "reliability & safety."** The gap is closed by (a) reframing,
(b) a reliability-consequence analysis, (c) softened claims, and (d) a
reliability-literature backbone — **not** by wording tweaks alone. Do these four
and desk-reject risk drops from *meaningful* to *low*.
