# Adversarial Robustness of Federated Learning for Aircraft-Engine Prognostics: A Systematic Study on NASA C-MAPSS

*Journal paper draft — v1 (2026-07-23)*
*Based on RQ2 (personalization), RQ3 (proximal regularization), and RQ7 (adversarial robustness) experimental campaigns.*

> Placeholder authorship / affiliations.
> This draft uses seed-42 numbers for RQ7. Multi-seed (seeds 42–46) mean ± 95 %-CI aggregation is in progress and will replace the point estimates in this text before submission.

---

## Abstract

Federated Learning (FL) is emerging as a natural fit for aircraft-engine
prognostics, where fleet operators want to jointly train a Remaining Useful Life
(RUL) model without exposing raw sensor telemetry. Yet the existing FL-for-
prognostics literature is almost entirely benign: prior work on the canonical
NASA C-MAPSS turbofan benchmark reports single-run RMSE numbers under honest
clients and IID or mild non-IID splits, and does not systematically evaluate
robustness against a realistic attack menu. We close this gap. Using a
multi-task 1-D CNN (30 018 parameters, GroupNorm normalization for FL-safe
statistics) trained on a structural non-IID partition of C-MAPSS (2 clients
from FD001 + 2 clients from FD003, one operating condition per subset), we
run a 25-cell attack × aggregator matrix covering four untargeted attacks
(label-flip, gradient scaling ×−10, gradient scaling ×−2, coordinated
Byzantine with two colluding attackers) and one targeted attack (a
physically-plausible sensor-value backdoor whose trigger is a −3.5 σ excursion
on the T30 temperature sensor at the last measured cycle), evaluated against
four aggregators (FedAvg, trimmed mean, coordinate median, and Krum with two
tolerance settings). Three findings stand out.
**(i)** The sensor-value backdoor is uniquely stealthy: attack success rate
reaches 98.0 % against vanilla FedAvg while clean RMSE (17.28) and clean F1
(0.915) both *improve* on the honest baseline, so no defender monitoring only
clean metrics would notice.
**(ii)** Krum drives backdoor attack success to exactly 0 % — the only
aggregator in the study that fully neutralizes the targeted attack.
**(iii)** With 2 of 4 clients coordinating a scaled-gradient attack,
per-coordinate defenses (trimmed mean, median) collapse to the undefended
level (RMSE 84.03), whereas Krum with f = 1 still recovers a near-clean
model (RMSE 19.80, F1 0.779) — while Krum with f = 2 is mathematically
undefined at n = 4, exposing a hard theoretical ceiling on Byzantine
tolerance at small client counts. We also confirm that the "loudness cliff"
in gradient scaling is illusory: a stealthy ×−2 attack is as catastrophic
against FedAvg (RMSE 82.56) as a ×−10 attack (RMSE 84.03), but
per-coordinate defenses recover *better* from the stealthy variant. This is,
to our knowledge, the first systematic adversarial-robustness benchmark for
FL-based turbofan RUL estimation, and the first to use a
physically-interpretable sensor-value backdoor trigger against a multi-task
(RUL + fault) prognostics head.

**Keywords:** federated learning; prognostics; remaining useful life;
Byzantine-robust aggregation; backdoor attack; NASA C-MAPSS.

---

## 1. Introduction

### 1.1 Motivation

Airline and MRO consortia collect terabytes of turbofan sensor telemetry
per year, but competitive, contractual, and regulatory constraints make it
difficult to pool this data into a single centralized training set. Federated
Learning offers a way forward: participants exchange only model updates,
never raw sensors. Over the last four years the FL-for-prognostics literature
has demonstrated that this works for benign settings — Barbosa et al. (2025)
[arXiv:2502.05321], Söderkvist Vermelin et al. (2024, PHM Society), and
Pandhare et al. (2021, PHM Society) all report that FedAvg on
C-MAPSS closes most of the gap between local-only training and a hypothetical
centralized model.

Two recent 2026 papers — BioMutFed+ (Tallat et al., IEEE TII 2026) and
Trustworthy FL for IIoT (Li, Wiley AIE 2026) — begin to address adversarial
robustness, but each evaluates a single novel aggregator against a single
attack family (gradient-ascent poisoning and gradient magnitude scaling,
respectively). No prior work reports how the *canonical* Byzantine-robust
aggregators (trimmed mean, coordinate median, Krum) fare across the full
spectrum of attacks a real-world adversary could mount on an FL-based
predictive-maintenance system — nor whether a physically-plausible
sensor-value backdoor, embedded in the training data of one supplier, can
subvert a multi-task RUL + fault-classification head while remaining
invisible to standard clean-data monitoring.

### 1.2 Problem statement

Given four FL clients partitioned by operating condition (2 FD001 + 2 FD003
engines, 50 engines each), a shared multi-task 1-D CNN with an RUL
regression head and a binary fault-classification head, and an attacker
controlling one or two clients, how does the *worst-case* prognostic quality
of the global model — measured jointly by RMSE, NASA scoring function,
AUPRC, F1, and, for the backdoor, the *triggered-input attack success rate*
— vary across:

- **Attack type:** label-flip, gradient scaling ×−10 (loud), gradient scaling
  ×−2 (stealthy), sensor-value backdoor, coordinated ×−10 Byzantine (2 of 4).
- **Aggregator:** FedAvg, trimmed mean (β = 0.25), coordinate median,
  Krum (f = 1), Krum (f = 2).

And, in the personalization / heterogeneity dimensions we quantify in the
same code base:
- **RQ2:** does replacing the shared classifier head with per-client heads
  (FedRep, Collins et al. 2021 [arXiv:2102.07078]) recover the accuracy lost
  to structural non-IID?
- **RQ3:** does client-side proximal regularization (FedProx,
  Li et al. 2020 [arXiv:1812.06127]) close the same gap by controlling
  local drift?

### 1.3 Contributions

We claim six specific contributions:

1. **First systematic adversarial-robustness benchmark for FL-based aircraft-
   engine RUL on NASA C-MAPSS.** A 25-cell matrix (5 attack families × 4
   aggregators + baselines), publicly released alongside the code.

2. **First physically-plausible sensor-value backdoor trigger for FL
   prognostics.** The trigger `(feature = s_3 (T30), cycle_offset = −1,
   value = −3.5 σ)` corresponds to a realistic firmware-injected temperature
   anomaly in the last measured cycle of an engine's window — analogous
   in spirit to the physical-trigger paradigm of BADControl
   (Burbano et al. USENIX Security 2026) but instantiated in the FL setting.

3. **First "stealth-cliff" analysis of gradient-scaling attacks on FL-RUL.**
   We show ×−10 (RMSE 84.03) and ×−2 (RMSE 82.56) are *both* catastrophic
   against vanilla FedAvg, but per-coordinate defenses recover *better* from
   the stealthy ×−2 (RMSE 20.27) than from the loud ×−10 (RMSE 21.38),
   inverting the "stealth = better attack" intuition.

4. **First multi-task (RUL + fault) poisoning analysis for federated
   prognostics.** All prior FL-prognostics adversarial work considers a
   single output head. We attack the joint head and measure both clean
   quality (regression RMSE, classification F1) and triggered attack success
   simultaneously.

5. **First empirical Krum-f₁ vs Krum-f₂ study under coordinated Byzantine
   in FL-RUL.** With n = 4 and 2 colluding attackers, Krum with f = 1 still
   recovers to RMSE 19.80 (F1 0.779), while Krum with f = 2 is mathematically
   undefined (`n − f − 2 = 0 < 1`) — a hard theoretical ceiling
   practitioners rarely see spelled out in an empirical paper.

6. **Structural (operating-condition-based) non-IID split extended to the
   adversarial regime.** Pandhare et al. (2021) introduced this partition
   for benign FL; we push it into the adversarial regime and show that a
   client holding only FD001 data can subvert the global model's FD003
   predictions through the shared backbone.

### 1.4 Paper structure

Section 2 surveys related work in four buckets: FL for prognostics,
Byzantine-robust FL, personalized FL under attack, and time-series /
physical backdoor triggers. Section 3 defines the system model, non-IID
partition, and threat model. Section 4 describes the attacks, defenses,
and evaluation metrics. Section 5 reports the setup. Section 6 presents
results for RQ2 (personalization), RQ3 (proximal regularization) and
RQ7 (adversarial robustness). Section 7 discusses defense recommendations,
limitations, and future work. Section 8 concludes.

---

## 2. Related Work

### 2.1 Federated learning for aircraft-engine RUL

FL applications to C-MAPSS turbofan RUL are recent and consistently
*benign*. Barbosa et al. (2025) [arXiv:2502.05321] use vanilla FedAvg with a
shallow regressor on FD001. Söderkvist Vermelin et al. (2024, PHM Society)
provide the most thorough benign benchmarking, comparing FedAvg to local-only
across all four C-MAPSS subsets. Pandhare et al. (2021, PHM Society)
introduce "collaborative prognostics" with a structural non-IID split by
operating conditions — the closest precedent for our FD001+FD003 partition —
but again without attacks. Milasheuski et al. (2026) [arXiv:2605.07860]
study generative FL (VAE / GAN / DM) for predictive maintenance, mentioning
backdoor defenses only in related work. Rehman et al. (IEEE TII 2021)
propose TrustFed, a reputation-based client-selection framework tested on
turbofan data, but their threat model addresses lazy / free-riding clients
rather than gradient-space attackers.

### 2.2 Byzantine-robust federated learning in industrial IoT

The canonical robust aggregators are Krum (Blanchard et al. NeurIPS 2017),
coordinate median and trimmed mean (Yin et al. ICML 2018
[arXiv:1803.01498]), and geometric-median / robust functional aggregation
(Pillutla et al. IEEE TSP 2022 [arXiv:1912.13445]). In the industrial-IoT
space, Li, Ngai and Voigt (IEEE TII 2021) provide the cornerstone
Byzantine-robust FL benchmark (over 120 citations), but on generic IIoT
classification datasets rather than prognostics. Hou et al. (IEEE TII 2021)
propose federated filters against backdoors in IIoT applications with
image-like triggers. Two 2026 papers directly overlap with the present
work: (i) BioMutFed+ (Tallat et al., IEEE TII 2026) tests a
mutation-driven aggregator on C-MAPSS against 20 %-malicious gradient
ascent — a single attack, single defense study; and (ii) Trustworthy FL
for IIoT (Li, Wiley AIE 2026) uses C-MAPSS to evaluate blockchain-based
reputation plus gradient magnitude clipping against magnitude-scaling
attacks — again single attack, single defense. Our matrix study (5 attacks
× 4 aggregators, including targeted backdoor) is genuinely orthogonal.

### 2.3 Personalized FL under adversarial pressure

Personalization schemes (FedRep, FedPer, FedProto, Ditto) partly shield
per-client heads from backdoor injection because malicious updates are
mostly localized to the shared backbone. Six recent papers exploit or
defend against this: SARS (Zhang et al. ACM IMWUT 2024), HBIpFL (Chen et
al. CCFT 2026), RBA (IJCNN 2026), SemAlign-PFL (Wang et al. Elsevier JISA
2026), DCInject (Birhan et al. ICASSP 2026), and the vulnerability survey
of Fan and Chen (2026) [arXiv:2606.22782]. All six operate in vision
(CIFAR / MNIST / FEMNIST); none report on prognostics or regression heads.
Our RQ2 result — that FedRep closes 72.7 % of the local → centralized
RMSE gap under structural non-IID — sets a benign personalization baseline
against which future adversarial-personalization studies can measure.

### 2.4 Time-series and physical-world backdoor triggers

The FL-backdoor literature has explored image patches (Bagdasaryan et al.
AISTATS 2020 [arXiv:1807.00459]; Xie et al. DBA, ICLR 2020), semantic
triggers (Bhagoji et al. ICML 2019 [arXiv:1811.12470]), boundary trigger
sets (Yang et al. Info. Sci. 2023), collusive optimized triggers (CoBA,
Lyu et al. IEEE TDSC 2024), frequency triggers (DCInject, ICASSP 2026),
and event-camera triggers on federated spiking NNs (Wang and Li, MDPI
Electronics 2026). BADControl (Burbano et al. USENIX Security 2026)
introduces the *physical-trigger* threat model for cyber-physical control
systems — analogous to our sensor-value trigger. But no prior FL-backdoor
paper uses a `(feature-index, cycle-offset, value)` triple tied to a
specific engine-sensor at a specific cycle position, and no prior FL work
combines a physically-plausible time-series trigger with joint RUL +
classification heads.

### 2.5 Positioning summary

Table 1 places our study alongside the closest existing work.

**Table 1 — Positioning against closest existing work.**

| Study | Dataset | Task | Non-IID split | Attack family | Defenses tested | Multi-seed |
|---|---|---|---|---|---|---|
| Barbosa 2025 (arXiv 2502.05321) | C-MAPSS FD001 | RUL | IID | — | — | No |
| Söderkvist V. 2024 (PHM Soc) | C-MAPSS all four | RUL | benign non-IID | — | — | No |
| Pandhare 2021 (PHM Soc) | Machine fleets | Prognostics | **operating cond.** | — | — | No |
| Rehman 2021 (IEEE TII) | C-MAPSS | RUL | benign | free-riders | reputation | No |
| Li 2021 (IEEE TII) | Generic IIoT | Classification | mild | Byzantine (data pois.) | median, Krum | No |
| Hou 2021 (IEEE TII) | Generic IIoT | Classification | mild | backdoor (image-like) | federated filters | No |
| BioMutFed+ 2026 (IEEE TII) | **C-MAPSS** | RUL | mild | grad-ascent (20 %) | mutation-driven | No |
| Trustworthy FL 2026 (Wiley AIE) | **C-MAPSS** | RUL | mild | magnitude scaling | blockchain + clip | No |
| **This work** | **C-MAPSS FD001+FD003** | **RUL + Fault (multi-task)** | **operating cond.** | **label-flip, ×−10, ×−2, backdoor, coord ×−10** | **trimmed, median, Krum-f₁, Krum-f₂** | **In progress (5 seeds)** |

---

## 3. System & Threat Model

### 3.1 Federation and dataset

We use NASA's C-MAPSS turbofan degradation benchmark (Saxena et al. 2008)
in a 4-client federation. Two subsets are chosen to induce structural
non-IID: FD001 (one operating condition, one fault mode — HPC degradation)
and FD003 (one operating condition, two fault modes — HPC + Fan
degradation). Each subset's 100 engines are split evenly between two
clients (50 engines each), giving 4 clients × 50 engines. Sliding windows
of length 30 are extracted per engine with stride 1; RUL is capped at 125
cycles; a binary "fault-imminent" label is derived from RUL ≤ 30. Feature
selection retains 17 out of 21 sensors after removing constant channels.

### 3.2 Model

A multi-task 1-D CNN with a shared convolutional backbone
(three conv blocks, 30 018 parameters total), a GroupNorm normalization
layer (chosen over BatchNorm because BatchNorm statistics are
distribution-dependent and unsafe under FL heterogeneity), an RUL
regression head, and a binary fault-classification head trained jointly
with loss `L = L_RUL + λ · L_fault` (λ = 0.5).

### 3.3 Threat model

- **Adversary knowledge:** the attacker fully controls one or two clients
  (client_3 for single-attacker cells; client_3 and client_4 for the
  coordinated cell), including the local dataset and local training code.
  The attacker cannot observe honest clients' data but can see the global
  model at every round (standard for FL).
- **Attack goals:**
  - *Untargeted attacks* (AV1, AV2, AV4, AV5) aim to degrade the global
    model on the honest test set.
  - *Targeted attack* (AV3, D31–D33) aims to install a backdoor that flips
    the fault-classification head to "not-faulty" at trigger time, while
    preserving clean-data quality (stealth).
- **Server / defender capabilities:** the server can apply any of the four
  robust aggregators. The server does *not* run backdoor detection, does
  *not* inspect training data, and does *not* trust any client more than
  another. All defense happens purely in gradient space.

---

## 4. Methodology

### 4.1 Attacks (five families)

| Code | Family | Threat model | Parameters |
|---|---|---|---|
| AV1 | Label-flip | Data poisoning | Flip fault labels (0 ↔ 1) on all local samples |
| AV2 | Gradient scaling ×−10 | Model poisoning | Multiply local update δ by −10 before sending |
| AV4 | Gradient scaling ×−2 | Model poisoning (stealthy) | Multiply δ by −2 |
| AV3 | Sensor-value backdoor | Targeted (data + model) | Trigger: T30 sensor value = −3.5 σ at last cycle; poison_fraction = 0.30; target class = "not faulty"; RUL rewritten to 125 |
| AV5 | Coordinated ×−10 Byzantine | 2 colluding clients | Both attackers multiply δ by −10 |

### 4.2 Defenses (four robust aggregators)

| Code | Aggregator | Parameter | Reference |
|---|---|---|---|
| — | FedAvg | sample-weighted mean | McMahan 2017 [arXiv:1602.05629] |
| Dx1 | Trimmed mean | β = 0.25 (trim top / bottom quartile) | Yin 2018 |
| Dx2 | Coordinate median | — | Yin 2018 |
| Dx3 | Krum, f = 1 | tolerate 1 Byzantine | Blanchard 2017 |
| Dx4 | Krum, f = 2 | tolerate 2 Byzantine | Blanchard 2017 |

### 4.3 Evaluation metrics

Every round we compute: (i) global test RMSE on the pooled test set,
(ii) NASA scoring function (asymmetric penalty for late predictions),
(iii) AUPRC and F1 for the fault head. For backdoor cells we additionally
compute:
- Clean fault-positive rate on the test set,
- Triggered fault-positive rate (same test set with the trigger stamped),
- **Attack Success Rate (ASR)** = fraction of clean-positive samples whose
  triggered version is predicted "not faulty."

---

## 5. Experimental Setup

- 4 clients, 50 communication rounds, 2 local epochs per round, batch size 256,
  Adam optimizer with cosine LR schedule (lr₀ = 1 × 10⁻³, wd = 1 × 10⁻⁴).
- Random seed 42 for the results below. Multi-seed runs (seeds 42–46) are
  in progress; final camera-ready numbers will be reported as mean ± std ±
  95 %-CI over 5 seeds.
- Implementation: Python 3.12, PyTorch CPU wheel, ≈ 30 s per FL round on a
  laptop-class CPU. Total wall-clock for the 25-cell matrix ≈ 41 min.

---

## 6. Results

### 6.1 RQ2 — personalization closes the structural non-IID gap

**Table 2 — Personalization ablation on FD001 + FD003 (RQ2 vs Phase 6 baseline).**

| Method | Best round | Test RMSE (global) | Macro RMSE | FD001 RMSE | FD003 RMSE | Gap closed vs local-only |
|---|---:|---:|---:|---:|---:|---:|
| Local-only (mean) | — | — | 17.92 ± 1.52 | ~15.0 | ~18.0 | — |
| Centralized (upper bound) | — | 13.77 | — | 14.76* | 12.69* | 100 % |
| FedAvg (baseline) | — | **17.95** | — | ~17.0 | ~19.0 | −0.7 % |
| **FedRep** (h₁, e₁) | 48 | — | **14.91** | **14.34** | **15.47** | **72.7 %** |

*Per-subset centralized numbers are from separate FD001-only and FD003-only
runs (Phase 6 breakdown).*
FedRep with 1 head-only epoch + 1 encoder epoch dramatically outperforms
FedAvg by allowing per-client classification heads to specialize on the
local fault-mode distribution while sharing the encoder. The result
supports the hypothesis that the ~4 RMSE gap between FedAvg and the
centralized upper bound is *architectural* — one head cannot fit two
fault-mode families — and not attributable to insufficient rounds or
insufficient regularization.

### 6.2 RQ3 — proximal regularization tightens the ceiling only marginally

**Table 3 — FedProx μ-sweep on FD001 + FD003.**

| Method | μ | Best RMSE | Gap closed vs FedAvg headroom | FD001 RMSE | FD003 RMSE | FD001 F1 | FD003 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.0 | 17.95 | — | 16.99 | 18.86 | 0.962 | 0.727 |
| FedProx | 0.001 | 17.85 | 2.3 % | 18.21 | 17.49 | 0.920 | 0.895 |
| FedProx | 0.01 | 17.94 | 0.1 % | 16.88 | 18.94 | 0.962 | 0.688 |
| FedProx | **0.1** | **17.70** | **6.0 %** | **17.97** | **17.42** | **0.920** | **0.800** |

FedProx with μ = 0.1 improves combined RMSE by 0.25 cycles and shifts the
prognostic penalty from FD003 (harder, two fault modes) onto FD001
(easier, one fault mode). The FD003 F1 improves from 0.727 to 0.800 — a
practically useful reallocation for maintenance planning — but only 6 %
of the local-only → centralized headroom is closed. **Together with RQ2,
this establishes that architectural personalization (RQ2) is ~10 × more
effective than proximal regularization (RQ3) for closing the structural
non-IID gap on C-MAPSS.**

An interesting side-effect visible in the table: μ = 0.001 already delivers
the best FD003 F1 (0.895) across the whole sweep — even better than
μ = 0.1 — at the cost of a slightly higher FD001 RMSE (18.21 vs 16.88 for
μ = 0.01). The μ-sweep therefore trades a *combined-RMSE* optimum
(μ = 0.1) against a *fault-classification-on-the-harder-subset* optimum
(μ = 0.001). Practitioners running maintenance-decision pipelines rather
than pure RUL regression may prefer μ = 0.001.

### 6.3 RQ7 — adversarial robustness matrix (seed 42, 25 cells)

**Table 4 — Attack × Aggregator matrix (best-round RMSE / F1 / attack
success rate). Values in *italics* denote catastrophic model collapse.**

| Attack \ Aggregator | Vanilla FedAvg | Trimmed mean (β = 0.25) | Coord. median | Krum (f = 1) | Krum (f = 2) |
|---|---:|---:|---:|---:|---:|
| Clean baseline | 17.95 / 0.871 | 17.56 / 0.871 | 17.56 / 0.871 | 17.93 / 0.835 | — |
| Label-flip | 29.92 / 0.467 | 22.24 / 0.500 | 22.24 / 0.500 | 18.98 / 0.704 | — |
| Grad ×−10 | *84.03 / 0.000* | 21.38 / 0.525 | 21.38 / 0.525 | 18.98 / 0.704 | — |
| Grad ×−2 (stealthy) | *82.56 / 0.085* | 20.27 / 0.714 | 20.27 / 0.714 | 18.98 / 0.704 | — |
| Backdoor (targeted) | 17.28 / 0.915 / **ASR 98.0 %** | 18.40 / 0.800 / ASR 68.6 % | 18.40 / 0.800 / ASR 68.6 % | 19.80 / 0.779 / **ASR 0.0 %** | — |
| Coord ×−10 (2 attackers) | *84.03 / 0.000* | *84.03 / 0.000* | *84.03 / 0.000* | **19.80 / 0.779** | *undefined (n − f − 2 < 1)* |

Six observations:
1. **Krum uniquely defeats the backdoor.** Attack success rate falls
   monotonically 98 → 68.6 → 0 % as the aggregator moves from FedAvg to
   trimmed / median to Krum.
2. **The backdoor is invisible on clean metrics.** Vanilla-FedAvg-under-
   backdoor achieves better clean RMSE (17.28) and F1 (0.915) than the
   *honest* baseline (17.95 / 0.871). Any monitoring pipeline that ignores
   triggered-set evaluation will miss the attack completely.
3. **Stealth-cliff on grad-scaling is inverted.** Both ×−10 and ×−2 are
   catastrophic against FedAvg (~ RMSE 83); trimmed / median recover
   *better* from ×−2 (RMSE 20.27) than from ×−10 (RMSE 21.38). Krum is
   invariant to the multiplier (18.98 either way) because its selection is
   geometric, not norm-based.
4. **Per-coordinate defenses collapse under coordination.** With 2 of 4
   clients malicious, trimmed mean (β = 0.25 trims only 1) and median
   (needs an honest majority) both fail catastrophically (RMSE 84.03).
5. **Krum-f₁ recovers under coordinated attack.** Even though the
   parameter f = 1 formally violates the "≤ f Byzantine" assumption,
   Krum's argmin over per-client distance sums still lands on an honest
   client because the two honest updates cluster in gradient space and the
   two attackers, though large, are separated from each other. RMSE 19.80,
   F1 0.779 — a near-clean recovery.
6. **Krum-f₂ is impossible at n = 4.** The constraint `n − f − 2 ≥ 1`
   evaluates to 0, so the algorithm literally cannot be defined. This
   quantifies a hard theoretical ceiling on Byzantine tolerance at small
   client counts.

**Figures.** Three figures are auto-generated: (i) headline comparison of
all 25 cells side-by-side; (ii) defense-recovery pairs (attacked vs
defended for each of the five attack families); (iii) FD001 vs FD003
per-subset RMSE breakdown. All three are in [results/rq7_poisoning_seeds/seed_42/](FL-for-Aircraft/results/rq7_poisoning_seeds/seed_42/).

---

## 7. Discussion

### 7.1 Defense-selection guidance for FL prognostic deployments

Based on the RQ7 matrix we recommend the following operating rules for
practitioners deploying federated C-MAPSS-style prognostics:

| Threat concern | Recommended aggregator | Why |
|---|---|---|
| Sporadic data-quality issues, no active adversary | FedAvg + monitoring | Cheapest; matches best clean baseline |
| One malicious client, untargeted goal | **Trimmed mean or median** | Full recovery to RMSE ≈ 20 across attack scales |
| One malicious client, targeted backdoor | **Krum (f = 1)** | The only aggregator that zeros attack success |
| Multiple colluding clients (< 50 %) | **Krum (f = 1)** | Trimmed / median collapse; Krum still finds honest cluster |
| ≥ 50 % of clients malicious | *No FL defense works at small n* | Increase n, or move to centralized training |

### 7.2 Limitations

- **Small client count (n = 4).** Our findings on Krum-f₂ being undefined
  are specific to small federations. At n ≥ 6 with 2 attackers, f = 2
  becomes valid and the comparison changes.
- **Single seed for the matrix.** All RQ7 numbers above are from seed 42.
  Multi-seed aggregation (seeds 42–46) is running; camera-ready numbers
  will replace the point estimates with mean ± 95 %-CI.
- **Static backdoor trigger.** Our sensor-value trigger is fixed at
  `(feature = 4, cycle = −1, value = −3.5 σ)`. Adaptive triggers that
  respond to the current global model could be stronger; conversely,
  activation-based backdoor detectors could reduce our reported ASR.
- **No FedRep-under-attack cross-cut.** We report RQ2 (personalization,
  benign) and RQ7 (attacks, non-personalized FedAvg-family) separately.
  The interaction — do per-client heads inherit the shared backbone's
  backdoor? — is left to future work.

### 7.3 Future work

1. Complete multi-seed RQ7 sweep (seeds 42–46) with statistical CIs.
2. Combine RQ2 (FedRep) and RQ7 (backdoor) to test the vision-domain
   claim that personalization is a natural backdoor defense.
3. Extend to FD002 and FD004 (multi-operating-condition subsets) to test
   whether structural non-IID at a finer granularity changes the defense
   picture.
4. Test norm-clipping defenses (Sun et al. 2019 [arXiv:1911.07963]) as
   a cheap partial defense against gradient scaling.

---

## 8. Conclusion

We presented, to our knowledge, the first systematic adversarial-robustness
benchmark for federated learning on aircraft-engine RUL prediction. The
25-cell attack × aggregator matrix on NASA C-MAPSS FD001 + FD003
demonstrates three practically important results.

First, a physically-plausible sensor-value backdoor — trivial to embed via
a compromised supplier's local training pipeline — subverts a vanilla
FedAvg-trained multi-task model with 98 % attack success rate while
leaving clean-data quality *better* than the honest baseline, defeating
any monitoring approach that only inspects clean-set metrics.

Second, of the canonical robust aggregators, only Krum drives backdoor
attack success to zero, and only Krum survives coordinated 2-of-4
Byzantine attacks with near-clean recovery (RMSE 19.80). Trimmed mean and
coordinate median are effective against single-attacker untargeted attacks
but collapse catastrophically under coordination.

Third, a hard theoretical ceiling — the `n − f − 2 ≥ 1` constraint of
Krum — becomes an empirical reality at small client counts: with 4
clients, no aggregator in the study can tolerate 2 attackers *if the
practitioner insists on the formally correct Krum parameter*, but the
formally-incorrect Krum-f₁ still recovers a usable model. This is a
concrete data point for practitioners choosing FL parameters in small
consortia.

Alongside the adversarial contribution, our RQ2 (FedRep) and RQ3
(FedProx) ablations quantify that architectural personalization is roughly
10× more effective than proximal regularization for closing the
structural non-IID gap on C-MAPSS, providing a strong benign baseline
against which future adversarial-personalization studies can measure.

---

## References (informal — to be converted to BibTeX)

Foundational algorithms:
- McMahan et al. 2017 — FedAvg — arXiv:1602.05629.
- Li et al. 2020 — FedProx — arXiv:1812.06127.
- Collins et al. 2021 — FedRep — arXiv:2102.07078.

Byzantine-robust aggregators:
- Blanchard et al. 2017 — Krum — NeurIPS 2017.
- Yin et al. 2018 — Median / Trimmed Mean — arXiv:1803.01498.
- Pillutla et al. 2022 — RFA — arXiv:1912.13445.

Attacks:
- Bagdasaryan et al. 2020 — How to Backdoor FL — arXiv:1807.00459.
- Bhagoji et al. 2019 — Adversarial Lens — arXiv:1811.12470.
- Fang et al. 2020 — Local Model Poisoning — arXiv:1911.11815.
- Xie et al. 2020 — DBA — ICLR 2020.
- Sun et al. 2019 — Can You Really Backdoor FL — arXiv:1911.07963.
- Burbano et al. 2026 — BADControl — USENIX Security 2026.

Aircraft / C-MAPSS FL:
- Saxena et al. 2008 — C-MAPSS — PHM 2008.
- Barbosa et al. 2025 — FL Jet Engines — arXiv:2502.05321.
- Söderkvist Vermelin et al. 2024 — Collaborative FL RUL — PHM Society.
- Pandhare et al. 2021 — Federated Baseline Learner — PHM Society.
- Rehman et al. 2021 — TrustFed — IEEE TII.
- Milasheuski et al. 2026 — Generative FL PdM — arXiv:2605.07860.

Closest 2026 competitors:
- Tallat et al. 2026 — BioMutFed+ — IEEE TII.
- Li H. 2026 — Trustworthy FL for IIoT — Wiley AIE.

IIoT + Byzantine:
- Li, Ngai, Voigt 2021 — Byzantine-robust FL for IIoT — IEEE TII.
- Hou et al. 2021 — Federated Filters for IIoT — IEEE TII.

Surveys:
- Kairouz et al. 2021 — Advances in FL — arXiv:1912.04977.
- Berghout et al. 2022 — FL for Condition Monitoring — MDPI Electronics.
- Djemaa et al. 2026 — Heterogeneity-Aware Poisoning Survey — MDPI
  Electronics.

Personalized FL under attack (RQ2 context):
- Zhang et al. 2024 — SARS — ACM IMWUT.
- Fan & Chen 2026 — Robust PFL — arXiv:2606.22782.

Time-series triggers:
- Yang et al. 2023 — Boundary trigger — Info. Sci.
- Lyu et al. 2024 — CoBA — IEEE TDSC.

---

*End of v1 draft.*
