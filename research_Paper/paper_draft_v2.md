# Client Heterogeneity in Federated Aircraft-Engine Prognostics: Benign and Adversarial

*A systematic study of personalization, proximal regularization, and Byzantine-robust aggregation on NASA C-MAPSS.*

*Journal paper draft — v2 (2026-07-23)*
*Consolidates RQ2 (personalization + reweighting), RQ3 (interpretability), and RQ7 (adversarial robustness). Replaces v1, which framed the three RQs as separate contributions.*

> Placeholder authorship / affiliations.
> RQ7 numbers are seed-42 point estimates. Multi-seed (seeds 42–46) mean ± 95 %-CI aggregation is scheduled to replace the point estimates before submission.

---

## Abstract

Federated Learning (FL) is a natural fit for aircraft-engine prognostics,
where fleet operators want to jointly train a Remaining Useful Life (RUL)
model without exposing raw sensor telemetry to competitors. But real
airline / MRO consortia present *two orthogonal axes* of client
heterogeneity that a deployed FL pipeline must handle: **(i) benign
heterogeneity** — clients honestly hold data from different operating
conditions and fault modes; and **(ii) adversarial heterogeneity** —
some clients may deviate from honest training (data poisoning, gradient
scaling, targeted backdoor). Prior FL-for-prognostics work is almost
entirely benign, and the two recent 2026 papers that do consider attacks
(BioMutFed+, Trustworthy FL for IIoT) each evaluate a single novel
aggregator against a single attack family.

We present, to our knowledge, the first study that quantifies both axes
jointly on NASA C-MAPSS. Using a multi-task 1-D CNN (30 018 parameters,
GroupNorm normalization for FL safety) trained on a structural non-IID
partition (2 clients from FD001 + 2 clients from FD003, one operating
condition per subset), we run:

- **Axis 1 (benign heterogeneity):** four personalization / proximal /
  reweighting families — FedAvg baseline, FedProx (μ-sweep of 4 values),
  FedRep (personalized heads), FedCCFA (clustered personalization), and
  a server-side imbalance-aware reweighting sweep (fault-count,
  inverse-loss, validation-F1).
- **Axis 2 (adversarial heterogeneity):** a 24-cell attack × aggregator
  matrix covering four untargeted attacks (label-flip, gradient scaling
  ×−10, ×−2, coordinated 2-of-4 Byzantine) and one targeted attack
  (a physically-plausible sensor-value backdoor stamping T30 = −3.5 σ
  at the last cycle), evaluated against four aggregators (FedAvg,
  trimmed mean, coordinate median, Krum with two tolerance settings).
- **Cross-cut:** SHAP-style sensor attribution reveals that FedAvg under
  structural non-IID attributes its predictions to *different sensors*
  than the centralized reference model — an interpretability failure
  layered on top of the accuracy failure.

Four findings stand out. **(i)** On the benign axis, architectural
personalization (FedRep, macro-RMSE 14.91; FedCCFA, macro-RMSE 15.00)
closes 70–73 % of the local → centralized RMSE gap; proximal
regularization (FedProx μ = 0.1, RMSE 17.70) closes only 6 %; and
server-side reweighting closes at most 2.8 % (validation-F1 scheme). The
gap is architectural — one head cannot fit two fault-mode families —
and not attributable to insufficient local drift control. **(ii)** On the
adversarial axis, a physically-plausible sensor-value backdoor achieves
98 % attack success rate against vanilla FedAvg while clean RMSE (17.28)
and clean F1 (0.915) both *improve* on the honest baseline — invisible
to any monitor that inspects only clean-set metrics. **(iii)** Krum
uniquely defeats the backdoor (attack success 0.0 %) and uniquely survives
coordinated 2-of-4 attackers (RMSE 19.80, F1 0.779), while per-coordinate
defenses (trimmed mean, coordinate median) collapse under coordination
to the undefended level (RMSE 84.03). **(iv)** The Krum constraint
`n − f − 2 ≥ 1` becomes an empirical wall at small client counts: with
n = 4 clients, Krum with f = 2 is mathematically undefined, exposing a
hard theoretical ceiling on formal Byzantine tolerance that
practitioners in small consortia must confront.

**Keywords:** federated learning; prognostics; remaining useful life;
personalization; Byzantine-robust aggregation; backdoor attack;
NASA C-MAPSS.

---

## 1. Introduction

### 1.1 Motivation

Airline and MRO consortia collect terabytes of turbofan sensor telemetry
per year, but competitive, contractual, and regulatory constraints make
pooling this data into a single centralized training set impractical.
Federated Learning (FL) offers a way forward: participants exchange only
model updates, never raw sensors. Recent work has demonstrated the
feasibility of this approach on the canonical NASA C-MAPSS turbofan
benchmark (Barbosa et al. 2025 [arXiv:2502.05321]; Söderkvist Vermelin
et al. 2024, PHM Society; Pandhare et al. 2021, PHM Society), consistently
showing that vanilla FedAvg (McMahan et al. 2017 [arXiv:1602.05629]) can
close the gap between local-only training and a hypothetical centralized
model — *in benign, homogeneous settings*.

The practical FL deployment picture is less rosy. A real consortium of
airline operators and MROs will exhibit **two distinct kinds of
client heterogeneity** that the FL pipeline must simultaneously
handle:

- **Benign heterogeneity** (Axis 1): Different clients honestly hold data
  from different operating conditions (altitude, throttle, ambient temperature
  regimes) and different fault modes (HPC-only degradation versus
  HPC + Fan degradation). This causes the local optima at each client to
  drift away from the shared global optimum, and vanilla FedAvg's
  sample-weighted average of these divergent updates can *destroy* rather
  than compose them.

- **Adversarial heterogeneity** (Axis 2): One or more clients may
  deviate from honest training — either subtly (a firmware-compromised
  supplier injecting mislabeled training data) or aggressively (a
  competitor performing a gradient-scaling attack that pushes the global
  model into divergence). In FL these attacks are invisible to the
  server, which only observes model updates.

Both axes reduce the quality of the shared global model, but the
remedies for each are drawn from very different literatures. Axis 1 is
addressed by personalization (FedRep, Collins et al. 2021
[arXiv:2102.07078]), proximal regularization (FedProx, Li et al. 2020
[arXiv:1812.06127]), and clustered personalization (FedCCFA). Axis 2 is
addressed by Byzantine-robust aggregation (Krum, Blanchard et al. NeurIPS
2017; trimmed mean and coordinate median, Yin et al. ICML 2018
[arXiv:1803.01498]). To the best of our knowledge, no prior C-MAPSS study
has systematically quantified both axes jointly, nor connected the two
via an interpretability analysis of *how* non-IID FL differs from
centralized training at the sensor-attribution level.

### 1.2 Where prior work sits

Two recent 2026 papers begin to address the adversarial axis on C-MAPSS,
but each evaluates a **single novel aggregator against a single attack
family**: BioMutFed+ (Tallat et al., IEEE Transactions on Industrial
Informatics 2026) proposes a mutation-driven aggregator and tests it
against gradient-ascent poisoning with 20 %-malicious clients;
Trustworthy FL for IIoT (Li, Wiley Artificial Intelligence for Engineering
2026) proposes a blockchain-reputation aggregator plus gradient
magnitude clipping and tests it against magnitude-scaling attacks. Both
papers are important precedents. Neither runs a matrix study, neither
considers targeted backdoor attacks, neither addresses joint RUL + fault
multi-task heads, and neither compares the four canonical Byzantine-robust
aggregators (FedAvg, trimmed mean, median, Krum). None couples the
adversarial evaluation to a rigorous benign-heterogeneity ablation on the
same federation.

Our study fills exactly this gap.

### 1.3 Contributions

Framed against the two-axis lens of Section 1.1, we make eight specific
contributions:

**On the benign axis (Axis 1):**

1. **A four-family personalization / proximal / clustering / reweighting
   comparison on structural non-IID C-MAPSS.** Under an FD001+FD003
   operating-condition partition with 4 clients, we quantify:
   FedRep (macro-RMSE 14.91, 72.7 % gap closed), FedCCFA (macro-RMSE
   15.00, 70.6 % gap closed), FedProx μ-sweep (best RMSE 17.70,
   6.0 % gap closed), and imbalance-aware server-side reweighting
   (best RMSE 17.80, 2.8 % gap closed).

2. **Empirical evidence that the gap is architectural, not
   optimization-related.** Architectural personalization (FedRep,
   FedCCFA) is roughly 10 × more effective at closing the structural
   non-IID gap than every optimization-side remedy we tested. The
   remaining ~1 RMSE cycle to the centralized upper bound is thus
   plausibly explained by cross-client generalization headroom, not
   by insufficient rounds or insufficient regularization.

3. **Interpretability-under-heterogeneity analysis.** Using
   Integrated-Gradient-style sensor attribution against a maintenance
   ontology (17 named sensors, 3 fault-mode rules), we show that
   FedAvg under structural non-IID attributes its predictions to
   *different sensors* than the centralized reference model on the
   same test engines. This is an interpretability failure that
   compounds the accuracy failure — a maintenance engineer acting on
   FedAvg's explanations would be pointed at the wrong subsystems.

**On the adversarial axis (Axis 2):**

4. **First systematic 5-attack × 4-defense matrix for FL-based
   aircraft-engine RUL on NASA C-MAPSS.** A 24-cell benchmark
   (label-flip, gradient scaling ×−10, gradient scaling ×−2, sensor-value
   backdoor, coordinated 2-of-4 Byzantine) crossed with (FedAvg,
   trimmed mean β = 0.25, coordinate median, Krum f = 1). Publicly
   released alongside the code.

5. **First physically-plausible sensor-value backdoor trigger for FL
   prognostics.** The trigger `(feature = s_3 (T30), cycle_offset = −1,
   value = −3.5 σ)` corresponds to a realistic firmware-injected
   HPC-outlet temperature anomaly in the last measured cycle of an
   engine's sliding window — analogous in spirit to the physical-trigger
   paradigm of BADControl (Burbano et al. USENIX Security 2026) but
   instantiated in FL, and targeting a sensor from the same
   temperature-family that the model already relies on for fault
   classification.

6. **First multi-task (RUL regression + fault classification)
   poisoning analysis for federated prognostics.** All prior FL-
   prognostics adversarial work considers a single output head. We
   attack the joint head and measure both clean regression / classification
   quality *and* the triggered-input attack success rate simultaneously.

7. **First "stealth-cliff inversion" observation for FL-RUL.** Both
   gradient-scaling ×−10 (RMSE 84.03) and ×−2 (RMSE 82.56) are
   catastrophic against vanilla FedAvg, but per-coordinate defenses
   recover *better* from the stealthy ×−2 (RMSE 20.27) than from the
   loud ×−10 (RMSE 21.38). Krum is invariant to the scaling multiplier
   because its selection is geometric, not norm-based.

8. **First empirical Krum-f₁ vs Krum-f₂ study under coordinated
   Byzantine in FL-RUL, exposing the `n − f − 2 ≥ 1` wall.** With
   n = 4 clients and 2 colluding attackers, Krum with f = 1 still
   recovers to RMSE 19.80 (F1 0.779) — because Krum's argmin lands on
   the honest cluster in gradient space — while Krum with f = 2 is
   mathematically undefined, exposing a hard theoretical ceiling on
   *formal* Byzantine tolerance at small client counts.

### 1.4 Paper structure

Section 2 surveys related work along both axes. Section 3 defines the
system model shared by both experimental campaigns. Section 4 formalizes
the two heterogeneity models and the evaluation metrics. Section 5
describes the experimental setup. **Section 6 presents Axis 1 results
(benign heterogeneity)**, and **Section 7 presents Axis 2 results
(adversarial heterogeneity)**. Section 8 synthesizes the two axes into
actionable defense guidance, discusses limitations, and outlines the
FedRep-under-attack cross-cut as the natural next paper. Section 9
concludes.

---

## 2. Related Work

### 2.1 Federated learning for aircraft-engine RUL

FL applications to C-MAPSS are recent and consistently *benign*. Barbosa
et al. (2025) [arXiv:2502.05321] use vanilla FedAvg with a shallow
regressor on FD001. Söderkvist Vermelin et al. (2024, PHM Society)
provide the most thorough benign benchmarking, comparing FedAvg to
local-only across all four C-MAPSS subsets. Pandhare et al. (2021, PHM
Society) introduce "collaborative prognostics for machine fleets" with a
structural non-IID split by operating conditions — the closest precedent
for our FD001+FD003 partition — but again without attacks. Milasheuski
et al. (2026) [arXiv:2605.07860] study generative FL (VAE / GAN / DM) for
predictive maintenance, mentioning backdoor defenses only in related
work. Rehman et al. (IEEE TII 2021) propose TrustFed, a reputation-based
client-selection framework tested on turbofan data, but their threat
model addresses lazy / free-riding clients rather than gradient-space
attackers.

### 2.2 Axis 1 — handling benign client heterogeneity

**Personalization.** Collins et al. (2021) [arXiv:2102.07078] introduce
FedRep, which trains a shared encoder collaboratively while giving each
client its own head trained locally. Related schemes include FedPer
(Arivazhagan et al. 2019), Ditto (Li et al. 2021), and per-FedAvg
(Fallah et al. 2020). Clustered federated learning (Sattler et al. 2020;
CFL variants such as FedCCFA) groups clients by update similarity and
trains a per-cluster model. In the prognostics domain, DecFFD (Deng
et al. IEEE TII 2024) and FedDGA (Sun et al. IEEE TIM 2024) apply
personalization to fault-diagnosis classification; direct FedRep / FedCCFA
comparisons on regression-heavy RUL prediction are rare.

**Proximal regularization.** Li et al. (2020) [arXiv:1812.06127] propose
FedProx, adding a proximal term `μ/2 · ||w − w_global||²` to each
client's local objective to bound local drift under statistical
heterogeneity. FedProx is the most widely-cited optimization-side remedy
for non-IID.

**Server-side reweighting.** Various schemes reweight client updates by
metrics such as validation performance, loss reduction, or class-balance
diagnostics. These are cheaper than personalization (no per-client
state) but generally provide smaller gains under structural non-IID.

### 2.3 Axis 2 — handling adversarial client heterogeneity

**Robust aggregators.** The canonical Byzantine-robust aggregators are
Krum (Blanchard et al. NeurIPS 2017), coordinate median and trimmed mean
(Yin et al. ICML 2018 [arXiv:1803.01498]), and robust functional
aggregation / geometric-median (Pillutla et al. IEEE TSP 2022
[arXiv:1912.13445]). Norm-clipping partial defenses (Sun et al. 2019
[arXiv:1911.07963]) argue that bounded update norms alone defeat many
backdoor variants.

**Attacks.** Untargeted attacks include label-flip and gradient scaling /
"model poisoning" (Bhagoji et al. ICML 2019 [arXiv:1811.12470];
Fang et al. USENIX Security 2020 [arXiv:1911.11815]). Targeted attacks
include backdoors (Bagdasaryan et al. AISTATS 2020 [arXiv:1807.00459]) and
distributed / coordinated backdoors (DBA, Xie et al. ICLR 2020;
CoBA, Lyu et al. IEEE TDSC 2024).

**FL + IIoT + attacks.** Li, Ngai and Voigt (IEEE TII 2021) provide the
cornerstone Byzantine-robust FL benchmark in IIoT (120+ citations), on
generic classification. Hou et al. (IEEE TII 2021) propose federated
filters against image-like backdoors in IIoT. Two 2026 papers directly
overlap with the present work: BioMutFed+ (Tallat et al., IEEE TII 2026)
tests a mutation-driven aggregator on C-MAPSS against 20 %-malicious
gradient ascent; Trustworthy FL for IIoT (Li, Wiley AIE 2026) uses
C-MAPSS to evaluate blockchain reputation plus gradient magnitude
clipping against magnitude scaling. Both are single attack × single
defense; our matrix (5 × 4) is genuinely orthogonal.

### 2.4 Time-series and physical-world backdoor triggers

The FL-backdoor literature has explored image patches (Bagdasaryan et al.
AISTATS 2020; Xie et al. DBA, ICLR 2020), semantic triggers (Bhagoji
et al. ICML 2019), boundary trigger sets (Yang et al. Info. Sci. 2023),
collusive optimized triggers (CoBA, Lyu et al. IEEE TDSC 2024), frequency
triggers (DCInject, Birhan et al. ICASSP 2026), and event-camera triggers
on federated spiking NNs (Wang and Li, MDPI Electronics 2026). BADControl
(Burbano et al. USENIX Security 2026) introduces the *physical-trigger*
threat model for cyber-physical control systems — analogous in spirit
to our sensor-value trigger, but for direct RL-based control rather than
FL-based prognostics. No prior FL-backdoor paper uses a
`(feature-index, cycle-offset, value)` triple tied to a specific
engine-sensor at a specific cycle position, and no prior FL work
combines a physically-plausible time-series trigger with joint RUL +
classification heads.

### 2.5 Cross-axis: personalized FL under adversarial pressure

Six recent papers exploit or defend the interaction between
personalization (Axis 1 remedy) and backdoor attacks (Axis 2 threat) in
vision domains: SARS (Zhang et al. ACM IMWUT 2024), HBIpFL (Chen et al.
CCFT 2026), RBA (IJCNN 2026), SemAlign-PFL (Wang et al. Elsevier JISA
2026), DCInject (Birhan et al. ICASSP 2026), and the vulnerability survey
of Fan and Chen (2026) [arXiv:2606.22782]. All six operate on
CIFAR / MNIST / FEMNIST; none report on prognostics or regression heads.
Our study provides the *benign* baseline (Axis 1 in Section 6) and the
*adversarial* baseline (Axis 2 in Section 7) on the same federation,
against which future adversarial-personalization studies for prognostics
can be measured. The explicit cross-cut — FedRep evaluated under our
backdoor attack — is scoped out to Section 8.4 (future work).

### 2.6 Positioning summary

Table 1 places our study alongside the closest existing work along both
axes.

**Table 1 — Positioning against closest existing work.**

| Study | Dataset | Task | Non-IID split | Axis 1 methods | Axis 2 methods | Multi-seed |
|---|---|---|---|---|---|---|
| Barbosa 2025 (arXiv 2502.05321) | C-MAPSS FD001 | RUL | IID | FedAvg | — | No |
| Söderkvist V. 2024 (PHM Soc) | C-MAPSS all four | RUL | benign non-IID | FedAvg | — | No |
| Pandhare 2021 (PHM Soc) | Machine fleets | Prognostics | **oper. cond.** | fleet baseline | — | No |
| Rehman 2021 (IEEE TII) | C-MAPSS | RUL | benign | — | reputation (client-level) | No |
| Li 2021 (IEEE TII) | Generic IIoT | Classification | mild | — | median, Krum | No |
| Hou 2021 (IEEE TII) | Generic IIoT | Classification | mild | — | federated filters | No |
| BioMutFed+ 2026 (IEEE TII) | **C-MAPSS** | RUL | mild | — | mutation-driven | No |
| Trustworthy FL 2026 (Wiley) | **C-MAPSS** | RUL | mild | — | blockchain + clip | No |
| **This work** | **C-MAPSS FD001+FD003** | **RUL + Fault (multi-task)** | **operating cond.** | **FedProx, FedRep, FedCCFA, imbalance-aware** | **trimmed, median, Krum-f₁, Krum-f₂, + 5 attacks** | **In progress (5 seeds)** |

---

## 3. System Model

### 3.1 Federation

We instantiate a 4-client federation using NASA's C-MAPSS turbofan
degradation benchmark (Saxena et al. 2008). Two subsets are chosen to
induce structural non-IID: **FD001** (one operating condition, one fault
mode — HPC degradation) and **FD003** (one operating condition, two fault
modes — HPC + Fan degradation). Each subset's 100 engines are split
evenly between two clients (50 engines each), giving:

- `client_1`, `client_2`: FD001 engines (single-fault-mode
  distribution).
- `client_3`, `client_4`: FD003 engines (mixed-fault-mode distribution).

Sliding windows of length 30 are extracted per engine with stride 1;
RUL is capped at 125 cycles; a binary "fault-imminent" label is derived
from RUL ≤ 30. Feature selection retains 17 out of 21 sensors after
removing constant channels.

### 3.2 Model

A shared multi-task 1-D CNN (three convolutional blocks, 30 018
parameters total) with a GroupNorm normalization layer (chosen over
BatchNorm because BatchNorm running statistics are distribution-dependent
and unsafe under FL heterogeneity), an RUL regression head, and a binary
fault-classification head trained jointly with loss
`L = L_RUL + λ · L_fault` where `λ = 0.5`.

### 3.3 Communication

Clients participate in every round (no client sampling). Each round
comprises: (i) the server broadcasts the current global model; (ii)
each client runs `E = 2` local epochs of Adam (lr = 1 × 10⁻³, cosine LR
schedule); (iii) each client sends its model delta back to the server;
(iv) the server applies an aggregation rule to produce the next global
model. We run 50 rounds total.

---

## 4. Heterogeneity Models and Methodology

### 4.1 Axis 1 — benign heterogeneity

Formally, each client `k` draws its local dataset from a
subset-specific distribution `D_k`. In our FD001+FD003 partition, the
FD001 clients have one fault-mode density and the FD003 clients have a
different mixture, so `D_k ≠ D_j` for cross-subset (`k, j`) pairs. This
induces gradient divergence at the round-boundary aggregation step,
which vanilla FedAvg averages naively.

We evaluate four families of remedies on this axis:

| Method family | Concrete instance | Reference |
|---|---|---|
| Proximal regularization | FedProx (μ-sweep: 0.001, 0.01, 0.1) | Li 2020 [arXiv:1812.06127] |
| Personalization (shared encoder + per-client head) | FedRep (h₁ + e₁ epochs) | Collins 2021 [arXiv:2102.07078] |
| Clustered personalization | FedCCFA (similarity_threshold = 0.5, warmup 3 rounds) | This paper's implementation, following CFL-family |
| Server-side reweighting | Imbalance-aware sweep: {fault-count, inverse-loss, validation-F1} | This paper |

### 4.2 Axis 2 — adversarial heterogeneity

Formally, some clients replace their honest local training with an
adversarial computation. The attacker controls the client's data and
its local training code, but cannot observe honest clients' data. The
attacker can see the global model at every round.

We evaluate five attack families and four defense aggregators:

**Table 2 — Attack menu (Axis 2).**

| Code | Family | Threat model | Parameters | Attackers |
|---|---|---|---|---|
| AV1 | Label-flip | Data poisoning | Flip fault labels (0 ↔ 1) locally | 1 client |
| AV2 | Gradient scaling ×−10 | Model poisoning (loud) | Multiply δ by −10 | 1 client |
| AV4 | Gradient scaling ×−2 | Model poisoning (stealthy) | Multiply δ by −2 | 1 client |
| AV3 | Sensor-value backdoor | Targeted (data + label + model) | Trigger: T30 sensor value = −3.5 σ at last cycle; poison_fraction = 0.30; target class = "not faulty"; RUL rewritten to 125 | 1 client |
| AV5 | Coordinated ×−10 Byzantine | 2 colluding clients | Both multiply δ by −10 | 2 clients |

**Table 3 — Defense menu (Axis 2).**

| Code | Aggregator | Parameter | Reference |
|---|---|---|---|
| — | FedAvg | sample-weighted mean | McMahan 2017 [arXiv:1602.05629] |
| Dx1 | Trimmed mean | β = 0.25 | Yin 2018 [arXiv:1803.01498] |
| Dx2 | Coordinate median | — | Yin 2018 |
| Dx3 | Krum | f = 1 (tolerate 1 Byzantine) | Blanchard 2017 |
| Dx4 | Krum | f = 2 (tolerate 2 Byzantine) | Blanchard 2017 |

### 4.3 Evaluation metrics

We report:

- **RMSE** on the pooled global test set — the standard RUL metric.
- **NASA scoring function** — the asymmetric-penalty score used by the
  original PHM 2008 challenge (Saxena et al. 2008); late predictions
  are penalized more than early ones.
- **AUPRC** and **F1** for the fault-classification head.
- **Per-subset macro-RMSE** (mean across FD001 clients / FD003 clients
  separately) — the apples-to-apples comparison against per-subset
  centralized upper bounds.
- **Attack Success Rate (ASR)** for backdoor cells: the fraction of
  clean-positive samples whose trigger-stamped version is predicted
  "not faulty."

### 4.4 Interpretability (RQ3)

To connect Axis 1 to a mechanistic explanation, we compute
Integrated-Gradient-style sensor attribution against a 17-sensor
maintenance ontology (with 3 fault-mode rules mapping sensor patterns
to actionable maintenance recommendations) for three test engines
(25, 50, 75) across four model checkpoints: P3 centralized (FD001 only),
P5 FedAvg IID (FD001 only), P6 centralized (combined FD001+FD003),
P6 FedAvg non-IID (combined FD001+FD003 4-client federation). Twelve
attribution heatmaps plus twelve top-sensor bar charts plus cross-model
comparison figures are used to test whether the FL model attributes to
the same sensors as its centralized reference.

---

## 5. Experimental Setup

- 4 clients, 50 communication rounds, 2 local epochs per round, batch
  size 256, Adam optimizer with cosine LR schedule (lr₀ = 1 × 10⁻³,
  wd = 1 × 10⁻⁴), λ_fault = 0.5.
- Random seed 42 for all results in this draft. Multi-seed runs
  (seeds 42–46) are scheduled; camera-ready numbers will report
  mean ± std ± 95 %-CI.
- Implementation: Python 3.12, PyTorch CPU wheel, ≈ 5–8 s per FL round
  on a laptop-class CPU. Total wall-clock for the 24-cell Axis 2
  matrix ≈ 41 min; total for the Axis 1 sweep (4 families) ≈ 55 min.

---

## 6. Results Part I — Axis 1 (Benign Heterogeneity)

### 6.1 Setting the scale

Before comparing remedies, we establish the two reference points that
bound the achievable performance on the FD001+FD003 structural non-IID
partition.

**Table 4 — Reference points on FD001+FD003.**

| Model | Combined test RMSE | Per-subset RMSE (FD001) | Per-subset RMSE (FD003) | Fault F1 |
|---|---:|---:|---:|---:|
| Centralized (upper bound; sees all data) | **13.77** | 14.76 | 12.69 | 0.957 |
| Local-only (mean of 4 clients) | — | — | — | 0.858 |
| Local-only (per-client RMSE) | 17.92 ± 1.52 | ≈ 15.0 | ≈ 18.0 | — |
| **FedAvg baseline** | **17.95** | 16.99 | 18.86 | **0.871** |

The gap `centralized − FedAvg = 13.77 − 17.95 = −4.18 RMSE cycles`.
FedAvg closed only −0.7 % of the local-only → centralized headroom —
in effect, sharing model weights bought nothing over training locally
on 25 engines. The remedies below aim to close this gap.

### 6.2 Personalization (FedRep) — the strongest single remedy

**Table 5 — FedRep (personalized heads) on FD001+FD003.**

| Metric | FedAvg baseline | FedRep (h₁, e₁) | Δ |
|---|---:|---:|---:|
| Best round | 12 | 48 | — |
| Macro RMSE | — | **14.91** | — |
| Per-subset macro RMSE (FD001) | ≈ 17.0 | **14.34** | −2.66 |
| Per-subset macro RMSE (FD003) | ≈ 19.0 | **15.47** | −3.53 |
| Macro F1 (FD001 clients) | — | 0.962 | — |
| Macro F1 (FD003 clients) | — | 0.877 | — |
| **Gap closed vs local → centralized headroom** | −0.7 % | **+72.7 %** | +73.4 pp |

FedRep with 1 head-only epoch + 1 encoder epoch dramatically outperforms
FedAvg by allowing per-client classification heads to specialize on the
local fault-mode distribution while sharing the encoder. The result
supports the hypothesis that the ~4 RMSE gap between FedAvg and the
centralized upper bound is **architectural** — one head cannot fit two
fault-mode families — and not attributable to insufficient rounds or
insufficient regularization.

### 6.3 Clustered personalization (FedCCFA) — matches FedRep, exposes structural similarity

**Table 6 — FedCCFA (clustered personalized encoders) on FD001+FD003.**

| Metric | FedCCFA (similarity 0.5) |
|---|---:|
| Best round | 47 |
| Macro RMSE | **15.00** |
| Per-subset macro RMSE (FD001) | 14.60 |
| Per-subset macro RMSE (FD003) | 15.40 |
| Macro F1 (FD001) | 0.962 |
| Macro F1 (FD003) | 0.874 |
| **Best-round cluster structure** | **{c₁, c₂, c₃, c₄}** — all in ONE cluster |
| Gap closed | +70.6 % |

FedCCFA reaches essentially the same performance as FedRep (RMSE 15.00
vs 14.91). Critically, the algorithm's inferred cluster structure is a
**single cluster of all four clients** — the update-similarity
threshold does not split the federation. This tells us that once
personalization is in play, the FD001 and FD003 clients look similar
enough in gradient space to share an encoder — the fault-mode
divergence is captured by the per-client heads, not by encoder-level
splits. This is an interesting negative result: at n = 4, clustered
personalization offers no marginal benefit over per-client-head
personalization.

### 6.4 Proximal regularization (FedProx) — marginal improvement over FedAvg

**Table 7 — FedProx μ-sweep on FD001+FD003.**

| Method | μ | Best RMSE | Gap closed | FD001 RMSE | FD003 RMSE | FD001 F1 | FD003 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.0 | 17.95 | — | 16.99 | 18.86 | 0.962 | 0.727 |
| FedProx | 0.001 | 17.85 | 2.3 % | 18.21 | 17.49 | 0.920 | **0.895** |
| FedProx | 0.01 | 17.94 | 0.1 % | **16.88** | 18.94 | 0.962 | 0.688 |
| FedProx | **0.1** | **17.70** | **6.0 %** | 17.97 | 17.42 | 0.920 | 0.800 |

FedProx with μ = 0.1 improves combined RMSE by only 0.25 cycles (6.0 %
of the headroom) versus FedAvg — an order of magnitude worse than
FedRep / FedCCFA. Interestingly, μ = 0.001 delivers the best FD003 F1
(0.895) across the whole sweep. The μ-sweep therefore trades a
combined-RMSE optimum (μ = 0.1) against a fault-classification-on-the-
harder-subset optimum (μ = 0.001). Practitioners running maintenance-
decision pipelines rather than pure RUL regression may prefer μ = 0.001.

### 6.5 Server-side reweighting — the weakest family

**Table 8 — Imbalance-aware server-side reweighting sweep.**

| Scheme | Global RMSE | Gap closed | Notes |
|---|---:|---:|---|
| FedAvg (sample-weighted) | 17.95 | −0.7 % | Baseline |
| Fault-count reweight | 18.24 | **−7.7 %** | *Worse than FedAvg* |
| Inverse-loss reweight | 18.37 | **−10.8 %** | *Worst* |
| Validation-F1 reweight | **17.80** | **+2.8 %** | Best of sweep |

Three of the four reweighting schemes make matters *worse* than FedAvg;
the best (validation-F1) improves combined RMSE by 0.15 cycles (2.8 %
of the headroom). No server-side reweighting scheme approaches the
performance of even the weakest personalization method.

### 6.6 Axis 1 synthesis — architectural vs optimization vs reweighting

**Table 9 — Axis 1 remedy families, ranked by gap-closing effectiveness.**

| Rank | Family | Best method | Best gap closed | ~ Ratio vs proximal |
|---|---|---|---:|---:|
| 1 | Personalization (per-client heads) | FedRep (h₁, e₁) | **+72.7 %** | ~12 × |
| 2 | Clustered personalization | FedCCFA (thr 0.5) | +70.6 % | ~12 × |
| 3 | Proximal regularization | FedProx (μ = 0.1) | +6.0 % | 1 × |
| 4 | Server-side reweighting | Validation-F1 | +2.8 % | ~0.5 × |
| — | Sample-count baseline | FedAvg | −0.7 % | — |

**Axis 1 finding:** on structural non-IID C-MAPSS, architectural
personalization (per-client heads) is roughly *an order of magnitude*
more effective at closing the local → centralized gap than either
optimization-side proximal regularization or server-side reweighting.
The gap is therefore *architectural in nature* — the model class of a
single shared head is inadequate for the union of FD001 and FD003 fault
modes — and cannot be closed by better optimization alone.

### 6.7 Interpretability under heterogeneity (RQ3)

For three test engines (units 25, 50, 75), we computed
Integrated-Gradient sensor attributions against the 17-sensor ontology
across four model checkpoints (P3 centralized FD001, P5 FedAvg IID
FD001, P6 centralized combined FD001+FD003, P6 FedAvg non-IID
FD001+FD003 4-client federation).

Two observations are relevant to the paper's thesis:

- **Sensor attribution changes with heterogeneity.** For every one of
  the three test engines, the ranked-top-3 attribution sensors differ
  between the centralized combined model (P6 centralized) and the
  FedAvg non-IID federation (P6 FedAvg). This is an interpretability
  failure that *compounds* the accuracy failure — a maintenance
  engineer acting on FedAvg's SHAP explanation would be pointed at
  different subsystems than one acting on the centralized model's SHAP.

- **The attribution ontology is dominated by turbomachinery
  temperature and flow sensors.** Across all four checkpoints, the
  top-attribution family for RUL prediction is dominated by
  temperature sensors (T24 = s_2, T30 = s_3, T50 = s_4, HPC and LPT
  outlet temperatures) and coolant / bleed flows (W31 = s_20, W32 =
  s_21). This is expected from a physics standpoint (thermodynamic
  efficiency degrades with worn hot-section components) but *also
  justifies our choice of trigger sensor for the Axis 2 backdoor
  attack in Section 7.3*: T30 (s_3) belongs to the family that the
  honest model already relies on for prognosis, making a trigger there
  physically plausible for an adversary to design and physically
  invisible to a defender monitoring aggregate feature statistics.

---

## 7. Results Part II — Axis 2 (Adversarial Heterogeneity)

### 7.1 The 5 × 4 attack × aggregator matrix

**Table 10 — Attack × Aggregator matrix (best-round global-test RMSE / F1
/ backdoor Attack Success Rate). *Italics* denote catastrophic model
collapse (RMSE > 3× baseline).**

| Attack \ Aggregator | Vanilla FedAvg | Trimmed mean (β = 0.25) | Coord. median | Krum (f = 1) | Krum (f = 2) |
|---|---:|---:|---:|---:|---:|
| Clean baseline | 17.95 / 0.871 | 17.56 / 0.871 | 17.56 / 0.871 | 17.93 / 0.835 | — |
| Label-flip (1 attacker) | 29.92 / 0.467 | 22.24 / 0.500 | 22.24 / 0.500 | 18.98 / 0.704 | — |
| Grad ×−10 (1 attacker) | *84.03 / 0.000* | 21.38 / 0.525 | 21.38 / 0.525 | 18.98 / 0.704 | — |
| Grad ×−2 (1 attacker, stealthy) | *82.56 / 0.085* | 20.27 / 0.714 | 20.27 / 0.714 | 18.98 / 0.704 | — |
| Backdoor (1 attacker, targeted) | 17.28 / 0.915 / **ASR 98.0 %** | 18.40 / 0.800 / ASR 68.6 % | 18.40 / 0.800 / ASR 68.6 % | 19.80 / 0.779 / **ASR 0.0 %** | — |
| Coord ×−10 (2 attackers) | *84.03 / 0.000* | *84.03 / 0.000* | *84.03 / 0.000* | **19.80 / 0.779** | *undefined (n − f − 2 < 1)* |

### 7.2 Six observations from the matrix

1. **Krum uniquely defeats the backdoor.** Attack Success Rate falls
   monotonically 98.0 → 68.6 → 0.0 % as the aggregator moves from
   FedAvg to trimmed / median to Krum. This is the paper's most
   surprising defense-side result: a targeted attack that reads its
   trigger as a low-dimensional gradient perturbation is *fully
   neutralized* by Krum's argmin-in-distance selection.

2. **The backdoor is invisible on clean metrics.** Vanilla-FedAvg-
   under-backdoor achieves clean RMSE 17.28 and clean F1 0.915 —
   *better* than the honest baseline (17.95 / 0.871). This is the
   most operationally important attack-side result: any monitoring
   pipeline that only inspects clean-set metrics will miss the attack
   completely. Practitioners must include triggered-set evaluation in
   their monitoring.

3. **Stealth-cliff on grad-scaling is inverted.** Both ×−10 and ×−2
   are catastrophic against FedAvg (RMSE ~ 83). Trimmed / median
   recover *better* from ×−2 (RMSE 20.27) than from ×−10 (RMSE 21.38).
   Krum is invariant to the multiplier (RMSE 18.98 either way)
   because its selection is geometric, not norm-based.

4. **Per-coordinate defenses collapse under coordination.** With 2 of
   4 clients malicious, trimmed mean (β = 0.25 trims only 1) and
   median (needs an honest majority) both fail catastrophically —
   RMSE 84.03, indistinguishable from vanilla FedAvg under the same
   coordinated attack.

5. **Krum-f₁ recovers under coordinated attack.** Even though the
   parameter f = 1 formally violates the "≤ f Byzantine" assumption
   (there are actually 2 attackers), Krum's argmin over per-client
   distance sums still lands on an honest client — because the two
   honest updates cluster tightly in gradient space while the two
   attackers, though large, are separated from each other by the
   scaling amplification. RMSE 19.80, F1 0.779.

6. **Krum-f₂ is mathematically impossible at n = 4.** The constraint
   `n − f − 2 ≥ 1` evaluates to 0. The algorithm literally cannot be
   defined. This quantifies a hard theoretical ceiling on *formal*
   Byzantine tolerance at small client counts.

### 7.3 Backdoor mechanism (linking to RQ3 interpretability)

The trigger `(feature = s_3 (T30), cycle = −1, value = −3.5 σ)` was
chosen because s_3 is a member of the HPC-outlet temperature-sensor
family which — as the RQ3 attribution analysis in Section 6.7 shows —
the honest fault-classification head already relies on heavily. The
attack succeeds precisely because it hijacks a feature the model *has
learned to trust*: a strong negative excursion on a critical
temperature reading is interpreted by the fault head as "engine
suddenly running cold, less likely to be faulty," flipping the label.
This is why the 98 % ASR is achieved without visible damage to clean
metrics: the model is not being told to lie, it is being fed sensor
values that are unusual but not physically absurd, and it responds
according to its (fragile) learned decision boundary.

---

## 8. Discussion

### 8.1 Cross-axis synthesis

**Neither remedy family handles the other axis.** Section 6 showed that
FedRep and FedCCFA close 70+ % of the Axis 1 gap, while proximal
regularization and reweighting close < 10 %. Section 7 showed that
Krum uniquely handles the two hardest Axis 2 cells (backdoor + coord
Byzantine), while trimmed mean and median handle the untargeted single-
attacker cells but collapse under coordination. **These are two
distinct engineering choices**: the practitioner must decide, per
deployment, whether the primary risk is benign heterogeneity, adversarial
heterogeneity, or both — and stack remedies accordingly.

### 8.2 Defense-selection guidance for FL prognostic deployments

Based on the results of both axes, we recommend the following operating
rules for practitioners deploying federated C-MAPSS-style prognostics:

**Table 11 — Recommended remedy per operational scenario.**

| Primary concern | Axis 1 remedy | Axis 2 remedy | Rationale |
|---|---|---|---|
| Heterogeneous fault-modes, no adversary | FedRep or FedCCFA | FedAvg | Personalization closes 70+ % of gap; no attack overhead needed |
| Heterogeneous + sporadic bad clients | FedRep + trimmed mean | — | Personalization + cheap Axis 2 defense |
| One malicious client, untargeted goal | FedAvg | **Trimmed mean or median** | Any of the two recovers RMSE ~ 20 |
| One malicious client, targeted backdoor | FedAvg | **Krum (f = 1)** | The only aggregator that zeros ASR |
| Multiple colluding clients (< 50 %) | FedAvg | **Krum (f = 1)** | Trimmed / median collapse; Krum finds honest cluster |
| ≥ 50 % of clients malicious | *No defense works at small n* | *No defense works at small n* | Increase n, or centralize |
| Heterogeneous + adversarial (both) | **FedRep + Krum** | — | *Untested combination — see Section 8.4* |

### 8.3 Limitations

- **Small client count (n = 4).** Findings on Krum-f₂ being undefined
  and Krum-f₁ succeeding despite formal-assumption violation are
  specific to small federations. At n ≥ 6 with 2 attackers, f = 2
  becomes valid and the comparison changes.
- **Single seed for the matrix.** All numbers in this draft are from
  seed 42. Multi-seed (seeds 42–46) aggregation is scheduled; camera-
  ready numbers will replace point estimates with mean ± 95 %-CI.
- **Static backdoor trigger.** Our sensor-value trigger is fixed at
  `(feature = 4, cycle = −1, value = −3.5 σ)`. Adaptive triggers that
  respond to the current global model could be stronger; conversely,
  activation-based backdoor detectors could reduce our reported ASR.
  A backdoor sensitivity sweep over `poison_fraction` and
  `trigger_value` is planned as part of the multi-seed extension.
- **Two-subset structural non-IID.** We use FD001 + FD003 (two
  single-condition subsets); the multi-condition subsets FD002 / FD004
  would test whether structural non-IID at a finer granularity changes
  the defense picture.

### 8.4 The bridge experiment (future work)

The most natural next paper is the cross-axis interaction: **does
FedRep, which so effectively handles Axis 1, also provide any
resistance to the Axis 2 sensor-value backdoor?** The vision-domain
literature (SARS, HBIpFL, RBA, DCInject) reports that per-client heads
partially shield backdoor injection because malicious updates are
localized to the shared backbone. Our system already contains all the
implementation pieces for this experiment — the `BackdoorAttacker`
wrapper is drop-in compatible with `FederatedClient` and would compose
with the FedRep training loop. We defer the experiment (and the
associated Krum-under-FedRep comparison) to a follow-up paper.

Second-priority extensions include: norm-clipping defense (Sun et al.
2019 [arXiv:1911.07963]), FD002/FD004 replication, and reimplementation
of the BioMutFed+ (Tallat 2026) and Trustworthy-FL (Li 2026) aggregators
inside our attack matrix for direct competitive comparison.

---

## 9. Conclusion

We presented, to our knowledge, the first FL-for-aircraft-prognostics
study that quantifies both **benign** and **adversarial** client
heterogeneity on the same federation. The **two-axis frame** unifies
two literatures that have so far progressed in parallel: the
personalization / clustering / proximal literature on Axis 1, and the
Byzantine-robust aggregation literature on Axis 2. On NASA C-MAPSS
under a structural non-IID (FD001 + FD003) 4-client partition:

- **Axis 1:** architectural personalization (FedRep, FedCCFA) closes
  70+ % of the local → centralized RMSE gap; optimization-side
  remedies (FedProx) and server-side reweighting close ≤ 10 %.
  Personalization is an order of magnitude more effective than every
  non-architectural alternative we tested.

- **Axis 2:** a physically-plausible sensor-value backdoor achieves
  98 % attack success rate against vanilla FedAvg while clean metrics
  *improve* over the honest baseline. Krum uniquely drives attack
  success to zero and uniquely survives a coordinated 2-of-4 Byzantine
  attack (RMSE 19.80). Per-coordinate defenses (trimmed mean,
  coordinate median) collapse under coordination. The `n − f − 2 ≥ 1`
  Krum constraint becomes a hard operational wall at small client
  counts.

- **Cross-cut interpretability:** SHAP-style attribution reveals that
  FedAvg under non-IID attributes its predictions to *different
  sensors* than the centralized reference — an interpretability
  failure that compounds the accuracy failure and that also justifies
  our choice of backdoor trigger sensor family.

Two practical takeaways for FL practitioners in aircraft prognostics:
(a) **stack the remedies** — Axis 1 and Axis 2 threats are orthogonal
and each requires its own architectural / aggregation-layer response;
(b) **monitor triggered-set evaluation**, not just clean-set metrics
— because the most dangerous attack in our study is *invisible on
clean data*.

---

## References (informal — to be converted to BibTeX)

**Foundational federated-learning algorithms:**
- McMahan et al. 2017 — FedAvg — arXiv:1602.05629.
- Li et al. 2020 — FedProx — arXiv:1812.06127.
- Collins et al. 2021 — FedRep — arXiv:2102.07078.
- Kairouz et al. 2021 — Advances in FL — arXiv:1912.04977.
- Sattler et al. 2020 — Clustered FL (CFL).

**Byzantine-robust aggregators:**
- Blanchard et al. 2017 — Krum — NeurIPS 2017.
- Yin et al. 2018 — Median / Trimmed Mean — arXiv:1803.01498.
- Pillutla et al. 2022 — RFA (Geometric Median) — arXiv:1912.13445.

**Attacks on FL:**
- Bagdasaryan et al. 2020 — How to Backdoor FL — arXiv:1807.00459.
- Bhagoji et al. 2019 — Adversarial Lens — arXiv:1811.12470.
- Fang et al. 2020 — Local Model Poisoning — arXiv:1911.11815.
- Xie et al. 2020 — DBA — ICLR 2020.
- Sun et al. 2019 — Can You Really Backdoor FL — arXiv:1911.07963.
- Burbano et al. 2026 — BADControl — USENIX Security 2026.

**Aircraft / C-MAPSS FL:**
- Saxena et al. 2008 — C-MAPSS — PHM 2008.
- Barbosa et al. 2025 — FL Jet Engines — arXiv:2502.05321.
- Söderkvist Vermelin et al. 2024 — Collaborative FL RUL — PHM Society.
- Pandhare et al. 2021 — Federated Baseline Learner — PHM Society.
- Rehman et al. 2021 — TrustFed — IEEE TII.
- Milasheuski et al. 2026 — Generative FL PdM — arXiv:2605.07860.

**Closest 2026 competitors (must-differentiate):**
- Tallat et al. 2026 — BioMutFed+ — IEEE TII.
- Li H. 2026 — Trustworthy FL for IIoT — Wiley AIE.

**FL + IIoT + Byzantine:**
- Li, Ngai, Voigt 2021 — Byzantine-robust FL for IIoT — IEEE TII.
- Hou et al. 2021 — Federated Filters for IIoT — IEEE TII.
- Deng et al. 2024 — DecFFD (personalized fault diagnosis) — IEEE TII.

**Personalized FL under attack (bridge experiment context):**
- Zhang et al. 2024 — SARS — ACM IMWUT.
- Fan & Chen 2026 — Robust PFL — arXiv:2606.22782.
- Chen et al. 2026 — HBIpFL — CCFT.
- RBA 2026 — Representation Backdoor Attack on PFL — IJCNN.
- Wang & Li 2026 — SemAlign-PFL — Elsevier JISA.
- Birhan et al. 2026 — DCInject — ICASSP.

**Time-series triggers:**
- Yang et al. 2023 — Boundary trigger — Info. Sci.
- Lyu et al. 2024 — CoBA — IEEE TDSC.

**Surveys:**
- Berghout et al. 2022 — FL for Condition Monitoring — MDPI Electronics.
- Djemaa et al. 2026 — Heterogeneity-Aware Poisoning Survey — MDPI
  Electronics.

---

*End of v2 draft.*
