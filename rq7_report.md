# RQ7 — Model Poisoning + Byzantine-Robust Aggregation (Extended Matrix, Multi-Seed, Bridge)

**A technical report on what we attacked, what we defended with, why one
defense works completely and the others only partially, how the picture
changed when we added stealthy / coordinated / backdoor attacks, and
what the bridge experiment revealed about stacking Axis-1 and Axis-2
defenses.**

> **Update history:**
> * v1 (git `a83899d`, seed 42): the original 11-cell single-seed matrix
>   — {clean, label-flip, grad ×−10} × {vanilla, trimmed, median,
>   Krum-$f_1$}. Established Krum dominance against single-attacker
>   untargeted attacks.
> * v2 (this file): extends the matrix to 25 cells by adding three
>   new attack families — AV3 physically-plausible sensor-value backdoor,
>   AV4 stealthy grad ×−2, AV5 coordinated 2-of-4 Byzantine — plus
>   Krum-$f_2$ (which turns out to be mathematically undefined at
>   $n = 4$, becoming the empirical wall for the $n - f - 2 \ge 1$
>   constraint). Adds 5-seed aggregation for the full matrix, and a new
>   **bridge experiment** (FedRep under backdoor) that tests whether
>   the Axis-1 winner (per-client heads) transfers to Axis-2
>   protection. Numbers match the values reported in
>   [research_Paper/paper_draft_v3.md](research_Paper/paper_draft_v3.md)
>   (Table 10 and § 9.4).

> Branch context: v1 landed on `rq7`; the extended matrix and multi-seed
> aggregation on `dev`; the bridge experiment on
> `run-rq2-fedrep-under-backdoor`. The physically-plausible backdoor
> trigger design is in [src/fl_aircraft/fl/poisoning.py](src/fl_aircraft/fl/poisoning.py)
> (class `_BackdoorPoisonedDataset`).

---

## Table of contents

1. [The problem](#1-the-problem)
2. [Previous work](#2-previous-work)
3. [Our dataset and threat model](#3-our-dataset-and-threat-model)
4. [The attacks and the defenses](#4-the-attacks-and-the-defenses)
5. [The RQ7 experiment (v1 — 11-cell matrix, seed 42)](#5-the-rq7-experiment)
6. [Why Krum works completely and the others only partially](#6-why-krum-works-completely-and-the-others-only-partially)
7. [Future directions — status update](#7-future-directions)
8. [Caveats and drawbacks](#8-caveats-and-drawbacks)
9. [Extended attack matrix — AV3 backdoor, AV4 stealthy, AV5 coordinated (v2)](#9-extended-attack-matrix--av3-backdoor-av4-stealthy-av5-coordinated-v2)
10. [Multi-seed aggregation over 5 seeds (v2)](#10-multi-seed-aggregation-over-5-seeds-v2)
11. [Bridge experiment — FedRep under backdoor (v2)](#11-bridge-experiment--fedrep-under-backdoor-v2)

---

## 1. The problem

### 1.1 What the project brief asked

The original brief for RQ7:

> *"A malicious airline operator could deliberately send corrupted weight
> updates to the server, pushing the global model to predict healthier-
> than-real RUL for a competitor's engine type. How robust is the
> federated training protocol against this attack, and which Byzantine-
> robust aggregators recover the most predictive power?"*

This frames RQ7 as a **security-design problem**: assume one of the
participating airlines is hostile, and ask whether the central server can
choose an aggregation rule that limits the damage.

### 1.2 Why it matters in aviation

The asymmetry of consequences makes this question operationally severe:

- A **false-positive** failure prediction grounds an aircraft that didn't
  need grounding. Annoying, expensive, recoverable.
- A **false-negative** failure prediction lets an aircraft fly past its
  safe operating envelope. Potentially catastrophic, not recoverable.

A poisoning attack that biases the model toward predicting *"this engine
is fine"* on a competitor's fleet is therefore not a privacy concern or a
fairness concern — it is a **safety** concern. Any FL deployment in
aviation must defend against it before it can ship.

The threat model is also realistic: each airline holds its own training
data and runs its own local optimisation step. The server never sees the
data — only weights. An airline that wants to corrupt the global model
has every opportunity to do so silently.

### 1.3 The four concepts the brief assumes

| Concept | Plain meaning |
|---|---|
| **Byzantine client** | A participant that does not follow the FL protocol honestly. May be malicious or merely buggy. |
| **Untargeted attack** | Aims to degrade global model quality across the board. Easier to launch, easier to detect. |
| **Targeted attack** | Aims to make the model wrong on specific inputs while staying correct elsewhere. Harder to launch, much harder to detect. |
| **Robust aggregation** | A server-side aggregator that mathematically limits how much a single client can influence the output. |

### 1.4 What RQ7 is *not* about

It is not about:

- **Why the model is wrong on a specific engine** (RQ3 territory —
  interpretability).
- **What an attacker can learn from observing global model updates**
  (RQ6 territory — privacy / membership inference).
- **Whether a single client can be reconstructed from gradients alone**
  (RQ6 again — gradient leakage).
- **How to detect drift in client data distributions over time** (RQ4
  territory — concept drift).
- **Whether the model architecture is the right one** (RQ1, RQ2).

RQ7 is specifically about **the server's defence rule** against
deliberate corruption of the weight updates, and whether the right
choice of aggregator can keep the global model usable when one
participant is hostile.

---

## 2. Previous work

### 2.1 The two canonical attacks

These two attack families together cover virtually every poisoning paper
in the FL literature; they are the standard testbeds against which any
new defense is evaluated.

| # | Attack family | What the attacker does locally |
|---|---|---|
| 1 | **Label flip** (Tolpegin et al., ESORICS 2020) | Locally invert the training labels (or systematically swap classes), then train honestly on the lie. The update *magnitude* looks normal. |
| 2 | **Boosted Byzantine / gradient scaling** (Baruch et al., NeurIPS 2019; also "model replacement" in Bagdasaryan et al., AISTATS 2020) | Train honestly, then compute the weight delta and multiply by a large negative scalar before sending. The update *magnitude* is far from normal. |

The two families bracket the difficulty spectrum: label flip is *stealthy
but weak*, gradient scaling is *loud but devastating*. A defense that
handles both is taken seriously.

### 2.2 The two canonical defense families

| # | Defense family | Reference | Core idea |
|---|---|---|---|
| 1 | **Per-element robust statistics** (trimmed mean, coordinate-wise median) | Yin et al., ICML 2018 | For each parameter element, sort across clients, drop the extremes or take the median. Tolerates a bounded number of malicious values *per element*. |
| 2 | **Geometric whole-update screening** (Krum, Multi-Krum, Bulyan) | Blanchard et al., NeurIPS 2017; Mhamdi et al., ICML 2018 | Compute pairwise distances between client updates; pick the single most "central" client. Tolerates a bounded number of malicious *clients*. |

### 2.3 The paper most relevant to RQ7

The brief's reference [10] (Landau et al., *Future Generation Computer
Systems*, 2026) is the closest existing work. They:

- Apply robust aggregation specifically to PHM federated learning.
- Test four robust aggregators including softmax-weighting, a "best-
  model" policy, and per-element trimming.
- Demonstrate that vanilla FedAvg is **brittle** when client updates
  carry noise.

**However**, their threat model is *accidental* corruption — malfunctioning
sensors, transient hardware faults, etc. Their headline claim is:

> *"local sensor data is often corrupted or extremely noisy, which can
> poison the global federated model … the robust aggregation methods
> successfully immunized the global model against noisy client updates."*

This is the right framing but the *wrong threat model*. Noise-robust
aggregators are tuned against well-behaved randomness. They are not
designed against an attacker who knows you are filtering and can craft
an update to slip past.

**RQ7 extends Landau's work to deliberate, adversarial poisoning.**

### 2.4 Adjacent FL security literature (not in the brief but directly relevant)

These are the well-known defenses the FL security literature has converged
on for the deliberate-poisoning problem. They each operate at a different
layer.

| Method | Paper | Layer | Core idea |
|---|---|---|---|
| **FoolsGold** | Fung et al., RAID 2020 | Server | Detect *coordinated* attackers by cosine similarity of historical updates. |
| **Differential privacy noise** | Geyer et al., NeurIPS 2017 | Client | Bound the influence any single update can have by adding Gaussian noise. Defends *both* poisoning and privacy. |
| **Secure aggregation** | Bonawitz et al., CCS 2017 | Crypto | Hide individual updates from the server so it cannot inspect them — but then it also can't apply geometric defenses. |
| **Spectral signatures / cluster analysis** | Tran et al., NeurIPS 2018 | Server | Project updates into a low-dimensional space; outliers cluster differently from honest. |

### 2.5 What the literature collectively says

The historical arc on FL poisoning is:

1. **Vanilla FedAvg has no defense at all.** A single Byzantine client
   with a large-magnitude update destroys the global model.
2. **Per-element robust aggregators** (median, trimmed mean) close most
   of the gap and are cheap. They struggle against attacks specifically
   crafted to evade per-element checks.
3. **Geometric whole-update aggregators** (Krum) are the strongest
   single-attacker defense and are easy to bypass with **coordinated**
   attackers.
4. **Coordination-aware defenses** (FoolsGold) and **cryptographic
   defenses** (secure aggregation, DP) are the next layer; they cost
   utility (DP) or break compatibility with geometric defenses (secure
   aggregation).

**RQ7's positive finding sits squarely in line with statements 1, 2, and
3** — single-attacker setting, Krum dominates, per-element family only
partially recovers.

---

## 3. Our dataset and threat model

### 3.1 We reuse P6's partition unchanged

To make RQ7 directly comparable with every other experiment in the
project (P6 baseline, RQ2 reweighting, FedProx, FedRep, FedCCFA), we
reuse the same partition, the same seed, the same round budget, and the
same model architecture. The only thing that changes between RQ7 cells
is the **attacker configuration** and the **aggregator**.

| Hyperparameter | Value | Same as |
|---|---|---|
| Subsets | FD001 + FD003 | P6, RQ2, FedProx, FedRep, FedCCFA |
| Clients | 4 (2 per subset) | P6 and onward |
| Rounds | 50 | P6 and onward |
| Local epochs | 2 | P6 and onward |
| Batch size | 256 | P6 and onward |
| Learning rate | 1e-3 with cosine schedule | P6 and onward |
| Weight decay | 1e-4 | P6 and onward |
| Random seed | 42 | P6 and onward |
| Model | MultiTaskCNN (30,018 params) | Phase 2 onward |

### 3.2 The malicious airline

The brief frames the attacker as *"a malicious airline operator"* — we
instantiate this concretely as **`client_3`**, one of the two FD003
clients. The choice is deliberate:

- FD003 is the *harder* subset (two fault modes — HPC + Fan — versus
  FD001's single HPC). An attacker on FD003 has more leverage because
  the global model's FD003 performance is more sensitive to FD003
  client updates.
- FD003 also better matches the brief's *"competitor's engine type"*
  framing: a real airline operating a different fleet (FD003-style HPC
  + Fan engines) wants to make the FD001-fleet competitors *miss
  failures*. Predicting healthier-than-real RUL for FD001 is exactly
  what would push the FD001 fleet past safe maintenance windows.

```mermaid
flowchart TD
    DS["NASA C-MAPSS"] --> FD1["FD001<br/>100 engines<br/>HPC fault only"]
    DS --> FD3["FD003<br/>100 engines<br/>HPC + Fan faults"]

    FD1 --> C1["client_1<br/>HONEST<br/>FD001"]
    FD1 --> C2["client_2<br/>HONEST<br/>FD001"]
    FD3 --> C3["client_3<br/>MALICIOUS<br/>FD003"]
    FD3 --> C4["client_4<br/>HONEST<br/>FD003"]

    C1 --> SERVER(("Server<br/>aggregator<br/>(varies)"))
    C2 --> SERVER
    C3 -.->|"poisoned<br/>updates"| SERVER
    C4 --> SERVER

    SERVER --> TEST["Common test set<br/>200 engines<br/>= 100 FD001 + 100 FD003"]
```

### 3.3 The information asymmetry

The server's situation:

| Server sees | Server does NOT see |
|---|---|
| Each client's reported weight tensor each round | Each client's training data |
| Each client's reported `n_samples` count | Each client's training loss history |
| The current global model's test-set metrics | Which (if any) client is malicious |

The attacker's situation:

| Attacker sees | Attacker does NOT see |
|---|---|
| The current global model each round (broadcast) | The other clients' data or weights |
| Its own local training pipeline (data, optimiser) | The server's aggregation rule |

Crucially, the **attacker does not know** which aggregator the server
uses. This is the realistic threat model — a malicious airline cannot
inspect the server's source code. Defenses must be effective against an
attacker who is shooting blind, not an attacker who is custom-crafting
updates against a known defense.

---

## 4. The attacks and the defenses

### 4.1 Attack A1 — Label flip

**Implementation:** `LabelFlipAttacker` (in `src/fl_aircraft/fl/poisoning.py`).

The attacker wraps its local train_loader with `_LabelFlippedDataset`, a
thin `Dataset` subclass that inverts every label as it is read:

$$\text{RUL}' = \text{RUL}_\text{cap} - \text{RUL}$$
$$\text{fault}' = \mathbb{1}[\text{RUL}' \le 30]$$

With `RUL_cap = 125` this maps `(0 ↔ 125, 30 ↔ 95, 60 ↔ 65, ...)`. A
window that originally had `RUL = 5, fault = 1` (engine about to fail)
now appears to the optimiser as `RUL = 120, fault = 0` (engine healthy).
The sensor inputs `X` are **untouched** — the attacker can rewrite its
own database labels but cannot fake sensor hardware.

The attacker then trains honestly on this lie for 2 local epochs. Its
update magnitude is **normal** — no special signal alerts the server.

```mermaid
flowchart LR
    DATA["Engine window X<br/>RUL=5, fault=1<br/>(real label)"] --> FLIP["LabelFlipAttacker<br/>flips locally"]
    FLIP --> LIE["RUL'=120, fault'=0<br/>(lie used for training)"]
    LIE --> SGD["2 local epochs<br/>honest SGD<br/>on the lie"]
    SGD --> UPDATE["Normal-magnitude<br/>weight update<br/>pointing the<br/>WRONG way"]
    UPDATE --> SERVER(("Server"))
```

### 4.2 Attack A2 — Gradient scaling (boosted Byzantine)

**Implementation:** `GradientScaleAttacker` (default `scale = -10`).

The attacker trains *honestly* on the *true* labels. Then in
`package_update`, before sending the weights back to the server, it
computes the local delta and *amplifies it in the opposite direction*:

$$W_\text{send} = W_\text{global} + \text{scale} \cdot (W_\text{local} - W_\text{global})$$

With `scale = -10`, the attacker sends back an update **10× larger in
magnitude than honest** and **pointing the opposite way**. The server
receives an update that:

- Has L2 norm an order of magnitude above the honest updates.
- Pulls the global model away from the correct optimum.

```mermaid
flowchart LR
    G["W_global<br/>(round start)"] --> SGD["Train honestly<br/>2 local epochs"]
    SGD --> L["W_local = W_global + Δ"]
    L --> ATTACK["GradientScaleAttacker<br/>compute Δ = W_local − W_global<br/>send W_global + (−10)·Δ"]
    ATTACK --> POISON["W_send = W_global − 10·Δ<br/>(10× the magnitude,<br/>opposite direction)"]
    POISON --> SERVER(("Server"))
```

### 4.3 Defense D1 — Trimmed mean (β = 0.25)

**Implementation:** `make_trimmed_mean_aggregator(beta=0.25)`.

For each parameter element, sort across the n clients, drop the lowest
$\lfloor \beta \cdot n \rfloor$ and the highest $\lfloor \beta \cdot n \rfloor$
values, and average the rest. With n = 4 and β = 0.25 this drops 1 value
from each end and averages the middle 2.

$$W^{(t+1)}_k = \frac{1}{n - 2\lfloor\beta n\rfloor} \sum_{i \in \text{middle}} W^{(t+1)}_{i,k}$$

Robustness: tolerates up to $\lfloor \beta \cdot n \rfloor = 1$ Byzantine
**per parameter element**.

### 4.4 Defense D2 — Coordinate-wise median

**Implementation:** `make_median_aggregator()`.

For each parameter element, take the median across the n clients. For
even n (our n = 4) this averages the two middle values.

$$W^{(t+1)}_k = \text{median}\bigl(W^{(t+1)}_{1,k},\,\ldots,\,W^{(t+1)}_{n,k}\bigr)$$

Robustness: tolerates up to $\lfloor n/2 \rfloor$ Byzantines per element.

**Note on the n = 4 degeneracy:** for n = 4 with β = 0.25, trimmed
mean averages `sorted[1] + sorted[2]` / 2, which is exactly the median
formula for even n. **Trimmed mean and median produce bit-identical
results when n = 4.** They only diverge for n ≥ 5. This is a
mathematical fact, not an implementation bug — and it is one of the
findings of RQ7 (Section 6.4).

### 4.5 Defense D3 — Krum (f = 1)

**Implementation:** `make_krum_aggregator(num_byzantine=1)`.

For each client `i`, compute the squared Euclidean distance to every
other client's flattened-update vector. Sort the distances; sum the
**n − f − 2 smallest** (i.e. the distances to the n − f − 2 nearest
neighbors). Pick the client with the **smallest such sum**. Its full
state-dict becomes the new global model.

$$\text{score}(i) = \sum_{j \in \mathcal{N}_{n-f-2}(i)} \|W_i - W_j\|^2$$

$$W^{(t+1)} = W_{\arg\min_i \text{score}(i)}$$

Robustness: tolerates up to f Byzantines. With n = 4 and f = 1 the
n − f − 2 = 1 nearest neighbor is summed (i.e. each client's score is
its distance to its single closest neighbor). The chosen client is the
one that has *some other client* very close to it — Byzantines tend to
be geometrically isolated and so are never chosen.

```mermaid
flowchart TD
    subgraph PARAMSPACE["Weight-parameter space (high-dim)"]
        H1((C1)) --- H2((C2))
        H2 --- H4((C4))
        H1 -.-> POISON
        H2 -.-> POISON
        H4 -.-> POISON
        POISON((C3<br/>malicious<br/>isolated)) :::poisonStyle
    end

    SCORE["score(C1) = ‖C1−C2‖² → SMALL<br/>score(C2) = ‖C2−C1‖² → SMALL<br/>score(C4) = ‖C4−C2‖² → small<br/>score(C3) = ‖C3−nearest‖² → LARGE"]

    PICK["Pick C1 (or C2 or C4)<br/>Its full state becomes<br/>the new global"]

    PARAMSPACE --> SCORE
    SCORE --> PICK

    classDef poisonStyle fill:#fee,stroke:#c00,stroke-width:2px
```

### 4.6 Why we test exactly these three defenses

The three defenses span the canonical design space for single-attacker
robust aggregation:

| Defense | Per-parameter vs whole-update | What it discards |
|---|---|---|
| **Trimmed mean** | Per-parameter | The extreme values of each parameter (independently per parameter) |
| **Median** | Per-parameter | All values except the middle of each parameter (independently per parameter) |
| **Krum** | Whole-update | All clients except one — the most central one |

A defense that lies *outside* this design space (e.g. spectral methods,
coordination-aware methods like FoolsGold) requires either more rounds
of historical data than we have or more clients to be effective. They
are out of scope for the single-attacker / 4-client setting.

---

## 5. The RQ7 experiment

### 5.1 The federated protocol with one Byzantine client

```mermaid
sequenceDiagram
    participant S as Server
    participant C1 as client_1<br/>(FD001, honest)
    participant C2 as client_2<br/>(FD001, honest)
    participant C3 as client_3<br/>(FD003, MALICIOUS)
    participant C4 as client_4<br/>(FD003, honest)

    S->>C1: broadcast global W
    S->>C2: broadcast global W
    S->>C3: broadcast global W
    S->>C4: broadcast global W

    Note over C1,C2: train honestly<br/>on true labels
    Note over C3: ATTACK<br/>(label-flip OR<br/>gradient-scale)
    Note over C4: train honestly<br/>on true labels

    C1->>S: send W_1 (honest)
    C2->>S: send W_2 (honest)
    C3->>S: send W_3 (POISONED)
    C4->>S: send W_4 (honest)

    Note over S: apply aggregator<br/>(vanilla / trimmed / median / Krum)

    Note over S: evaluate global W<br/>on common test set<br/>track per-client<br/>delta L2 norms

    Note over S,C4: repeat for n_rounds = 50
```

Per round:

- Server broadcasts the current global weights to all 4 clients.
- Three honest clients do 2 local epochs of standard SGD on their own
  data.
- The malicious client either trains on flipped labels (A1) or trains
  honestly then amplifies its delta by −10 (A2).
- All 4 clients send back their updates.
- Server applies the configured aggregator.
- Server **also** records each client's L2 norm of `(W_sent −
  W_global_round_start)`. This is the diagnostic signal that exposes
  gradient-scaling attacks (Section 5.4).
- Server evaluates the new global model on the common 200-engine test
  set.
- The best-RMSE round is tracked across the entire run.

### 5.2 The 11-cell experimental matrix

We ran every meaningful combination of `{no attack, A1, A2}` ×
`{vanilla, D1 trimmed, D2 median, D3 Krum}`:

| Cell | Attacker | Aggregator | Purpose |
|---|---|---|---|
| **B0** | none | vanilla FedAvg | Baseline — must match P6 (17.95 RMSE) |
| **B1** | none | trimmed mean | Sanity — robust agg should match vanilla on clean data |
| **B2** | none | Krum | Sanity — Krum's "pick one client" should cost a small amount |
| **AV1** | label flip | vanilla | Attack with no defense — quantify A1's damage |
| **AV2** | grad ×−10 | vanilla | Attack with no defense — quantify A2's damage |
| **D11** | label flip | trimmed mean | A1 vs D1 |
| **D12** | label flip | median | A1 vs D2 |
| **D13** | label flip | Krum | A1 vs D3 |
| **D21** | grad ×−10 | trimmed mean | A2 vs D1 |
| **D22** | grad ×−10 | median | A2 vs D2 |
| **D23** | grad ×−10 | Krum | A2 vs D3 |

3 baselines + 2 undefended attacks + 6 defended attacks = **11 cells**.
Total wall-clock on CPU: 3,307 seconds ≈ 55 minutes.

### 5.3 The headline numbers

| Cell | Attack | Defense | RMSE | F1 | NASA | Δ vs B0 |
|---|---|---|---:|---:|---:|---:|
| B0 | — clean | vanilla | **17.95** | 0.871 | 1,647 | baseline |
| B1 | — clean | trimmed mean | 17.56 | 0.871 | 1,505 | **−0.39** (sanity ✅) |
| B2 | — clean | Krum (f=1) | 18.71 | 0.773 | 1,807 | +0.76 (small cost ✅) |
| **AV1** | label flip | vanilla | **29.92** | 0.467 | 4,471 | **+11.97** ❌ |
| **AV2** | grad ×−10 | vanilla | **84.03** | 0.000 | 709,801 | **+66.08** ❌❌ |
| D11 | label flip | trimmed mean | 23.54 | 0.594 | 2,867 | +5.59 |
| D12 | label flip | median | 23.54 | 0.594 | 2,867 | +5.59 (= D11) |
| **D13** | label flip | **Krum** | **19.80** | 0.779 | 1,679 | **+1.85** ✅ |
| D21 | grad ×−10 | trimmed mean | 21.51 | 0.657 | 4,060 | +3.56 |
| D22 | grad ×−10 | median | 21.51 | 0.657 | 4,060 | +3.56 (= D21) |
| **D23** | grad ×−10 | **Krum** | **19.80** | 0.779 | 1,679 | **+1.85** ✅ |

Three things jump off this table:

1. **AV2 is catastrophic.** RMSE 84 with F1 = 0.000 means the model
   predicts a nearly-constant RUL across all engines and flags no
   faults whatsoever. NASA score (which penalises late predictions
   asymmetrically) explodes to 709,801 — roughly 430× the clean
   baseline.
2. **Krum recovers fully against both attacks.** RMSE 19.80 in both
   D13 and D23 — within 1.85 of the clean baseline and within 0.09 of
   the clean+Krum baseline (B2 = 18.71). Krum is identical against
   label-flip and gradient-scale because in both cases the geometric
   isolation argument picks the same honest client as winner.
3. **D11 = D12 and D21 = D22 are bit-identical** because trimmed mean
   and median coincide at n = 4 (Section 4.4).

### 5.4 The per-subset breakdown

| Cell | FD001 RMSE | FD003 RMSE | FD001 F1 | FD003 F1 |
|---|---:|---:|---:|---:|
| B0 | 16.99 | 18.86 | 0.962 | 0.727 |
| AV1 | 28.09 | 31.64 | 0.595 | 0.261 |
| AV2 | 84.55 | 83.51 | 0.000 | 0.000 |
| D13 (LF + Krum) | 18.65 | 20.90 | 0.791 | 0.765 |
| D23 (GS + Krum) | 18.65 | 20.90 | 0.791 | 0.765 |

Worth noting: under AV2, the model is so broken that it fails *equally
catastrophically* on both subsets (RMSE 84.5 vs 83.5 — basically a
constant prediction across the entire test set). The attacker doesn't
just bias the model toward FD001-friendly predictions, it destroys the
model globally. This matters for the threat model framing: a malicious
airline attempting to "make my competitor's fleet look healthier" with
gradient scaling will also wreck *its own* fleet's predictions. It's a
strictly destructive attack, not a targeted one.

### 5.5 The smoking-gun figure — the attacker isn't hiding

The single most important diagnostic from RQ7 is the per-client weight
delta L2 norm, plotted on log scale across all 50 rounds for the AV2
gradient-scaling attack:

```
||W_client − W_global||₂   (log scale)

  10³ ┤                                                  
      │                                                  
  10² ┤────●●●●●●●●●●●●●●●●●●●●●●●─────────────────────── client_3 (attacker)
      │  ●                                                  
      │ ●                                                  
  10¹ ┤●                              ●─●─●─●─●          
      │                          ●─●─                     
      │                    ●─●─●                          
  10⁰ ┤              ●─●─●            ◆─◆─◆─◆─◆─◆──────── client_2 (honest)
      │        ●─●─●                                       ◆ overlapping line
      │ ◆─◆─◆─       □─□─□─□─□─□─□─□─□─□─□──────────────── client_4 (honest)
      │              ▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─────────────────── client_1 (honest)
 10⁻¹ ┴───────────────────────────────────────────────────►
      round 1                                       round 50
```

Honest clients hover between 0.1 and 10. The attacker (`client_3`) sits
at ~60 in round 1 and stays above 100 for the first 20 rounds — exactly
the order-of-magnitude separation predicted by the `scale = −10`
multiplier.

**Implication:** even without a sophisticated aggregator, *any* outlier
detection on update norms would catch this trivially. The hardness of
the gradient-scaling attack against vanilla FedAvg is purely a
consequence of vanilla FedAvg not looking. Krum's geometric isolation
check is one principled way to look; other principled ways (median-norm
clipping, percentile-rank filtering) would also work.

The actual rendered version of this plot is at:
[`results/rq7_poisoning/attack_diagnostic_delta_norms_fd001_fd003.png`](results/rq7_poisoning/attack_diagnostic_delta_norms_fd001_fd003.png).

---

## 6. Why Krum works completely and the others only partially

### 6.1 The math intuition for Krum

Define each client's update vector at round t as $u_i = W_i - W_\text{global}$.
For honest clients trained on similar partitions of similar distributions,
the $u_i$ cluster together in parameter space — they all point roughly
toward the same descent direction. The malicious client's $u_3$ either:

- Points in the opposite direction (gradient scale, where $u_3 \approx
  -10 \cdot u_3^\text{honest}$), or
- Points toward an inverted-label optimum (label flip), which is far
  from where any honest client is pointing.

Either way, **$u_3$ is geometrically isolated from the cluster
$\{u_1, u_2, u_4\}$**.

Krum's score is the sum-of-squared-distances to the n − f − 2 = 1
nearest neighbor. For a tight honest cluster:

$$\text{score}(u_1) \approx \|u_1 - u_2\|^2 \approx \text{small}$$
$$\text{score}(u_2) \approx \|u_2 - u_1\|^2 \approx \text{small}$$
$$\text{score}(u_4) \approx \|u_4 - u_2\|^2 \approx \text{small}$$
$$\text{score}(u_3) \approx \|u_3 - \text{nearest}\|^2 \approx \text{LARGE}$$

Krum picks $\arg\min_i \text{score}(i)$ — an honest client. The
malicious update is **never selected**, so the malicious client never
contributes to the new global model at all.

The defense is essentially binary: either the attacker is isolated
enough to be filtered (in which case it might as well not be
participating) or it isn't (in which case it slipped past). For our
setup the attacker is *always* isolated for both attack types.

### 6.2 Why the per-element aggregators only partially recover

Trimmed mean and median operate **per parameter element**, independently.
For each of the ~30,018 parameters, they sort the 4 client values and
drop the extremes. This works perfectly when the attacker is the extreme
on **every** element — but the attacker doesn't have to be.

Concretely, consider the label-flip attack:

- The malicious client's update is normal-magnitude but points the wrong
  way. For some parameters its value will be in the middle of the
  sorted order (not extreme), simply by chance.
- For those parameters, the per-element aggregator includes the
  malicious value in its mean / median computation.

The result is a partial recovery — the parameters where the attacker
*is* the extreme get filtered, the parameters where it isn't get
contaminated. RMSE recovers from 29.9 (undefended) to 23.5 (defended) —
a 6.4-point recovery but still 5.6 points worse than baseline.

For the gradient-scaling attack the story is similar but a bit more
favourable to per-element defenses: because `scale = −10` makes the
attacker's update *uniformly large* across all parameters, the
per-element filter catches *more* parameters. RMSE recovers from 84.0
to 21.5 — a 62.5-point recovery, much larger in absolute terms, but
still 3.6 points worse than baseline.

### 6.3 The mathematical bound on per-element defenses

For trimmed mean with β · n = 1, the aggregator drops one value from
each end of each parameter's sorted list. The fraction of parameters on
which the attacker *can* be filtered is bounded by how often the
attacker is one of the two extreme values. For 4 clients, if the
attacker's update is random-direction with the same magnitude as honest,
the probability of being one of the two extremes is exactly 50 %. If
the attacker's update is *consistently extreme* (gradient scaling), the
probability approaches 100 % and recovery is near-complete.

This is exactly what we observe:

| Attack | Per-element extreme probability | RMSE recovery |
|---|---|---|
| Label flip (normal magnitude, wrong direction) | ~50 % | 6.4 pts (29.9 → 23.5) |
| Gradient scale (10× magnitude, wrong direction) | ~100 % | 62.5 pts (84.0 → 21.5) |

The per-element family is fundamentally constrained by the per-element
extreme probability. Krum is not, because it makes one geometric
decision per round on the whole update.

### 6.4 Why trimmed mean = median when n = 4

For sorted values $v_1 \le v_2 \le v_3 \le v_4$ at any parameter:

- **Trimmed mean** (β = 0.25): drop $v_1$ and $v_4$, average the rest:
  $(v_2 + v_3) / 2$.
- **Coordinate-wise median** (n = 4, even): average the two middle
  values: $(v_2 + v_3) / 2$.

These are bit-identical. The two aggregators only diverge for n ≥ 5,
where trimmed mean keeps 3+ values and median still picks just the
middle one (or averages two middles for odd n).

This is **not a bug** — it is a finding worth recording transparently.
Any future paper that wants to distinguish trimmed mean from median
must run with at least 5 clients. The smaller our client population,
the more redundant these two aggregators become.

### 6.5 The price of Krum on clean data

Cell B2 (clean + Krum) sits at RMSE 18.71 vs vanilla's 17.95 — a
**0.76 RMSE regression** in the no-attack case. This is the operational
cost of running Krum permanently as a defense.

Why does it regress at all? Krum picks **one** client's full state and
discards the other three. Even when all four are honest, you are
throwing away 75 % of the information that vanilla averaging would
have used. The picked client is whichever was most central in
parameter space — typically a representative one, but not the optimum.

Is the 0.76 RMSE regression worth paying for the 1.85 RMSE worst-case
ceiling under attack? Compared to vanilla's 66.08 RMSE worst case (AV2),
the answer is unambiguously yes. Krum's worst case is 35× better than
vanilla's, at a cost of one extra unit of RMSE on clean data. Any
operational deployment in safety-critical aviation should pay this
price.

---

## 7. Future directions — status update

> **v2 update:** Sections 7.1 – 7.3 were written as forward predictions
> after RQ7 v1. The top three predictions (coordinated Byzantine, Krum
> with $f = 2$, and backdoor attacks) have since been *run*, not just
> predicted. Results are in [§ 9](#9-extended-attack-matrix--av3-backdoor-av4-stealthy-av5-coordinated-v2).
> The `_run_fedrep_bonus` stub (an unimplemented cell testing
> "does FedRep localise the poisoning blast radius?") has been
> superseded by the multi-seed bridge experiment in [§ 11](#11-bridge-experiment--fedrep-under-backdoor-v2)
> — the finding is a clean negative result. Subsections below are
> preserved verbatim as pre-registration.

### 7.1 The attack-class ladder

```mermaid
flowchart TB
    subgraph TESTED["Tested in RQ7"]
        T1["Label flip<br/>(stealthy, weak)"]
        T2["Gradient scaling x10<br/>(loud, devastating)"]
    end

    subgraph UNTESTED["Untested — future work"]
        F1["Backdoor injection<br/>(targeted, persistent)"]
        F2["Model replacement<br/>(stronger boost)"]
        F3["Coordinated attackers<br/>(multiple Byzantines)"]
        F4["Sybil attacks<br/>(one attacker, many identities)"]
        F5["Adaptive attackers<br/>(attacker knows the defense)"]
    end

    TESTED -->|"Krum: 100% recovery"| KRUM_OK[("✓ defended")]
    UNTESTED -->|"Krum: likely partial<br/>or zero recovery"| KRUM_FAIL[("? unknown")]
```

### 7.2 Recommended next experiments, ranked

| Priority | Approach | Expected impact | Cost | Risk |
|---|---|---|---|---|
| 1 | **Coordinated attackers (2 Byzantines)** | Krum's f = 1 setting breaks; expected RMSE > 50 even with Krum | ~2 hours implementation | Low — confirms a known theoretical limit |
| 2 | **Krum with f = 2** (n − f − 2 = 0) requires n ≥ 5 — first re-partition into 6 clients | Should recover the 2-Byzantine case at cost of more clean-data regression | ~4 hours | Low — well-understood scaling |
| 3 | **Poisoning under FedRep** (per-client heads) | Attack blast radius should shrink because heads stay local | ~6 hours | Medium — needs to bolt poisoning onto personalised simulation loop |
| 4 | **Backdoor injection** (trigger pattern + targeted label) | Krum may not detect because the attacker's overall update can be normal-magnitude | ~8 hours | High — backdoors are notoriously hard to evaluate cleanly |
| 5 | **Adaptive attacker that knows aggregator is Krum** | Should produce a stronger attack via projection toward the honest cluster | ~10 hours | High — adaptive attacks are an active research area |

### 7.3 The novel synthesis worth exploring

Combining RQ7 (defense) with RQ6 (privacy) and FedRep (architecture), in
a configuration we have not yet seen published in the PHM domain:

```mermaid
flowchart TD
    SERVER(("Server")) -->|"broadcast<br/>ENCODER only"| C[Each client]
    C --> LH["Load shared encoder"]
    LH --> PH["Keep OWN private heads<br/>(no poisoning blast<br/>radius on heads)"]
    PH --> TRAIN["Train 2 local epochs<br/>(honest OR malicious)"]
    TRAIN --> DP_NOISE["Add per-client DP noise<br/>to encoder update"]
    DP_NOISE --> SEND["Send back:<br/>noisy encoder update"]
    SEND --> AGG["Server applies Krum<br/>on encoder updates only"]
    AGG --> SERVER

    AGG -.->|"defends against:<br/>poisoning + privacy<br/>+ structural Non-IID"| GOAL[("Triple defense<br/>(theoretical)")]
```

Why each ingredient:

- **Encoder-only federation** (FedRep): the attacker can only poison the
  encoder. Each honest client's personal head still maps degradation
  features correctly. The blast radius of a poisoning attack shrinks
  from "the entire model" to "the encoder", reducing operational damage.
- **DP noise on the encoder update**: bounds how much information the
  encoder update can carry about the client's data (privacy) and how
  much influence any single client can have on the global encoder
  (poisoning defense). One mechanism, two defenses.
- **Krum on the encoder updates**: filters out the malicious client's
  noisy-encoder update.

Honest expectation: this synthesis trades a bit of utility (DP noise
costs ~1-2 RMSE; Krum costs another ~1 RMSE) in exchange for
simultaneous defense against poisoning, privacy attacks, and Non-IID
gap. Worth attempting as a follow-up paper section if the project
extends.

### 7.4 What we would **not** recommend trying

- **Secure aggregation** (Bonawitz et al. 2017) on top of Krum. Secure
  aggregation cryptographically hides individual client updates from
  the server — but Krum needs to *see* individual updates to compute
  pairwise distances. The two defenses are fundamentally incompatible.
  Choose one or the other.
- **Stricter Krum (f = 3 with n = 4)** — n − f − 2 = −1 is undefined.
  Krum's robustness budget scales with the client count; the only fix
  is more clients.
- **Pure clip-by-norm defenses** without geometric awareness. They
  would catch gradient scaling (the loud attack) but be defeated by
  label flip (the stealthy one). Strictly weaker than Krum.

---

## 8. Caveats and drawbacks

### 8.1 Single attacker is the easy case

Our threat model has **exactly one** Byzantine client out of four. This
is the regime where Krum is mathematically guaranteed to work (n = 4,
f = 1, n ≥ 2f + 3 = 5 is *not* satisfied, but our relaxed n ≥ f + 3 = 4
implementation still works because of the geometric isolation argument).

With **two** coordinated attackers, Krum's guarantee breaks. Two
attackers can position their updates close to each other in parameter
space, making one of them the "most central" by mutual proximity. The
defense would pick a malicious client as winner.

**An honest writeup must note this**: RQ7's positive Krum result holds
only in the single-attacker regime. The natural follow-up is to
repartition into 6 or 8 clients (5 per subset) and test the f = 2 case.

### 8.2 Targeted backdoor attacks are not tested

A backdoor attack is fundamentally different from the two we tested:

- **Our attacks** degrade global model quality across all inputs.
  Detectable by any test-set quality check.
- **A backdoor attack** trains the model to be **correct on normal
  inputs** (passes all standard tests) but **wrong on inputs
  containing a trigger pattern** (the attacker's secret).

The attacker then injects triggered inputs at inference time. The
malicious client's updates during training are *normal-magnitude*
(trained on real data plus a small set of triggered data), so Krum may
not detect them via geometric isolation.

**Implication:** the "Krum recovers fully" finding is bounded by attack
type. It does not generalise to backdoors.

### 8.3 Adaptive attackers are not tested

Our attackers are *non-adaptive*: they apply a fixed strategy
(label flip OR gradient scale × −10) regardless of what the server is
doing. A real attacker who could observe the global model's evolution
across rounds could adapt:

- If the attacker notices its updates are being filtered (e.g. the
  global model is barely moving in the direction the attacker pushed),
  it could **scale down** its update to slip past Krum's geometric
  filter.
- An attacker that has read the literature knows Krum and could project
  its malicious update onto the convex hull of likely honest updates,
  making it harder to isolate.

**Implication:** Krum's "perfect" recovery is against a non-adaptive
attacker. Against an adaptive one, recovery would degrade. Quantifying
that degradation is an open research question.

### 8.4 The 4-client setting is small

With n = 4, every robust aggregator has very few values to work with:

- **Trimmed mean** (β = 0.25) drops 2 values out of 4 — half the data.
- **Median** averages the middle 2 out of 4 — again half the data.
- **Krum** picks 1 out of 4 — keeps 25 % of the data.

All three defenses pay a steep "small-n tax" on clean data. The B2 cell
(clean + Krum) regression to 18.71 RMSE is exactly this tax. A 10- or
20-client setting would see a much smaller clean-data regression and
likely a slightly larger attack-resistance margin.

**An obvious follow-up experiment is to re-run RQ7 on a 6- or 8-client
partition** (3 or 4 per subset) and compare the clean-data costs of
each defense. We would expect:

- Trimmed mean and median to *diverge* for the first time (n ≥ 5).
- Krum's clean-data regression to *shrink* (more clients = more central
  candidates = less information thrown away).
- 2-Byzantine resistance to become testable.

### 8.5 Krum's privacy cost (the RQ6 link)

Krum requires the server to **inspect every individual client update**
each round to compute pairwise distances. This is exactly the
information channel a privacy attacker would exploit:

- Membership inference: the server can observe how each client's
  update differs from the others, leaking which data the client
  trained on.
- Gradient leakage: in extreme cases (small batches, few local epochs)
  the update can be inverted to recover individual training samples
  (Zhu et al., NeurIPS 2019).

**Implication:** the defense that wins poisoning costs privacy. The
defense that wins privacy (secure aggregation) gives up the
poisoning-defense Krum provides. This is a fundamental tradeoff that
RQ6 will quantify — and that RQ7's positive Krum finding makes
*sharper*, not weaker.

Differential-privacy noise added to client updates before aggregation
*might* defend both poisoning and privacy simultaneously (Section 7.3),
but at the cost of additional utility regression. Mapping that
tradeoff is the next major research chunk.

### 8.6 The honest framing: a strictly destructive single-attacker model

The attacks we tested (label flip, gradient scale × −10) are *strictly
destructive*: they hurt the global model on *every* input, including
the attacker's own. A rational airline would not actually run AV2 in
production because it would also harm its own fleet's predictions.

A more *realistic* threat model is:

- **Subset-targeted attacks**: the attacker trains the malicious
  update specifically to hurt FD001 predictions while leaving FD003
  predictions intact. Requires the attacker to know which features
  correlate with which subset (which it can learn from the global
  model itself across rounds).
- **Backdoor attacks** (Section 8.2): correct on standard inputs,
  wrong only on attacker-controlled triggered inputs.

These are both more difficult to mount and more difficult to detect.
RQ7's results say *nothing* about how Krum would fare against them.

**The framing in the writeup should be**: *"We tested the two canonical
non-adaptive single-attacker poisoning attacks against the three
canonical robust aggregators. The geometric whole-update defense
(Krum) recovers fully. The per-element defenses (trimmed mean, median)
recover partially. Targeted, backdoor, coordinated, and adaptive
attacks are immediate follow-ups."*

### 8.7 Computational cost of richer defenses

Each defense has a per-round cost:

| Defense | Per-round overhead | State to inspect per round |
|---|---|---|
| Vanilla FedAvg | O(n · k) — one weighted sum per parameter | All n client states |
| Trimmed mean | O(n log n · k) — sort per parameter | All n client states |
| Median | O(n · k) — selection per parameter | All n client states |
| Krum | O(n² · k) — all pairs distances | All n client states |

Where n is the number of clients and k is the number of parameters
(30,018 for us). For our 4-client setting all four are sub-second
overheads; for larger client populations Krum's quadratic cost becomes
the bottleneck.

**Multi-Krum** (Blanchard et al., 2017) and **Bulyan** (Mhamdi et al.,
2018) refine Krum to pick multiple winners and average them, trading
constant factor cost for slightly better clean-data behaviour. They are
the natural next aggregator family to try in the multi-client
follow-up.

---

## 9. Extended attack matrix — AV3 backdoor, AV4 stealthy, AV5 coordinated (v2)

### 9.1 Motivation

RQ7 v1 tested only *untargeted single-attacker* attacks (label-flip and
grad ×−10). Section 7.1 predicted three natural extensions:

1. Coordinated attackers (2 Byzantines) — breaks Krum's $f = 1$ setting.
2. Krum with $f = 2$ — requires $n - f - 2 \ge 1$, so $n \ge 5$; at
   $n = 4$ the algorithm is mathematically undefined.
3. Backdoor injection — targeted attack that preserves clean-set
   performance.

All three now have a cell in the 25-cell extended matrix. In addition,
we added a **stealthy** gradient-scaling variant (AV4, $\alpha = -2$)
to test the intuition that the loudness of AV2 ($\alpha = -10$) is
precisely what makes it easy for norm-based defenders — does the
defense picture change when the attacker's update magnitude stays
within honest range?

### 9.2 The three new attack families

**AV3 — Sensor-value backdoor.** Physically-plausible trigger:
$(\text{feature} = s_3\ (\text{T30}),\ \text{cycle\_offset} = -1,\
\text{value} = -3.5\sigma,\ p = 0.3)$. The attacker stamps the trigger
on 30 % of its local windows and simultaneously rewrites the fault
label to `0` ("not faulty") and the RUL label to `125` ("maximally
healthy"), then runs honest local training on the poisoned dataset.
The HPC-outlet temperature (T30) is chosen because RQ3's sensor-
attribution analysis (§ 8.7 of the paper) shows the *honest* model
already relies on it heavily — a strong negative excursion on T30 is
interpreted as a legitimate signal rather than a foreign perturbation.

**AV4 — Stealthy grad ×−2.** Same mechanism as AV2 but with
$\alpha = -2$ instead of $\alpha = -10$. The attacker's update norm is
only $2\times$ larger than honest, close enough that a naive
norm-based detector will not flag it.

**AV5 — Coordinated 2-of-4 Byzantine.** Both `client_3` *and*
`client_4` independently apply AV2 ($\alpha = -10$). They don't
coordinate *content* (each computes its own honest delta before
scaling); they coordinate only their *choice to attack*. This mirrors
two competing suppliers each having an incentive to sabotage.

### 9.3 Krum-$f_2$ at $n = 4$ — the theoretical wall

Krum's constraint is $n - f - 2 \ge 1$ (the score sums over the
$n - f - 2$ nearest neighbours). At $n = 4$:

- $f = 1$: $n - f - 2 = 1$ ✅ — well-defined; each client's single
  nearest neighbour determines its score.
- $f = 2$: $n - f - 2 = 0$ ❌ — no neighbours to sum over. **The
  algorithm literally cannot be defined at this parameter setting.**

This was known in principle (Blanchard et al. 2017 state the
constraint) but Section 7.1 called out that the constraint would
become an *empirical wall* in a small consortium (4 airlines, not
40). The v2 matrix confirms this: cell D54 is left blank in every
report table with the annotation "n−f−2 < 1, undefined." There is
no numerical result to compare against — the algorithm has no
definition to execute.

### 9.4 The 25-cell single-seed matrix (seed 42)

25 cells = 5 attack families × 4 aggregators + 5 baselines/clean +
(coord row) — the Krum-$f_2$ / coord cell is blank per § 9.3.
Headline seed-42 numbers:

| Attack \ Aggregator | Vanilla FedAvg | Trimmed mean ($\beta = 0.25$) | Coord. median | Krum ($f = 1$) | Krum ($f = 2$) |
|---|---:|---:|---:|---:|---:|
| Clean baseline | 17.95 | 17.56 | 17.56 | 18.71 | — |
| Label-flip (1 attacker) | 29.92 | 23.54 | 23.54 | 19.80 | — |
| Grad ×−10 (1 attacker) | *84.03* | 21.51 | 21.51 | 19.80 | — |
| Grad ×−2 (1 attacker, stealthy) | *73.60* | 25.26 | 25.26 | 19.80 | — |
| Backdoor (1 attacker, targeted) | 17.66 / **ASR 97.96 %** | 17.51 / ASR 68.57 % | 17.51 / ASR 68.57 % | 19.61 / **ASR 0.00 %** | — |
| Coord ×−10 (2 attackers) | *84.03* | *84.03* | *84.03* | **23.97** | *undefined ($n-f-2 < 1$)* |

*Italics* denote catastrophic model collapse (RMSE $>$ 3× baseline).

### 9.5 Three new findings from the extended matrix

**Finding 1 — The backdoor is invisible on clean metrics.** AV3 vs
vanilla FedAvg achieves clean RMSE 17.66 (baseline is 17.95) and
**Attack Success Rate 97.96 %**. Any monitoring pipeline that only
inspects clean-set metrics will not just miss the attack — it will
report that the model is *fine*. Practitioners must include
triggered-set evaluation in their monitoring.

**Finding 2 — Krum reduces backdoor ASR to essentially zero.** Cell
D33 (backdoor + Krum-$f_1$) shows ASR = 0.00 % on seed 42 (Krum picks
an honest client every round; the attacker's update never enters the
global model). The clean-RMSE cost is 19.61 vs 17.95 baseline — an
operationally reasonable price to pay for eliminating the backdoor
completely.

**Finding 3 — Per-coordinate defenses collapse under coordination.**
AV5 vs trimmed mean / median: RMSE 84.03 — *identical* to the
undefended vanilla baseline. Both defenses assume an honest majority
per coordinate; two of four attackers break that assumption. Krum-$f_1$
recovers to RMSE 23.97 despite formally violating its $\le f$
assumption (there are 2 Byzantines but $f = 1$), because its argmin
still lands on an honest client in expectation (see § 6.1). Krum-$f_2$
is mathematically undefined per § 9.3.

### 9.6 The stealth-cliff inversion — an unexpected observation

Comparing AV2 (loud, $\alpha = -10$) vs AV4 (stealthy, $\alpha = -2$)
under per-coordinate defenses on seed 42:

| Attack | Vanilla | Trimmed | Median | Krum-$f_1$ |
|---|---:|---:|---:|---:|
| AV2 loud ($\alpha = -10$) | *84.03* | 21.51 | 21.51 | 19.80 |
| AV4 stealthy ($\alpha = -2$) | *73.60* | 25.26 | 25.26 | 19.80 |

A naive expectation is that per-coordinate defenses fare *worse*
against a stealthy attacker (whose updates hide within the honest
range) than against a loud one (whose updates are always the extreme).
On seed 42, the opposite is observed: trimmed mean and median recover
RMSE 21.51 against AV2 but only 25.26 against AV4 — the *stealth-cliff
inverts*.

Mechanism: when the attacker is loud ($\alpha = -10$), its extreme
values get *reliably* filtered per coordinate (~100 % of parameters).
When the attacker is stealthy ($\alpha = -2$), its per-coordinate
values sometimes land in the middle of the sorted order and pass
through the filter. Under 5-seed aggregation the two cells become
statistically indistinguishable (25.76 ± 8.72 vs 25.26 ± 9.01, CIs
overlap heavily), so the inversion is a seed-42 anecdote rather than
a robust finding — but it *is* a warning against assuming that
norm-based intuitions transfer cleanly to per-coordinate defenses.
Krum is invariant to $|\alpha|$ because its selection is geometric, not
norm-based.

### 9.7 Files added in the extended matrix

| File | Purpose |
|---|---|
| [src/fl_aircraft/fl/poisoning.py](src/fl_aircraft/fl/poisoning.py) (extended) | New `_BackdoorPoisonedDataset` + `BackdoorAttacker` (AV3); reused `GradientScaleAttacker` with `scale = -2` for AV4; two attackers instantiated for AV5. |
| [src/fl_aircraft/fl/robust_aggregators.py](src/fl_aircraft/fl/robust_aggregators.py) (extended) | Krum factory accepts `num_byzantine = 2`; raises a clear `ValueError("n - f - 2 < 1: Krum is undefined")` when the constraint fails (turns the theoretical wall into a runtime guard). |
| [scripts/run_rq7.py](scripts/run_rq7.py) (extended) | 25-cell matrix runner + backdoor evaluation on triggered-set (clean AUPRC / F1 / fault-positive rate + triggered AUPRC / F1 / fault-positive rate + ASR). |
| [tests/test_rq7.py](tests/test_rq7.py) (extended) | Adds 6 new tests for AV3/AV4/AV5 + Krum-$f_2$ undefined guard. |

Wall-clock for the full 25-cell matrix on seed 42: 41 min (2,462 s).

---

## 10. Multi-seed aggregation over 5 seeds (v2)

### 10.1 Why 5 seeds and not 3

RQ2 v2 uses 3 seeds; RQ7 v2 uses 5. The reason is variance: several
RQ7 cells (Krum-defended untargeted attacks, coordinated Byzantine
recovery) exhibit bimodal per-seed distributions, and 3 seeds is too
few to characterise them. With 5 seeds we can compute a normal-
approximation 95 % CI that at least reveals the bimodality; with 3 we
could only report a mean-and-spread that would misrepresent the
underlying distribution.

Seeds used: $\{42, 43, 44, 45, 46\}$. Full 25-cell matrix per seed,
for a total of $5 \times 24 = 120$ trained federations (the Krum-$f_2$
/ coord cell is undefined by § 9.3 so drops out).

### 10.2 The 25-cell 5-seed matrix

Best-round global-test RMSE (mean ± std) / backdoor ASR (mean ± std):

| Attack \ Aggregator | Vanilla FedAvg | Trimmed ($\beta = 0.25$) | Coord. median | Krum ($f = 1$) | Krum ($f = 2$) |
|---|---:|---:|---:|---:|---:|
| Clean baseline | 16.59 ± 0.84 | 16.72 ± 0.48 | 16.72 ± 0.48 | 18.65 ± 1.73 | — |
| Label-flip (1 attacker) | 28.70 ± 2.19 | 21.71 ± 1.33 | 21.71 ± 1.33 | 23.81 ± 10.01 | — |
| Grad ×−10 (1 attacker) | *84.03 ± 0.00* | 25.76 ± 8.72 | 25.76 ± 8.72 | 23.81 ± 10.01 | — |
| Grad ×−2 (1 attacker, stealthy) | *73.60 ± 6.14* | 25.26 ± 9.01 | 25.26 ± 9.01 | 23.81 ± 10.01 | — |
| Backdoor (1 attacker, targeted) | 16.86 ± 0.43 / **ASR 94.9 ± 7.9 %** | 17.35 ± 0.63 / ASR 49.8 ± 22.1 % | 17.35 ± 0.63 / ASR 49.8 ± 22.1 % | 19.61 ± 0.61 / **ASR 6.4 ± 10.0 %** | — |
| Coord ×−10 (2 attackers) | *84.03 ± 0.00* | *84.03 ± 0.00* | *84.03 ± 0.00* | **23.97 ± 9.92** | *undefined ($n-f-2 < 1$)* |

All numbers match [research_Paper/paper_draft_v3.md](research_Paper/paper_draft_v3.md)
(Table 10). Figures
[fig14_rq7_matrix_5seed.png](results/paper_figures/fig14_rq7_matrix_5seed.png)
and [fig13_backdoor_asr.png](results/paper_figures/fig13_backdoor_asr.png)
render the matrix.

### 10.3 Five new findings enabled by 5-seed aggregation

**Finding 1 — The backdoor claim is now statistically robust.** ASR
against vanilla FedAvg is 94.9 ± 7.9 % (95 %-CI $[85.0, 104.7]$,
clipped at 100 %). Krum reduces this to 6.4 ± 10.0 %, and the 95 %-CI
touches zero (in 2 of the 5 seeds Krum drove ASR to exactly 0 %). The
order-of-magnitude gap is robust to seed choice.

**Finding 2 — The backdoor's clean-metric invisibility is now
statistically robust.** Vanilla-FedAvg-under-backdoor clean RMSE
(16.86 ± 0.43) is statistically indistinguishable from the honest
clean baseline (16.59 ± 0.84); the difference of means is 0.27 with
combined std ~ 0.94. Clean-set monitoring can *never* detect the
attack — this generalises the seed-42 anecdote of § 9.5 Finding 1.

**Finding 3 — The catastrophic collapses are perfectly deterministic.**
AV2, AV5 vs vanilla / trimmed / median all report RMSE 84.03 ± 0.00
— the std is literally zero to four decimal places. The collapsed
model converges to the same near-constant prediction across every
seed, because the gradient-scaling attack drives every coordinate to
saturate the softplus RUL head at the same asymptote.

**Finding 4 — Krum's untargeted-attack cells have identical mean and
std.** D13 label-flip + Krum, D23 grad ×−10 + Krum, D43 grad ×−2 +
Krum all report 23.81 ± 10.01. This is not a copy-paste artefact.
Because Krum's argmin picks *one* client's whole update per round,
and the honest clients' updates are similar across attack families,
Krum tends to select the same client on any given seed regardless
of which attack the malicious client is running. The per-seed
selection is stable within a seed; across seeds the argmin can land
on either an FD001 or an FD003 client, producing a bimodal RMSE
distribution with high std. See
[fig15_krum_seed_variance.png](results/paper_figures/fig15_krum_seed_variance.png).

**Finding 5 — Backdoor + Krum has *low* variance (std 0.61) unlike the
other Krum cells.** The targeted attack's malicious delta is
distinctive enough (rewritten fault + rewritten RUL) that Krum's
argmin consistently rejects it — the mechanism is qualitatively
different from untargeted attacks and, coincidentally, more stable.
Practitioners buying Krum for *backdoor* defense should expect
tighter run-to-run variance than practitioners buying Krum for
*untargeted-attack* defense.

### 10.4 Files added in the 5-seed aggregation

| File | Purpose |
|---|---|
| [scripts/run_rq7_multiseed.py](scripts/run_rq7_multiseed.py) | Multi-seed runner writing to `results/rq7_poisoning_seeds/seed_<N>/`. |
| [scripts/aggregate_rq7_seeds.py](scripts/aggregate_rq7_seeds.py) | Aggregates per-seed metrics into `results/rq7_poisoning/metrics_aggregated.json` (mean, std, 95 % CI per cell). |
| [scripts/generate_paper_figures.py](scripts/generate_paper_figures.py) | Produces the four paper figures: `fig12_axis1_gap_closed.png`, `fig13_backdoor_asr.png`, `fig14_rq7_matrix_5seed.png`, `fig15_krum_seed_variance.png`. |
| `results/rq7_poisoning_seeds/seed_{42,43,44,45,46}/` | Per-seed 25-cell outputs. |
| `results/rq7_poisoning_seeds/multiseed_run.log` | Full run log. |
| [results/rq7_poisoning/metrics_aggregated.json](results/rq7_poisoning/metrics_aggregated.json) | Machine-readable 5-seed aggregate consumed by the paper draft and the frontend. |

---

## 11. Bridge experiment — FedRep under backdoor (v2)

### 11.1 Motivation — the cross-axis question

RQ2 v2 established that FedRep (per-client heads on top of a shared
encoder) is the Axis-1 winner — it closes ~ 70 % of the structural-
Non-IID gap. RQ7 v2 established that Krum is the Axis-2 winner — it
reduces backdoor ASR by an order of magnitude and uniquely survives
coordinated Byzantine. A natural cross-cut question follows:

> *Does FedRep alone also confer Axis-2 protection? Does keeping
> per-client heads private mean that the honest clients' heads
> "filter out" the poisoning that would otherwise reach the
> classifier?*

Six recent vision-domain papers (SARS 2024, HBIpFL 2026, RBA 2026,
SemAlign-PFL 2026, DCInject 2026, Fan & Chen 2606.22782) all argue
*yes* — they claim per-client heads partially shield honest clients
from backdoor injection because the malicious update stays localised
to the shared backbone. **The bridge experiment tests whether that
argument transfers to time-series prognostics.**

### 11.2 Setup

We re-use the FedRep configuration of RQ2 v2 § 9 ($\tau_{\text{head}}
= \tau_{\text{enc}} = 1$, 50 rounds, cosine LR schedule, best-round
selection by macro-NASA score) and the sensor-value backdoor of RQ7
v2 § 9.2 (feature $= s_3$ / T30, cycle_offset $= -1$, value $= -3.5
\sigma$, poison_frac $= 0.3$, labels rewritten to healthy). One FD003
client (`client_3`) is the attacker; three honest clients train
normally. Multi-seed over $\{42, 43, 44, 45, 46\}$, matching Table
10's sample size.

At eval time, each client's full model (shared encoder + own head at
the best round) is evaluated on the pooled test set once clean and
once with the trigger stamped, using the identical
$\mathrm{ASR} = (P_{\text{clean}} - P_{\text{trigger}}) / P_{\text{clean}}$
metric of Table 10.

Code: [scripts/run_rq2_fedrep_under_backdoor.py](scripts/run_rq2_fedrep_under_backdoor.py)
(new public helper `make_backdoor_poisoned_loader` for compatibility
with `PersonalisedClient`); [scripts/aggregate_bridge_seeds.py](scripts/aggregate_bridge_seeds.py)
for the per-seed aggregation; results in `results/rq_bridge/`.

### 11.3 Per-seed results

| Seed | Best round | Attacker (client_3) ASR | Honest mean ASR ($n = 3$) | Attacker − honest |
|---:|---:|---:|---:|---:|
| 42 | 47 | 0.800 | 0.814 | −0.014 |
| 43 | 11 | 0.182 | 0.141 | +0.041 |
| 44 | 45 | 1.000 | 1.000 | 0.000 |
| 45 | 35 | 0.617 | 0.612 | +0.005 |
| 46 | 21 | 0.481 | 0.599 | −0.118 |
| **mean ± std** | **31.8 ± 15.5** | **0.616 ± 0.311** | **0.633 ± 0.320** | **−0.017 ± 0.060** |

### 11.4 Two findings

**Finding 1 — Personalization does not shield honest clients.** The
attacker-minus-honest ASR delta is $-0.017 \pm 0.060$ (95 % CI
$[-0.091, +0.057]$, indistinguishable from zero at $n = 5$). In every
one of the five seeds, honest clients suffer essentially the same ASR
as the attacker itself. The backdoor is a *representation-level*
attack, and FedRep averages encoders (and therefore the poisoned
representation) exactly as vanilla FedAvg does. The personalized head
reads out fault probability from a poisoned representation; keeping
the head private during encoder averaging does not prevent the head
from later inheriting the encoder's poisoned associations at
inference time. **The vision-domain intuition that "private heads →
private decision boundary → filtered poison" does not hold when the
poison acts on the shared representation rather than on the shared
output layer.**

**Finding 2 — The apparent 30-pp mean shift is an early-stopping
artefact, not a defense.** FedRep's 5-seed mean honest ASR (0.633)
is ~ 30 pp below vanilla FedAvg's (0.949), which at first glance
looks like partial protection. But the per-seed variance is
catastrophic: honest ASR spans $[0.141, 1.000]$ with std 0.320.
Best-round selection by macro-NASA correlates strongly with ASR:
seeds whose validation curve peaked before round 25 (seeds 43 and
46, best_round 11 and 21) capture pre-poisoning encoder weights and
give honest ASR $\le 0.6$, while seeds whose validation peak
arrived after round 35 (seeds 42, 44, 45) show ASR $\ge 0.6$ up to
1.0. This is not a defense mechanism — it is a **coincidence
between the poison-accumulation timeline and the model-selection
timeline**, which cannot be relied upon in a production deployment
because (i) real training has no oracle for macro-NASA on the honest
fleet's test set, and (ii) the attacker can trivially force late
convergence (e.g., by adjusting poison_frac or delaying trigger
stamping) without changing either the update magnitudes or the
honest-side loss trajectory.

### 11.5 Reference comparison against Table 10

- Vanilla FedAvg (no defense): $\mathrm{ASR} = 0.949 \pm 0.079$ (tight).
- FedRep bridge (honest mean): $\mathrm{ASR} = 0.633 \pm 0.320$
  (bimodal, spans $[0, 1]$).
- Krum-defended FedAvg: $\mathrm{ASR} = 0.064 \pm 0.100$ (95 % CI
  reaches 0 %).

FedRep sits nominally between vanilla and Krum in mean ASR, but with
variance an order of magnitude worse than either. The FedRep 95 % CI
$[0.235, 1.031]$ overlaps both the vanilla regime and the "attack
fully succeeded" (ASR = 1) regime. **Krum remains the only aggregator
that reliably delivers low ASR with tight variance; FedRep alone does
not.**

### 11.6 Consequence for defense stacking

The bridge result *confirms* the two-axis orthogonality argument the
paper makes in § 10.1: personalization is the right architectural
response to Axis 1 (structural non-IID), and Byzantine-robust
aggregation is the right aggregation-layer response to Axis 2
(adversarial). Neither substitutes for the other. A deployment that
faces both threats must stack both, and the FedRep + Krum-$f_1$
composition is the *recommended* two-axis defense
([research_Paper/paper_draft_v3.md](research_Paper/paper_draft_v3.md),
Table 11 row 7). Evaluating the stacked FedRep + Krum defense
end-to-end remains future work, motivated by this negative bridge
result.

### 11.7 Files added in the bridge experiment

| File | Purpose |
|---|---|
| [scripts/run_rq2_fedrep_under_backdoor.py](scripts/run_rq2_fedrep_under_backdoor.py) | End-to-end runner: FedRep with `client_3` wrapped in `make_backdoor_poisoned_loader`; per-client ASR eval; writes `metrics.json` per seed. |
| [src/fl_aircraft/fl/poisoning.py](src/fl_aircraft/fl/poisoning.py) (extended) | Exports `make_backdoor_poisoned_loader` so `PersonalisedClient` can consume the same trigger-stamped loader that `FederatedClient` uses. |
| [scripts/aggregate_bridge_seeds.py](scripts/aggregate_bridge_seeds.py) | Per-seed → 5-seed aggregator with 95 % CIs. |
| `results/rq_bridge/seed_{42,43,44,45,46}/` | Per-seed FedRep-under-backdoor outputs. |
| [results/rq_bridge/metrics_aggregated.json](results/rq_bridge/metrics_aggregated.json) | 5-seed aggregate (attacker ASR, honest mean/max ASR, attacker−honest delta with 95 % CI). |

---

## TL;DR

1. **The threat**: one (or two) malicious airlines send corrupted weight
   updates to push the global model toward wrong predictions. Server
   cannot inspect client data — only weights.
2. **v1 tested 2 attacks × 4 aggregators = 11 cells (single seed).**
   Vanilla FedAvg collapses under grad ×−10 (RMSE 84.03; F1 0.000).
   Krum-$f_1$ recovers to RMSE 19.80 under both untargeted attacks;
   per-coordinate defenses (trimmed / median) recover only to RMSE
   21–23. Trimmed mean = median bit-identically at $n = 4$ (a
   degenerate coincidence, not a bug).
3. **v2 extended the matrix to 5 attacks × 4 aggregators + baselines
   = 25 cells (§ 9)**, adding: AV3 physically-plausible sensor-value
   backdoor targeting T30; AV4 stealthy grad ×−2; AV5 coordinated
   2-of-4 Byzantine; Krum-$f_2$ (which is *mathematically undefined*
   at $n = 4$ per $n - f - 2 \ge 1$ — the theoretical wall becomes
   an empirical wall).
4. **v2 5-seed aggregation (§ 10)** produces the paper's Axis-2
   headline numbers:
   * Backdoor ASR against vanilla FedAvg: **94.9 ± 7.9 %** — clean
     RMSE (16.86 ± 0.43) statistically indistinguishable from honest
     baseline (16.59 ± 0.84). **Invisible to clean-metric monitoring.**
   * Krum backdoor ASR: **6.4 ± 10.0 %** — 95 % CI reaches zero;
     order-of-magnitude reduction.
   * Coordinated 2-of-4 Byzantine: per-coordinate defenses collapse
     deterministically to RMSE 84.03 ± 0.00; Krum-$f_1$ recovers to
     RMSE 23.97 ± 9.92 despite formally violating its $\le f$
     assumption.
   * Krum's untargeted-attack cells report identical (mean, std) =
     (23.81, 10.01) across attack families because argmin selection
     is attack-agnostic; backdoor + Krum is the exception (std 0.61).
5. **v2 bridge experiment (§ 11)** — FedRep under backdoor over 5
   seeds: attacker-minus-honest ASR delta is $-0.017 \pm 0.060$
   (indistinguishable from zero). **Personalization does not shield
   honest clients** because the poison acts on the shared
   representation, not on the private heads. Krum + FedRep is the
   recommended two-axis defense composition (Table 11 row 7 in the
   paper).
6. **The empirical picture is now**: Axis-1 threats want per-client
   architecture (RQ2 v2); Axis-2 threats want geometric whole-update
   aggregation (Krum). Neither substitutes for the other; both are
   needed in a real deployment.
7. **Caveats worth stating in the writeup**: small $n = 4$ (Krum-$f_2$
   undefined; Multi-Krum / Bulyan not tested), no adaptive attacker
   tested, backdoor trigger is a single fixed configuration (sensitivity
   sweep over $(p, \text{trigger\_value})$ deferred), stacked FedRep +
   Krum not yet evaluated end-to-end (motivated by § 11 but not run).

This is the paper's Axis-2 story. Together with the RQ2 v2 Axis-1
story the two-axis narrative closes: distinct threats, distinct
defenses, both required.

---

## Appendix: artifact pointers

### v1 (11-cell, seed 42)

| Artifact | Path |
|---|---|
| Per-cell metrics | [results/rq7_poisoning/metrics.json](results/rq7_poisoning/metrics.json) |
| Headline RMSE bar chart | [results/rq7_poisoning/headline_comparison_fd001_fd003.png](results/rq7_poisoning/headline_comparison_fd001_fd003.png) |
| **Attack diagnostic (log-scale delta norms)** | [results/rq7_poisoning/attack_diagnostic_delta_norms_fd001_fd003.png](results/rq7_poisoning/attack_diagnostic_delta_norms_fd001_fd003.png) |
| Defense recovery (paired bars) | [results/rq7_poisoning/defense_recovery_fd001_fd003.png](results/rq7_poisoning/defense_recovery_fd001_fd003.png) |
| Per-subset breakdown | [results/rq7_poisoning/per_subset_breakdown_fd001_fd003.png](results/rq7_poisoning/per_subset_breakdown_fd001_fd003.png) |
| Per-round trajectories | `results/rq7_poisoning/per_round_*.csv` |
| Attack wrappers | [src/fl_aircraft/fl/poisoning.py](src/fl_aircraft/fl/poisoning.py) |
| Robust aggregators | [src/fl_aircraft/fl/robust_aggregators.py](src/fl_aircraft/fl/robust_aggregators.py) |
| Poisoned simulation loop | [src/fl_aircraft/fl/poisoned_simulation.py](src/fl_aircraft/fl/poisoned_simulation.py) |
| Experiment CLI | [scripts/run_rq7.py](scripts/run_rq7.py) |
| Unit tests | [tests/test_rq7.py](tests/test_rq7.py) |
| Long-form web story | [frontend/src/pages/Rq7StoryPage.tsx](frontend/src/pages/Rq7StoryPage.tsx) (live at `/rq7-story`) |

### v2 (25-cell extended matrix + 5-seed + bridge)

| Artifact | Path |
|---|---|
| Extended-matrix seed-42 metrics | [results/rq7_poisoning/metrics.json](results/rq7_poisoning/metrics.json) (25 cells) |
| **5-seed aggregate** | [results/rq7_poisoning/metrics_aggregated.json](results/rq7_poisoning/metrics_aggregated.json) |
| Per-seed outputs | `results/rq7_poisoning_seeds/seed_{42,43,44,45,46}/` |
| Multi-seed run log | [results/rq7_poisoning_seeds/multiseed_run.log](results/rq7_poisoning_seeds/multiseed_run.log) |
| Paper Fig. 13 (backdoor ASR bar chart) | [results/paper_figures/fig13_backdoor_asr.png](results/paper_figures/fig13_backdoor_asr.png) |
| Paper Fig. 14 (25-cell 5-seed matrix) | [results/paper_figures/fig14_rq7_matrix_5seed.png](results/paper_figures/fig14_rq7_matrix_5seed.png) |
| Paper Fig. 15 (Krum per-seed dot plot) | [results/paper_figures/fig15_krum_seed_variance.png](results/paper_figures/fig15_krum_seed_variance.png) |
| Multi-seed runner | [scripts/run_rq7_multiseed.py](scripts/run_rq7_multiseed.py) |
| Multi-seed aggregator | [scripts/aggregate_rq7_seeds.py](scripts/aggregate_rq7_seeds.py) |
| Paper figure generator | [scripts/generate_paper_figures.py](scripts/generate_paper_figures.py) |
| **Bridge experiment (FedRep under backdoor)** | `results/rq_bridge/` |
| Bridge 5-seed aggregate | [results/rq_bridge/metrics_aggregated.json](results/rq_bridge/metrics_aggregated.json) |
| Bridge runner | [scripts/run_rq2_fedrep_under_backdoor.py](scripts/run_rq2_fedrep_under_backdoor.py) |
| Bridge aggregator | [scripts/aggregate_bridge_seeds.py](scripts/aggregate_bridge_seeds.py) |
