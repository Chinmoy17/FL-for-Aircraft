# Literature Survey — Federated Learning for Aircraft-Engine PHM under Heterogeneity & Attack

> **Purpose.** Detailed, verified summaries of every paper in
> `research_Paper/Lit_Papers/`, plus how each supports (or bounds) our claims and
> narrative. Read alongside
> [`methods_provenance_and_novelty.md`](methods_provenance_and_novelty.md) (what we
> built) and [`../RESULTS_CONSOLIDATED.md`](../RESULTS_CONSOLIDATED.md) (our numbers).
>
> Each entry: **Problem · Method · Results · Implications · → For our paper.**
> Text is extracted and read from the PDFs (via `pypdf`), not guessed.
>
> **Note:** `FL_RUL_2025.pdf` is a **duplicate** of `2506.00499v1.pdf` (Landau et al.).
> Filenames vs. true arXiv IDs are reconciled per entry (e.g. `Layer_Specific_Backdoor.pdf`
> is arXiv **2602.15161**, not 2605.07860).

---

## 0. Papers at a glance

| # | Paper (year, venue) | Theme | File |
| --- | --- | --- | --- |
| 1 | McMahan et al. 2017 — **FedAvg** (AISTATS) | Foundational FL | `1602.05629` |
| 2 | Li et al. 2020 — **FedProx** (MLSys) | Heterogeneity | `1812.06127` |
| 3 | Pillutla, Kakade, Harchaoui 2022 — **RFA / geometric-median** (IEEE TSP) | Robust aggregation | `1912.13445` |
| 4 | Farhadkhani et al. 2022 — **Poisoning ≡ Byzantine** (ICML) | Robustness theory | `2202.08578` |
| 5 | Nguyen et al. 2024 — **Backdoor survey** (**EAAI**) | Backdoor attacks/defenses | `Backdoor_attack_Comprehensive_study` (arXiv 2303.02213) |
| 6 | Foroughi et al. 2026 — **Layer-Smoothing Attack (LSA)** (IEEE ICC) | Stealthy backdoor | `Layer_Specific_Backdoor` (arXiv 2602.15161) |
| 7 | Fan & Chen 2026 — **Robust Personalized FL** (arXiv) | PFL robustness | `2606.22782` |
| 8 | Landau et al. 2025 — **Aircraft RUL + robust agg** (arXiv) | FL-for-RUL (domain twin) | `2506.00499` = `FL_RUL_2025` |
| 9 | Barbosa et al. 2025 — **Jet-engine RUL** (arXiv) | FL-for-RUL, C-MAPSS | `2502.05321` |
| 10 | Jeong, Yue, Chung 2025 — **Fed-Joint** (arXiv) | FL-for-RUL, turbofan | `FedJoint` (arXiv 2503.13404) |
| 11 | Arunan et al. 2023 — **Matched-feature FL prognostics** (IEEE TASE) | FL-for-PHM, heterogeneity | `FL_Heteroginity_2023` |
| 12 | Milasheuski et al. 2026 — **Generative FL for PdM** (IEEE) | FL-for-PdM, partial federation | `2605.07860` |
| 13 | Tallat et al. 2026 — **BioMutFed+** (IEEE **TMC**) | C-MAPSS (FD004) + FL + Byzantine robustness — **nearest overlap** | `BioMutFed_...` |

---

## 1. Foundational federated learning

### 1.1 FedAvg — McMahan et al., 2017 (AISTATS) · `1602.05629`
- **Problem.** Train a shared model from decentralized on-device data without moving raw data to a server.
- **Method.** Clients run local SGD; the server takes a **sample-count-weighted average** of client weights each round. Communication-efficient (fewer rounds via more local work).
- **Results.** Matches centralized quality on IID/mildly-non-IID vision/language benchmarks at far lower communication.
- **Implications.** The canonical FL baseline — and the aggregation rule every robust method replaces.
- **→ For our paper.** Our **FedAvg baseline** (IID 14.16 / non-IID 17.95 RMSE). The **non-IID failure** motivates Axis-1 (personalization); FedAvg's averaging is exactly what Krum/trimmed/median/RFA replace in Axis-2.

### 1.2 FedProx — Li et al., 2020 (MLSys) · `1812.06127`
- **Problem.** FedAvg drifts and destabilizes under **statistical + systems heterogeneity**.
- **Method.** Add a **proximal term** `(μ/2)‖w − w_global‖²` to each client's local objective, penalizing drift from the round-start global model; tolerates partial/variable local work.
- **Results.** More stable, modestly more accurate than FedAvg on heterogeneous benchmarks.
- **Implications.** Aggregation-/objective-layer fix for heterogeneity — cheap but limited.
- **→ For our paper.** We implement FedProx faithfully (μ-sweep). Result: **17.70 RMSE — only ~+6% of the gap closed**, our evidence that *objective-layer tweaks are insufficient* and that the cure lives at the **architecture layer** (FedRep).

---

## 2. Robust aggregation & robustness theory (Axis 2 foundations)

### 2.1 RFA (geometric-median aggregation) — Pillutla, Kakade, Harchaoui, 2022 (IEEE TSP) · `1912.13445`
- **Problem.** The arithmetic mean is **not robust** — a *single* corrupted update degrades the global model for everyone.
- **Method.** Replace the mean with an **approximate geometric (spatial/L1) median**, computed by a smoothed **Weiszfeld** alternating-minimization that reuses **privacy-preserving secure-aggregation** primitives; ~1–3× communication cost. Two variants: one-step, and an **on-device personalization** variant.
- **Results.** Robust to corruption in **up to half** the devices under *bounded heterogeneity*; beats the mean when corruption is high, competitive when low; validated on CV/NLP with linear/CNN/RNN models.
- **Implications.** A principled robust aggregator **alternative to Krum / trimmed-mean / median**, and it explicitly pairs robustness with **personalization**.
- **→ For our paper.** (a) A **robust aggregator we did *not* test** — natural comparison / future-work baseline against our Krum. (b) Its personalization variant means **robustness+personalization has been combined before (in spirit)** — so **SHARP's composition is not unprecedented**; our contribution is the *domain* (aeroengine RUL) + the *encoder-only Krum* variant, not the idea of composing them. Cite to stay honest. (c) "Bounded heterogeneity" caveat mirrors our finding that robust rules strain under structural non-IID.

### 2.2 An Equivalence Between Data Poisoning and Byzantine Gradient Attacks — Farhadkhani et al., 2022 (ICML) · `2202.08578`
- **Problem.** Is the strong "Byzantine" threat model (arbitrary gradients) realistic, and what does robustness cost under heterogeneity?
- **Method.** Prove that **every gradient (Byzantine) attack reduces to data poisoning** in any personalized-FL system with PAC guarantees; derive **impossibility results** for robust learning under **high heterogeneity**.
- **Results.** Data poisoning (realistic) is as powerful as Byzantine attacks; strong robustness is **provably hard** when clients are very heterogeneous; a practical attack is derived and shown effective against classical PFL.
- **Implications.** Formal backbone linking *poisoning* ↔ *Byzantine*, and a warning that **heterogeneity and robustness are in tension**.
- **→ For our paper.** (a) Theoretical grounding for our **two-axis framing** (poisoning and Byzantine are the same coin). (b) Supports our empirical finding that **defenses weaken as heterogeneity/difficulty grows** (FD002+4). (c) Rigorous support that **personalized FL is not automatically robust** — cite next to Fan & Chen for our FedRep-vulnerability result.

### 2.3 BioMutFed+ (bio-inspired robust FL) — Tallat et al., 2026 (IEEE **TMC**) · `BioMutFed_...` — **nearest overlap**
- **Problem.** FL in IIoT / Industry 5.0 faces non-IID data, adversarial (Byzantine / model-poisoning) threats, and scalability limits.
- **Method.** **BioMutFed+**, a biologically-inspired FL framework combining (1) adaptive **mutation-based** gradient perturbations, (2) **pheromone-driven** adaptive client selection, and (3) robust **trimmed-mean** aggregation; convergence proved at O(1/√T). A lightweight variant, **FedMutAdam**, adds adaptive gradient clipping + server-side Adam for resource-constrained devices.
- **Results.** On **Digits** (vision), **NASA FD004** (predictive maintenance), and **Wisconsin Breast Cancer**: convergence within ~20 rounds, **1.2–2.9× variance reduction** vs established FL, and improved robustness under **20% malicious clients**.
- **Implications.** Bio-inspired perturbation + robust aggregation can harden FL against generic model poisoning across mixed benchmarks.
- **→ For our paper — closest work, but NOT a true twin (position carefully).** It shares **C-MAPSS (FD004) + FL + adversarial robustness**, so a reviewer *will* cite it. **But:** (a) FD004 is **one of three generic benchmarks**, not a prognostics study — **no RUL modelling, no per-fault-mode heterogeneity analysis**; (b) robustness = **model-poisoning/Byzantine measured by variance/convergence**, **not a failure-masking backdoor, not ASR, not the accuracy-is-blind safety point**; (c) it adds **bio-inspired metaphors** (mutation, pheromone) — a style EAAI is explicitly wary of. **Suggested positioning:** *"BioMutFed+ (Tallat et al., 2026) hardens FL on mixed IIoT benchmarks including C-MAPSS FD004 via bio-inspired perturbation and trimmed-mean aggregation; we instead target aeroengine RUL specifically, show that accuracy is blind to a failure-masking backdoor, and analyse the personalization–robustness tension — none of which that work addresses."*
- **Correction to earlier notes:** this paper is in **IEEE Transactions on Mobile Computing (TMC)**, DOI **10.1109/TMC.2026.3707971** — *not* IEEE TII, and it is **less of a direct competitor than previously implied**.

---

## 3. Backdoor attacks & the "accuracy is blind" theme (our core safety point)

### 3.1 Backdoor Attacks and Defenses in FL: Survey — Nguyen et al., 2024 (**EAAI**) · `Backdoor_attack_Comprehensive_study` (arXiv 2303.02213)
- **Problem.** Backdoors — a trigger makes the model misbehave on chosen inputs while **appearing normal otherwise** — are under-studied in FL specifically.
- **Method.** Systematic literature review; taxonomy of **attacks** (data-poisoning vs model-poisoning) and **defenses** by phase: **pre-aggregation, in-aggregation (robust aggregators), post-aggregation**.
- **Results.** Documents that FL-backdoor research is **growing exponentially** (their Fig. 1); notes most defenses are **"attack-driven"** — they only stop *known* attacks and can be **circumvented** by an adversary aware of them.
- **Implications.** The field is important, active, and defenses are not future-proof.
- **→ For our paper.** **High-value citation — it's in our target journal (EAAI).** Use it to (a) establish the topic's importance/growth, (b) place our backdoor as **data-poisoning** with an **in-aggregation** (Krum) defense in their taxonomy, and (c) support the honest caveat that robust aggregation is **attack-specific**, not a universal shield.

### 3.2 Exploiting Layer-Specific Vulnerabilities (Layer-Smoothing Attack, LSA) — Foroughi et al., 2026 (IEEE ICC) · `Layer_Specific_Backdoor` (arXiv 2602.15161)
- **Problem.** Existing defenses treat the network as a **black box**, ignoring that a few **layers** carry most of the backdoor.
- **Method.** **Layer Substitution Analysis** identifies **backdoor-critical (BC) layers** (swap a malicious layer into a benign model, measure the backdoor-success drop). **LSA** poisons *only* BC-layers, fine-tunes with non-critical layers frozen, and uses **weight approximation to benign updates** (a λ-controlled, ReLU-smoothed perturbation) so malicious updates look statistically normal.
- **Results.** **Up to 97% backdoor success rate while maintaining high main-task accuracy**, **bypassing SOTA defenses including Multi-Krum, Trimmed-Mean, and FLAME**, across architectures/datasets.
- **Implications.** Two big ones: (1) **accuracy is blind to backdoors** — you can have high accuracy and a 97% backdoor; (2) **robust aggregators (Krum, trimmed-mean) are bypassable** by layer-targeted, benign-mimicking attacks.
- **→ For our paper.** **The strongest external support for our thesis** that *RMSE/accuracy alone cannot certify safety — you must measure attack-success*. **But it cuts both ways and you must state this honestly:** our Krum defense works against our *magnitude/geometry-detectable* backdoor; **an LSA-style adaptive attack would likely evade it.** Cite as (a) motivation for the ASR metric **and** (b) an explicit **limitation / future-threat** — our defense is not future-proof against adaptive, layer-aware adversaries.

---

## 4. Personalized-FL robustness (our FedRep-vulnerability claim)

### 4.1 Towards Robust Personalized FL: Vulnerability Assessment and Defense Co-Design — Fan & Chen, 2026 · `2606.22782`
- **Problem.** Personalized FL (PFL) is adopted for heterogeneity — but is it **secure**? Identified blind spot: PFL under adversarial attack.
- **Method.** Systematic + theoretical analysis showing PFL is **more vulnerable to transfer-based adversarial (evasion) attacks** than centralized learning: a malicious client uses its own personalized model (same architecture, different params — a **gray-box** setting) to craft **adversarial examples** that transfer to peers' models. Defense: stochastic input noise + input-scaled trace regularization + parameter-sensitivity maximization.
- **Results.** Significant accuracy drops across PFL methods; "**first systematic study of adversarial threats in PFL**."
- **Implications.** Personalization does **not** confer robustness.
- **→ For our paper.** Supports our **"personalization alone is not robust"** finding — **but read the threat model carefully.** Fan & Chen attack with **test-time adversarial examples (evasion)**; **we attack with a training-time backdoor (poisoning).** They are **complementary, not identical.** Cite as *"PFL vulnerability has been shown for evasion attacks (Fan & Chen 2026); we show a distinct, training-time backdoor vulnerability of representation-personalized FL (FedRep) in prognostics."* This distinction actually **strengthens** our novelty (different attack class, different domain).

---

## 5. FL for PHM / RUL / predictive maintenance (domain landscape)

### 5.1 Federated learning for collaborative RUL prognostics: an aircraft engine case study — Landau et al., 2025 · `2506.00499` (= `FL_RUL_2025`)
- **Problem.** Airlines lack enough run-to-failure data individually and won't share raw data; also sensor data is **noisy**.
- **Method.** FL for RUL with a **decentralized validation** procedure (each client validates on its own held-out set, shares only loss) and **four robust aggregation policies** (Full/Random validation × Best-model/Softmax) — the Best-model policy is a **Blanchard/Krum-style selection**. 1-D CNN on **N-CMAPSS** (6 airlines train / 3 test engines), **80/20 within-engine train/validation split**.
- **Results.** FL beats isolated training for 5/6 airlines (RMSE ~9.9 "flights" vs ~15.8); robust policies help **when clients have noisy data**.
- **Implications.** Robust aggregation is useful for **noise-robust** aircraft FL RUL, and validation matters for aviation certification (EASA).
- **→ For our paper. This is your closest neighbor — position against it explicitly.** They do **noise robustness**; you do **adversarial/backdoor robustness**. They use N-CMAPSS + a validation split; you use classic C-MAPSS + (documented) final-epoch reporting. "FL for aircraft RUL + robust aggregation" is **already published**, so your differentiators are the **adversarial threat, the failure-masking/RMSE-blindness safety finding, and the personalization+robustness composition.** Also the source of your validation-split discussion.

### 5.2 Using Federated ML in Predictive Maintenance of Jet Engines — Barbosa et al., 2025 · `2502.05321`
- **Problem.** Predict jet-engine RUL collaboratively without sharing data.
- **Method.** FedAvg-style FL with a nonlinear model on **C-MAPSS** (publicly available NASA data).
- **Results.** Demonstrates FL RUL on C-MAPSS; positions FL for aviation predictive maintenance.
- **Implications.** Confirms C-MAPSS + FL RUL as an active, credible setting.
- **→ For our paper.** A clean **domain baseline** citation ("FL RUL on C-MAPSS has been demonstrated") — reinforces that your *setting* is standard, so your contribution is the **robustness/safety layer**, not the demonstration of FL-RUL itself.

### 5.3 Fed-Joint — Jeong, Yue, Chung, 2025 · `FedJoint` (arXiv 2503.13404)
- **Problem.** RUL from condition-monitoring signals that are **not parametrically specified**, held at different sites, unshareable.
- **Method.** **Federated joint modeling** of nonlinear degradation signals (a **federated multi-output Gaussian process**) *and* time-to-failure (a **federated survival model**); nonparametric.
- **Results.** Outperforms alternatives in simulation + a **turbofan** run-to-failure case study.
- **Implications.** A statistically principled (non-deep) branch of FL prognostics.
- **→ For our paper.** Shows the **breadth** of FL-RUL methods (statistical, not just deep nets) and that **turbofan RUL is a standard testbed**. Contrast: your work is **deep, multi-task, and adversarially-focused** — Fed-Joint has no attacker.

### 5.4 Matched-feature FL industrial health prognostics — Arunan et al., 2023 (IEEE TASE) · `FL_Heteroginity_2023`
- **Problem.** FL health prognostics with **heterogeneous edge devices** (dissimilar degradation, unequal dataset sizes) — naive averaging dilutes distinct local features.
- **Method.** A **feature-similarity-matched parameter aggregation** — match neurons with similar feature-extraction functions *before* averaging, so distinct local extractors aren't smeared.
- **Results.** Up to **44.5% / 39.3%** improvement for state-of-health / RUL estimation on Li-ion battery + **turbofan** data vs naive averaging.
- **Implications.** Heterogeneity-aware aggregation matters a lot in PHM — a middle ground between plain FedAvg and full personalization.
- **→ For our paper.** A strong **Axis-1 (heterogeneity)** comparison point in the *same domain family*; supports your claim that **structural heterogeneity breaks naive averaging** and motivates representation-level fixes (your FedRep). No adversary — again, your security axis is the gap.

### 5.5 On the Tradeoffs of On-Device Generative Models in Federated PdM — Milasheuski et al., 2026 (IEEE) · `2605.07860`
- **Problem.** Generative models (VAE, GAN, Diffusion) for **unsupervised anomaly detection** in federated predictive maintenance — utility vs communication vs heterogeneity tradeoffs.
- **Method.** Compare VAE/GAN/DM under **full vs partial federation** (sharing only subsets of model components); propose a **taxonomy where partial component sharing = personalization**.
- **Results.** Distinct utility/stability/scalability tradeoffs; for **Diffusion Models, partial federation (decoder sharing) can beat full federation** in bandwidth-constrained non-IID settings.
- **Implications.** **Partial component sharing** is a principled personalization lever in PdM.
- **→ For our paper.** Conceptually **parallel to your FedRep split** (share the encoder, keep heads private = partial federation for personalization) — cite as independent support that **partial federation/personalization is the right tool for PdM heterogeneity**. Different task (unsupervised generative AD vs your supervised multi-task RUL).

---

## 6. Claims-support map (which paper backs which claim)

| Our claim | Supported by (in folder) |
| --- | --- |
| **Accuracy/RMSE is blind to backdoors — measure ASR** | **Foroughi (97% BSR + high accuracy)**, Nguyen survey (stealth by design) |
| FL backdoor threat is **real and growing** | Nguyen (exponential growth), Foroughi |
| **Robust aggregation** defends poisoning | RFA (geometric median), Farhadkhani (theory); (Krum/Yin — to collect) |
| **Personalization alone ≠ robust** | Fan & Chen (evasion on PFL), Farhadkhani (poisoning≡Byzantine in PFL) |
| Robust rules assume ~IID; **strain under heterogeneity** | Farhadkhani (impossibility), RFA ("bounded heterogeneity") |
| Naive averaging **breaks under structural heterogeneity** | Arunan (matched features), FedProx (limited), Milasheuski |
| **Partial federation/personalization** fits PdM | Milasheuski (partial sharing), Arunan, Landau |
| FL for **aeroengine/turbofan RUL** is established | Landau, Barbosa, Fed-Joint, Arunan |
| **Defenses are attack-driven / not future-proof** (honest caveat) | **Foroughi (LSA bypasses Krum/trimmed/FLAME)**, Nguyen |

---

## 7. The gap we fill (what none of these do)

No paper in this set combines **all** of:
1. a **failure-masking backdoor on federated aeroengine RUL *regression*** (not classification),
2. the **safety framing** that accuracy is blind → ASR is mandatory,
3. a **systematic robust-aggregation characterization under structural fault-mode non-IID** C-MAPSS, and
4. a **personalization + robust-aggregation composition** (SHARP) in this domain.

- Landau = **noise**, not adversarial; N-CMAPSS.
- Barbosa / Fed-Joint / Arunan / Milasheuski = FL-PHM with **no attacker**.
- Nguyen / Foroughi / Fan & Chen / RFA / Farhadkhani = strong on attacks/robustness but **not PHM/RUL** (vision/NLP/IoT classification).
- **BioMutFed+ (Tallat 2026) = the closest overlap** — C-MAPSS (FD004) + FL + Byzantine robustness — but FD004 is *one of three generic benchmarks*, robustness is *variance/convergence under model poisoning* (bio-inspired mutation + trimmed-mean), with **no RUL analysis, no backdoor, and no accuracy-blindness / ASR safety framing.**

**Honest boundary:** items (1)–(2) are your firmest novelty; (3)–(4) are *domain-first* combinations of existing tools (RFA already paired robustness+personalization; Foroughi shows adaptive attacks can beat robust aggregation). Claim domain-first, not method-first.

---

## 8. Referenced but **not yet in the folder** (please download → `Lit_Papers/`)

I could not find these in the folder; several are central to the related-work and to
attribution. Titles/venues are stable; arXiv IDs marked *(verify)* should be
confirmed before citing (I will not fabricate identifiers).

| Paper | Why we need it | Link / ID |
| --- | --- | --- |
| Blanchard et al. 2017 — **Krum** ("Machine Learning with Adversaries: Byzantine-Tolerant Gradient Descent") | We *use* Krum — must cite the source | NeurIPS 2017 (search by title) |
| Yin et al. 2018 — **Trimmed-mean / coordinate-median** | We *use* both | ICML 2018, arXiv **1803.01498** *(verify)* |
| Collins et al. 2021 — **FedRep** | Our MT-FedRep extends it | ICML 2021, arXiv **2102.07078** |
| Chen et al. 2024 — **FedCCFA** | We implement a simplified variant | NeurIPS 2024 (search by title — *verify*) |
| Gu et al. 2017 — **BadNets** | Origin of trigger backdoors | arXiv **1708.06733** |
| Bagdasaryan et al. 2020 — **How To Backdoor Federated Learning** | Model-replacement backdoor baseline | AISTATS 2020, arXiv **1807.00459** |
| Xie et al. 2020 — **DBA (Distributed Backdoor Attack)** | Coordinated-attacker baseline | ICLR 2020 (search by title) |
| Zhang et al. 2024 — **SARS** (personalized-FL robustness) | We cite for "personalization ≠ robust" | *(verify title/venue)* |
| Chuang & Zhang 2026 — adaptive robust aggregation vs poisoning/backdoor | Closest **EAAI** security paper | EAAI, **DOI 10.1016/j.engappai.2026.114734** |

> If any of these are paywalled, send me the title/arXiv and I'll fold a full summary
> into this file once the PDF is in `Lit_Papers/`.

---

## 9. How to use this in the paper

- **Related Work spine:** (a) FL foundations [FedAvg, FedProx] → (b) heterogeneity/personalization in PHM [Arunan, Milasheuski, Landau; FedRep to-collect] → (c) robustness/backdoors [RFA, Farhadkhani, Nguyen, Foroughi; Krum/Yin to-collect] → (d) the intersection we occupy (§7 gap).
- **Motivation (Introduction):** lead with **Foroughi + Nguyen** for "accuracy is blind, threat is growing," then narrow to aeroengine RUL [Landau, Barbosa].
- **Honesty guardrails baked in above:** Fan & Chen = evasion (not backdoor); RFA already composed robustness+personalization; Foroughi/LSA can bypass Krum (state as limitation); Landau already did robust-agg aircraft FL RUL (for noise).

*All summaries derived from the PDF text in `Lit_Papers/`. Verify every citation's
final metadata (year/venue/pages) against the publisher record before submission.*
