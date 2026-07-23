# Reading list for the "FL-for-Aircraft" journal paper

Curated after a targeted literature sweep across arXiv, IEEE Xplore, ACM DL, Wiley,
Springer, MDPI, PHM Society, USENIX (July 2026). Papers are grouped by the section
of our paper where they will be cited. Direct PDF links are given where the paper
is open-access; otherwise the DOI / venue is listed so you can pull it through
your institution's library.

Legend:
- **[OA]**  = open access — direct download works
- **[PW]**  = paywalled — get via institution / author webpage
- **[MUST]** = must-cite; closest work to ours
- **[FDN]** = foundational method we build on
- **[REL]** = related but distant; cite for context

---

## A. Two closest competitors — MUST cite & differentiate

1. **[MUST] [PW] Tallat, R. et al. (2026).**
   *BioMutFed+: Mutation-Driven Federated Learning for IIoT.*
   IEEE Transactions on Industrial Informatics.
   - DOI / URL: https://ieeexplore.ieee.org/abstract/document/11581414/
   - Why: Uses **C-MAPSS as testbed**, evaluates a mutation-driven aggregator against
     gradient-ascent poisoning with 20 % malicious clients. Our study extends
     this to a 4-attack × 4-aggregator matrix with backdoor + multi-task.

2. **[MUST] [PW] Li, H. (2026).**
   *Trustworthy Federated Learning for Industrial IoT: Balancing Robustness
   and Fairness via Blockchain-Based Reputation.*
   IET / Wiley — Artificial Intelligence for Engineering.
   - DOI / URL: https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/aie2.70012
   - Why: Uses **C-MAPSS turbofan**, introduces gradient magnitude clipping
     against "sophisticated scaling attacks". Complements our per-coordinate
     robust aggregator comparison; no backdoor, no coordinated attackers.

---

## B. FL for aircraft-engine RUL / prognostics — baseline domain

3. **[OA] Barbosa, A. M. et al. (2025).**
   *Using Federated Machine Learning in Predictive Maintenance of Jet Engines.*
   arXiv:2502.05321.
   - PDF: https://arxiv.org/pdf/2502.05321
   - Why: Most recent vanilla FL + C-MAPSS + RUL paper. Use as the "prior
     work does FL on C-MAPSS but ignores security" reference.

4. **[OA] Söderkvist Vermelin, W., Mishra, M., Eng, M. P. et al. (2024).**
   *Collaborative Training of Data-Driven Remaining Useful Life Prediction
   Models Using Federated Learning.*
   PHM Society European Conference.
   - PDF: https://www.researchgate.net/publication/384643737
   - Why: Foundational open-access FL+C-MAPSS+RUL reference from the PHM
     community. Cite for "prior FL-RUL work is entirely benign."

5. **[OA] Pandhare, V., Jia, X., Lee, J. (2021).**
   *Collaborative Prognostics for Machine Fleets Using a Novel Federated
   Baseline Learner.*
   Annual Conf. of the PHM Society.
   - PDF: http://papers.phmsociety.org/index.php/phmconf/article/view/2989
   - Why: Introduces **structural non-IID by operating conditions** for FL
     prognostics — closest precedent for our FD001+FD003 partition. No attacks.

6. **[OA] Milasheuski, U., Baraldi, P., Zio, E., Savazzi, S. (2026).**
   *On the Tradeoffs of On-Device Generative Models in Federated Predictive
   Maintenance Systems.*
   arXiv:2605.07860.
   - PDF: https://arxiv.org/pdf/2605.07860
   - Why: Recent (2026) FL+PdM overview covering VAE/GAN/DM tradeoffs.
     Only mentions backdoor defense in related work.

7. **[REL] [PW] Rehman, M. H. et al. (2021).**
   *TrustFed: A Framework for Fair and Trustworthy Cross-Device FL in IIoT.*
   IEEE Transactions on Industrial Informatics.
   - PDF (author copy): https://www.academia.edu/download/81768765/09416805.pdf
   - Why: One of the first FL+turbofan-engine papers. Focus is trust/fairness
     at the client-selection level, not gradient-space attacks.

---

## C. Foundational FL algorithms — FDN

8. **[FDN] [OA] McMahan, H. B. et al. (2017).**
   *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
   AISTATS 2017. arXiv:1602.05629.
   - PDF: https://arxiv.org/pdf/1602.05629
   - Why: Original FedAvg paper.

9. **[FDN] [OA] Li, T. et al. (2020).**
   *Federated Optimization in Heterogeneous Networks (FedProx).*
   MLSys 2020. arXiv:1812.06127.
   - PDF: https://arxiv.org/pdf/1812.06127
   - Why: RQ3 baseline.

10. **[FDN] [OA] Collins, L., Hassani, H., Mokhtari, A., Shakkottai, S. (2021).**
    *Exploiting Shared Representations for Personalized Federated Learning (FedRep).*
    ICML 2021. arXiv:2102.07078.
    - PDF: https://arxiv.org/pdf/2102.07078
    - Why: RQ2 baseline.

---

## D. Byzantine-robust aggregators — FDN

11. **[FDN] [OA] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., Stainer, J. (2017).**
    *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Krum).*
    NeurIPS 2017.
    - PDF: https://papers.nips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
    - Why: Original Krum; we compare f=1 vs f=2 under coordinated attackers.

12. **[FDN] [OA] Yin, D., Chen, Y., Ramchandran, K., Bartlett, P. (2018).**
    *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.*
    ICML 2018. arXiv:1803.01498.
    - PDF: https://arxiv.org/pdf/1803.01498
    - Why: Coordinate median + trimmed mean, with the theoretical rates we
      reference when explaining our aggregator choice.

13. **[FDN] [OA] Pillutla, K., Kakade, S., Harchaoui, Z. (2022).**
    *Robust Aggregation for Federated Learning (RFA — Geometric Median).*
    IEEE Transactions on Signal Processing. arXiv:1912.13445.
    - PDF: https://arxiv.org/pdf/1912.13445
    - Why: Modern robust-aggregation paper you'll want in the related-work
      table even if you don't run RFA yourself.

---

## E. Poisoning & backdoor attacks on FL — FDN

14. **[FDN] [OA] Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., Shmatikov, V. (2020).**
    *How to Backdoor Federated Learning.*
    AISTATS 2020. arXiv:1807.00459.
    - PDF: https://arxiv.org/pdf/1807.00459
    - Why: The reference paper for FL backdoors. Our sensor-value trigger is
      an industrial-domain instantiation of this threat model.

15. **[FDN] [OA] Bhagoji, A. N., Chakraborty, S., Mittal, P., Calo, S. (2019).**
    *Analyzing Federated Learning Through an Adversarial Lens.*
    ICML 2019. arXiv:1811.12470.
    - PDF: https://arxiv.org/pdf/1811.12470
    - Why: Foundational adversarial analysis of FedAvg.

16. **[FDN] [OA] Fang, M., Cao, X., Jia, J., Gong, N. (2020).**
    *Local Model Poisoning Attacks to Byzantine-Robust Federated Learning.*
    USENIX Security 2020. arXiv:1911.11815.
    - PDF: https://arxiv.org/pdf/1911.11815
    - Why: Best-known model-poisoning attack; cite to justify why we test
      Krum / trimmed-mean / median rather than assuming they suffice.

17. **[FDN] [OA] Xie, C., Huang, K., Chen, P.-Y., Li, B. (2020).**
    *DBA: Distributed Backdoor Attacks against Federated Learning.*
    ICLR 2020.
    - OpenReview: https://openreview.net/forum?id=rkgyS0VFvr
    - Why: Coordinated-attackers precedent for our AV5+D51-D54 cells.

18. **[REL] [OA] Sun, Z., Kairouz, P., Suresh, A. T., McMahan, H. B. (2019).**
    *Can You Really Backdoor Federated Learning?*
    arXiv:1911.07963.
    - PDF: https://arxiv.org/pdf/1911.07963
    - Why: The counter-argument paper. Cite in discussion when interpreting
      how norm clipping affects our backdoor success rate.

---

## F. Personalized FL + backdoor (RQ2 territory)

19. **[REL] [PW] Zhang, W., Li, Y., An, L., Wan, B., Wang, X. (2024).**
    *SARS: A Personalized FL Framework Towards Fairness and Robustness
    Against Backdoor Attacks.*
    Proc. ACM IMWUT.
    - DOI: https://dl.acm.org/doi/abs/10.1145/3678571
    - Why: Uses FedRep as a comparison baseline; helpful when you discuss
      whether FedRep's personal head buys robustness in prognostics.

20. **[REL] [OA] Fan, M., Chen, C. (2026).**
    *Towards Robust Personalized Federated Learning: Vulnerability Assessment
    and Defense Co-Design.*
    arXiv:2606.22782.
    - PDF: https://arxiv.org/pdf/2606.22782
    - Why: 2026 systematic PFL vulnerability survey — good for framing the
      FedRep-under-attack cross-cut (even though we defer that experiment).

---

## G. Time-series & industrial backdoor triggers (RQ7 trigger design)

21. **[REL] [PW] Yang, D., Luo, S., Zhou, J., Pan, L., Yang, X., Xing, J. (2023).**
    *Efficient and Persistent Backdoor Attack by Boundary Trigger Set
    Constructing Against Federated Learning.*
    Information Sciences.
    - DOI: https://www.sciencedirect.com/science/article/pii/S0020025523013282
    - Why: Non-image trigger design in FL.

22. **[REL] [PW] Lyu, X. et al. (2024).**
    *CoBA: Collusive Backdoor Attacks with Optimized Trigger to Federated
    Learning.*
    IEEE Trans. Dependable and Secure Computing.
    - DOI: https://ieeexplore.ieee.org/abstract/document/10638807/
    - Why: Learned collusive triggers — contrast with our physically-
      interpretable fixed sensor-value trigger.

23. **[REL] [OA] Burbano, L., Sasahara, H., Song, R., Celik, Z. B. et al. (2026).**
    *BADControl: Backdoor Attacks Against Control Systems.*
    USENIX Security 2026.
    - PDF: https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_burbano.pdf
    - Why: **Philosophically the closest** paper on "physical triggers" in
      cyber-physical systems. Non-FL, but you can borrow their threat-model
      framing wholesale.

---

## H. FL + IIoT + Byzantine (adjacent domain grounding)

24. **[REL] [PW] Li, S., Ngai, E., Voigt, T. (2021).**
    *Byzantine-Robust Aggregation in Federated Learning Empowered
    Industrial IoT.*
    IEEE Transactions on Industrial Informatics. (120+ citations)
    - DOI: https://ieeexplore.ieee.org/abstract/document/9614992/
    - Why: The cornerstone FL+IIoT+Byzantine paper. Not prognostics, but
      the closest large-scale precedent for our framing.

25. **[REL] [PW] Hou, B., Gao, J., Guo, X., Baker, T., Zhang, Y. et al. (2021).**
    *Mitigating the Backdoor Attack by Federated Filters for Industrial
    IoT Applications.*
    IEEE Transactions on Industrial Informatics.
    - DOI: https://ieeexplore.ieee.org/abstract/document/9536411/
    - Why: Closest FL+IIoT+backdoor defense paper.

---

## I. Surveys — cite once in Related Work

26. **[OA] Berghout, T., Benbouzid, M., Bentrcia, T., Lim, W. H., Amirat, Y. (2022).**
    *Federated Learning for Condition Monitoring of Industrial Processes:
    A Review on Fault Diagnosis Methods, Challenges, and Prospects.*
    MDPI Electronics.
    - HTML/PDF: https://www.mdpi.com/2079-9292/12/1/158
    - Why: Comprehensive FL-for-PHM survey; establishes that prior FL-PHM
      work is dominated by classification (fault diagnosis) not RUL.

27. **[OA] Djemaa, A., Djenouri, D., Legg, P. (2026).**
    *Heterogeneity-Aware Poisoning Attacks and Mitigation in Federated
    Learning: A Comprehensive Survey and Taxonomy.*
    MDPI Electronics.
    - HTML/PDF: https://www.mdpi.com/2079-9292/15/13/2876
    - Why: Recent (2026) taxonomy of FL poisoning under heterogeneity —
      directly supports our "structural non-IID + attack" positioning.

28. **[OA] Kairouz, P. et al. (2021).**
    *Advances and Open Problems in Federated Learning.*
    Foundations and Trends in ML. arXiv:1912.04977.
    - PDF: https://arxiv.org/pdf/1912.04977
    - Why: The canonical FL survey — cite in intro once for framing.

---

## J. Original C-MAPSS dataset paper

29. **[OA] Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008).**
    *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.*
    IEEE Int. Conf. on Prognostics and Health Management (PHM 2008).
    - Direct PDF: https://www.phmsociety.org/sites/phmsociety.org/files/phm_submission/2008/phmc_08_50.pdf
    - Why: The original C-MAPSS paper. Mandatory citation for dataset provenance.

---

## K. RUL evaluation metrics

30. **[OA] Saxena, A., Celaya, J., Balaban, E., Goebel, K., Saha, B., Saha, S., Schwabacher, M. (2008).**
    *Metrics for Evaluating Performance of Prognostic Techniques.*
    IEEE Int. Conf. on Prognostics and Health Management (PHM 2008).
    - PDF: https://ti.arc.nasa.gov/publications/1863/download/ (may need mirror)
    - Alt search: Google Scholar for the title — several mirrors exist.
    - Why: Source of the NASA scoring function we use.

---

## Download priority

**Tier 1 (download first — read before writing intro / related work):**
1, 2, 3, 5, 8, 11, 12, 14, 16, 24, 29

**Tier 2 (download next — methods & threat model sections):**
9, 10, 13, 15, 17, 22, 23, 25, 26, 27

**Tier 3 (nice-to-have — for discussion & future work):**
4, 6, 7, 18, 19, 20, 21, 28, 30

---

## After downloading

- Store PDFs in this folder using the naming pattern
  `NN_firstauthor_year_shortkey.pdf` — e.g. `01_tallat_2026_biomutfedplus.pdf`.
- Keep this list open when reading; annotate with 1-line comments as you
  go, so we can lift them straight into the related-work section later.
- I can start drafting the **related-work section** and the
  **intro novelty paragraph** as soon as the seed-42 rerun finishes and
  the multi-seed sweep is queued.
