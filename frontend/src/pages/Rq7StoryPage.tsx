import { useEffect, useState } from "react";
import { api } from "../api";
import {
  AnchorStat,
  Bullet,
  FormulaBlock,
  HypothesisCard,
  SmokingGunFigure,
  StoryHero,
  StorySection,
} from "../components/story";
import type { PhaseMetrics, ProjectSummary } from "../summaryTypes";

/**
 * Long-form story page for RQ7 — model poisoning attacks + Byzantine-robust
 * aggregation defenses.
 *
 * Sibling of /rq2-story and /rq3-story. RQ7 sits on the *security* side
 * of the project's research-question split. The story arc:
 *
 *   1. The threat — a malicious airline (client_3 on FD003) wants the
 *      global model to predict its competitor's FD001 engines as
 *      healthier than they really are.
 *   2. Two attacks (label flip, boosted gradient ×-10) tested.
 *   3. Three Byzantine-robust aggregators (trimmed mean, median, Krum)
 *      tested against both attacks.
 *   4. Verdict: gradient scaling is catastrophic (RMSE explodes to 84);
 *      Krum recovers within 1.85 RMSE of the clean baseline.
 *
 * Follows the same UI-craft "trust" template as the other story pages
 * (Instrument Serif hero, 3 anchor stats, sections, smoking-gun figure,
 * synthesis at the end).
 */
export function Rq7StoryPage() {
  const [phase, setPhase] = useState<PhaseMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getSummary()
      .then((res) => {
        if (cancelled) return;
        const p: ProjectSummary = res.summary;
        const rq7 = p.phases["rq7_poisoning"] ?? null;
        if (!rq7) {
          setErr(
            "RQ7 phase not found in summary.json. Run scripts/run_rq7.py first.",
          );
          return;
        }
        setPhase(rq7);
      })
      .catch((e: Error) => {
        if (!cancelled) setErr(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (err) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12">
        <div className="rounded-md border border-bad bg-bad/10 px-4 py-3 text-sm text-bad">
          {err}
        </div>
      </div>
    );
  }
  if (!phase) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-12 text-text-dim">
        <span className="spinner" /> Loading RQ7 results…
      </div>
    );
  }

  return <Rq7Article phase={phase} />;
}

function Rq7Article({ phase }: { phase: PhaseMetrics }) {
  return (
    <article className="mx-auto px-6 py-10">
      {/* HERO ----------------------------------------------------------- */}
      <StoryHero
        eyebrow="Security finding · RQ7"
        lead={
          <>
            What happens when one of the four airlines decides to cheat?
            We made one client malicious and ran two attack strategies
            against four aggregators. One attack pushed RMSE from 17.95
            to 84.03. One defense restored it to 19.80.
          </>
        }
      >
        One bad airline.{" "}
        <em className="text-accent not-italic">Catastrophic damage.</em>
      </StoryHero>

      {/* ANCHOR NUMBERS ------------------------------------------------- */}
      <div className="max-w-3xl mx-auto mt-12 grid grid-cols-1 sm:grid-cols-3 gap-4">
        <AnchorStat
          tone="bad"
          value="4.7×"
          label="RMSE degradation under boosted-Byzantine attack"
          sub="17.95 → 84.03 cycles. Vanilla FedAvg has no defense at all."
        />
        <AnchorStat
          tone="good"
          value="1.85"
          label="RMSE gap from clean baseline once Krum kicks in"
          sub="From RMSE 84 catastrophe to RMSE 19.8 recovery."
        />
        <AnchorStat
          value="11"
          label="Cells in the attack × defense matrix"
          sub="3 baselines + 2 undefended attacks + 6 defended attacks."
        />
      </div>

      {/* THE THREAT ----------------------------------------------------- */}
      <StorySection title="The threat model">
        <p>
          The brief frames the attacker as <em>"a malicious airline operator
          could deliberately send corrupted weight updates to the server,
          pushing the global model to predict healthier-than-real RUL for a
          competitor's engine type"</em>. We instantiate this concretely on
          our 4-client P6 partition: <strong>client_3</strong> operates
          FD003 engines and wants the global model to under-predict failures
          on FD001 engines (their competitor's fleet).
        </p>
        <p>
          The server has no way to distinguish honest updates from poisoned
          ones — it only sees weight tensors. The only signal available is
          the updates themselves. Defenses must operate on that signal alone.
        </p>
      </StorySection>

      {/* THE TWO ATTACKS ------------------------------------------------ */}
      <StorySection title="Two attacks">
        <p>
          Both attacks use the same data the honest FD003 clients see. The
          malice lives in <em>what the attacker does with that data during
          local training</em> and <em>what update they send back</em>.
        </p>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
          <HypothesisCard
            label="A1 — Label flip"
            formula="RUL → 125 − RUL,   fault re-derived"
            text="Locally invert RUL labels, then train honestly on the lie. The malicious update pulls the global model in a wrong direction but its magnitude looks normal — the server can't spot it."
          />
          <HypothesisCard
            label="A2 — Gradient scaling"
            formula="W_send = W_global + (−10) · (W_local − W_global)"
            text="Train honestly. Before sending, compute the delta and multiply by −10. The server receives an update 10× the magnitude of an honest one, pointing in the opposite direction."
          />
        </div>
      </StorySection>

      {/* THE THREE DEFENSES --------------------------------------------- */}
      <StorySection title="Three Byzantine-robust aggregators">
        <p>
          All three are drop-in replacements for vanilla FedAvg. They
          operate on the same per-round client updates the server already
          receives — no extra communication, no extra protocol changes.
        </p>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3">
          <HypothesisCard
            label="D1 — Trimmed Mean (β=0.25)"
            formula="drop lowest 1 + highest 1, avg rest"
            text="Yin et al. ICML 2018. Per parameter, sort across clients, drop the extremes, average the middle. Tolerates ⌊β·n⌋ Byzantines per element."
          />
          <HypothesisCard
            label="D2 — Coordinate-wise Median"
            formula="per-element median across clients"
            text="Yin et al. 2018. The harshest aggregator possible — discards information from every honest client at every parameter, but tolerates ⌊n/2⌋ Byzantines."
          />
          <HypothesisCard
            label="D3 — Krum (f=1)"
            formula="pick client closest to its n−f−2 neighbors"
            text="Blanchard et al. NeurIPS 2017. Compute pairwise distances between client updates; pick the single client most geometrically central. Its update becomes the new global model verbatim."
          />
        </div>
      </StorySection>

      {/* RESULTS TABLE -------------------------------------------------- */}
      <StorySection title="The 11-cell matrix">
        <p>
          Same partition / seed / 50 rounds as P6 + every other RQ2 follow-
          up. The only thing that changes between cells is the attacker
          configuration and the aggregator:
        </p>
        <Rq7ComparisonTable />
        <p className="mt-4 text-text-dim text-sm">
          Color: <span className="text-text-muted">gray</span> = baseline
          (no attack); <span className="text-bad">red</span> = attack
          against vanilla aggregator (no defense);{" "}
          <span className="text-good">green</span> = attack + defense.
        </p>
      </StorySection>

      {/* SMOKING GUN FIGURE I — the headline bar chart ----------------- */}
      <SmokingGunFigure
        eyebrow="The smoking-gun figure (I)"
        title="Gradient-scaling is catastrophic; Krum restores order"
        artifactPath={
          phase.artifacts?.["headline_comparison_png"] ??
          "results/rq7_poisoning/headline_comparison_fd001+fd003.png"
        }
        alt="Bar chart of best-round RMSE across 11 cells. Vanilla under gradient-scale attack reaches 84.03; Krum keeps RMSE around 19.80 under both attacks."
        caption={
          <>
            The leftmost three gray bars are clean baselines (no attacker).
            The two red bars are attacks against vanilla FedAvg —{" "}
            <span className="font-mono-num text-text">29.9</span> for label
            flip,{" "}
            <strong className="text-bad">
              <span className="font-mono-num">84.0</span> for gradient
              scaling
            </strong>{" "}
            (a 4.7× degradation from baseline). The six green bars are
            attacks against defenses. The two leftmost green bars (Krum
            cells) sit at <span className="font-mono-num text-text">19.8</span>
            — within 1.85 RMSE of the clean baseline. The dashed line is
            centralized P6 (13.77) for reference.
          </>
        }
      />

      {/* SMOKING GUN FIGURE II — the delta-norm diagnostic ----------------- */}
      <SmokingGunFigure
        eyebrow="The smoking-gun figure (II)"
        title="The attacker isn't hiding — their update is 10× larger every round"
        artifactPath={
          phase.artifacts?.["attack_diagnostic_png"] ??
          "results/rq7_poisoning/attack_diagnostic_delta_norms_fd001+fd003.png"
        }
        alt="Log-scale line plot of per-client weight update L2 norms across 50 rounds. Client_3's red line sits an order of magnitude above the three honest gray lines for the entire training."
        caption={
          <>
            Per-client L2 norm of{" "}
            <span className="font-mono-num text-text">
              ||W_client − W_global||
            </span>{" "}
            on log scale during the gradient-scaling attack. Honest clients
            (gray) hover between{" "}
            <span className="font-mono-num text-text">0.1</span> and{" "}
            <span className="font-mono-num text-text">10</span>. The attacker
            (red, <span className="font-mono-num">client_3</span>) is at{" "}
            <span className="font-mono-num text-bad">60</span> in round 1
            and stays above <span className="font-mono-num text-bad">100</span>{" "}
            for 20 rounds — exactly as predicted by the scale=−10 multiplier.{" "}
            <strong className="text-text">
              The signal an outlier-detector needs is staring the server
              in the face.
            </strong>{" "}
            That's why Krum's geometric "pick the most-typical client"
            check works — there's nothing typical about a client moving
            10× faster than everyone else.
          </>
        }
      />

      {/* DEFENSE RECOVERY ---------------------------------------------- */}
      <StorySection title="Defense recovery — paired comparison">
        <p>
          For each (attack, defense) combination, the chart compares
          undefended vs defended RMSE side-by-side:
        </p>
      </StorySection>

      <SmokingGunFigure
        eyebrow="The recovery picture"
        title="All three defenses help; Krum recovers fully"
        artifactPath={
          phase.artifacts?.["defense_recovery_png"] ??
          "results/rq7_poisoning/defense_recovery_fd001+fd003.png"
        }
        alt="Paired bars: red 'undefended' and green 'defended' for each attack-defense combination. Krum's green bars are the lowest."
        caption={
          <>
            Red bars are the undefended attack's RMSE; green bars are the
            same attack with a defense. The Krum cells (rightmost in each
            attack group) recover to{" "}
            <span className="font-mono-num text-good">19.80</span> — within
            1.85 RMSE of the clean baseline. Trimmed mean and median provide
            only partial recovery against the label-flip attack (23.5 RMSE)
            but acceptable recovery against gradient scaling (21.5 RMSE).
          </>
        }
      />

      {/* THE FOUR FINDINGS ---------------------------------------------- */}
      <StorySection title="Four findings worth keeping">
        <ol className="mt-4 space-y-3 list-decimal list-inside">
          <Bullet>
            <strong>Krum dominates both attacks identically.</strong> RMSE
            19.80 / F1 0.779 against both label flip AND gradient scaling.
            Once client_3 is geometrically isolated, it never contributes
            again — defense is essentially perfect.
          </Bullet>
          <Bullet>
            <strong>Gradient scaling is catastrophic when undefended.</strong>
            {" "}RMSE explodes to{" "}
            <span className="font-mono-num text-bad">84.03</span>, F1 collapses
            to <span className="font-mono-num text-bad">0.000</span>. The
            model predicts a near-constant value across all engines and never
            flags any fault. A real maintenance pipeline running this would
            ground zero aircraft when failures are imminent.
          </Bullet>
          <Bullet>
            <strong>The attacker isn't subtle.</strong> The delta-norm
            diagnostic plot shows client_3's update magnitude sits an order
            of magnitude above honest clients for the entire training. Any
            outlier-detection layer that looks at update norms would catch
            this trivially.
          </Bullet>
          <Bullet>
            <strong>Trimmed mean = median when n=4.</strong> Both
            aggregators produced bit-identical numbers on D11/D12 and
            D21/D22 cells. For n=4 with β=0.25, trimmed mean averages
            the middle 2 of 4 sorted values per parameter, which equals
            <span className="font-mono-num text-text"> (sorted[1] + sorted[2])/2</span>{" "}
            — exactly the median formula for even n. They only diverge for
            n ≥ 5. A real airline consortium with 6+ members would see
            different behaviors from each.
          </Bullet>
        </ol>
      </StorySection>

      {/* WHY THIS EXTENDS LANDAU ---------------------------------------- */}
      <StorySection title="Extending Landau et al. (2026)">
        <p>
          Reference [10] in the project brief (Landau et al., 2026) was the
          closest existing work — they introduced robust aggregation for the
          PHM federated-learning setting, but their threat model was{" "}
          <em>accidental</em> noise (sensors malfunctioning, accidentally
          bad updates):
        </p>
        <blockquote className="border-l-2 border-accent/70 pl-4 italic text-text-dim my-4">
          "local sensor data is often corrupted or extremely noisy, which
          can poison the global federated model … the robust aggregation
          methods successfully immunized the global model against noisy
          client updates."
        </blockquote>
        <p>
          That's the right idea but the wrong threat model. Noise-robust
          aggregators are tuned against well-behaved randomness, not against
          an attacker who knows you're filtering and can craft an update to
          slip past. Our RQ7 results show the gap: vanilla FedAvg fails
          spectacularly under{" "}
          <strong>deliberate</strong> Byzantine attacks, and the
          per-parameter robust aggregators Landau favors only partially
          recover. The geometric whole-update check (Krum) is the
          intervention that actually works against adversaries.
        </p>
        <FormulaBlock>
          {"score(i) = Σ over n−f−2 closest neighbors of ||Wᵢ − Wⱼ||²"}
        </FormulaBlock>
        <p>
          The headline framing for the writeup: <em>"Landau et al. 2026
          introduced robust aggregation to defend against accidentally
          corrupted updates. We extend their framework to deliberately
          crafted poisoning attacks — and show that an attacker-aware
          aggregator (Krum) is required."</em>
        </p>
      </StorySection>

      {/* OPERATIONAL TAKEAWAY ------------------------------------------- */}
      <StorySection title="What this means for a real airline consortium">
        <p>
          If the project ever ships into a real multi-airline deployment,
          the default vanilla FedAvg server is{" "}
          <strong>operationally unsafe</strong>. A single malicious member
          can drive the global model to predict constant healthy across all
          aircraft. The asymmetry of consequences here is severe — under-
          predicted failures get engines flown past safe limits.
        </p>
        <p>
          The fix is one line of code: swap{" "}
          <span className="font-mono-num text-text">fedavg_aggregate</span>{" "}
          for{" "}
          <span className="font-mono-num text-accent">
            make_krum_aggregator(num_byzantine=1)
          </span>
          . The cost is{" "}
          <span className="font-mono-num text-text">0.76</span> RMSE
          regression in the clean case (18.71 vs 17.95) — well worth the
          1.85-RMSE worst-case ceiling under attack vs the alternative of
          84-RMSE collapse.
        </p>
        <p className="mt-4">
          Full details:{" "}
          <a href="/results" className="text-accent">
            → Results / rq7_poisoning
          </a>{" "}
          shows the per-round trajectories, per-cell metrics, and
          per-subset breakdowns.
        </p>
      </StorySection>

      {/* FUTURE WORK ---------------------------------------------------- */}
      <StorySection title="Future directions">
        <ul className="mt-4 space-y-3">
          <Bullet>
            <strong>Backdoor attacks</strong> — the attacker injects a
            trigger pattern into specific (cycle, sensor) cells and labels
            those triggered windows as max-healthy. The model learns two
            functions: correct predictions on clean inputs (passes standard
            tests) and "always healthy" on triggered inputs. The attacker
            then injects triggered FD001 sensor readings at deployment time.
            Krum may not defend against this if multiple compromised
            clients coordinate.
          </Bullet>
          <Bullet>
            <strong>Coordinated attackers</strong> — Krum's f=1 setting
            requires only 1 Byzantine. With 6+ clients and 2 coordinated
            attackers, Krum's geometric isolation argument breaks down
            because the attackers can be geometrically close to each other.
            FoolsGold and similar coordination-aware defenses are the next
            step.
          </Bullet>
          <Bullet>
            <strong>RQ6 connection</strong> — a poisoning-defense scheme
            that requires inspecting client updates (like Krum) leaks
            information about each client's training data via the update
            structure. Quantifying this leakage is RQ6 territory and
            connects the two security questions naturally.
          </Bullet>
          <Bullet>
            <strong>Poisoning under FedRep</strong> — if the heads stay
            local (FedRep architecture), the attacker can only poison the
            encoder. Each honest client's personal head still maps
            degradation features correctly. The blast radius should be
            much smaller. Worth running as a 12th cell in a follow-up.
          </Bullet>
        </ul>
      </StorySection>
    </article>
  );
}

// ===========================================================================
// 11-cell comparison table
// ===========================================================================
function Rq7ComparisonTable() {
  type Row = {
    label: string;
    rmse: string;
    f1: string;
    group: "baseline" | "attack" | "defense";
    highlight?: boolean;
  };
  const rows: Row[] = [
    { label: "B0  clean + vanilla FedAvg (control)", rmse: "17.95", f1: "0.871", group: "baseline" },
    { label: "B1  clean + trimmed mean", rmse: "17.56", f1: "0.871", group: "baseline" },
    { label: "B2  clean + Krum (f=1)", rmse: "18.71", f1: "0.773", group: "baseline" },
    { label: "AV1  label-flip + vanilla", rmse: "29.92", f1: "0.467", group: "attack" },
    { label: "AV2  grad ×-10 + vanilla", rmse: "84.03", f1: "0.000", group: "attack" },
    { label: "D11  label-flip + trimmed mean", rmse: "23.54", f1: "0.594", group: "defense" },
    { label: "D12  label-flip + median", rmse: "23.54", f1: "0.594", group: "defense" },
    { label: "D13  label-flip + Krum (f=1)", rmse: "19.80", f1: "0.779", group: "defense", highlight: true },
    { label: "D21  grad ×-10 + trimmed mean", rmse: "21.51", f1: "0.657", group: "defense" },
    { label: "D22  grad ×-10 + median", rmse: "21.51", f1: "0.657", group: "defense" },
    { label: "D23  grad ×-10 + Krum (f=1)", rmse: "19.80", f1: "0.779", group: "defense", highlight: true },
  ];

  const rowClass = (r: Row) => {
    if (r.highlight) return "bg-good/5";
    switch (r.group) {
      case "baseline":
        return "text-text-dim";
      case "attack":
        return "bg-bad/5 text-bad";
      case "defense":
        return "text-text";
    }
  };

  return (
    <div className="overflow-x-auto rounded-md border border-border bg-bg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-bg-subtle/40 text-text-dim">
            <th className="text-left px-4 py-2 font-medium">Cell</th>
            <th className="text-right px-4 py-2 font-medium">RMSE</th>
            <th className="text-right px-4 py-2 font-medium">F1</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((r) => (
            <tr key={r.label} className={rowClass(r)}>
              <td className="px-4 py-2">{r.label}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.rmse}</td>
              <td className="px-4 py-2 text-right font-mono-num">{r.f1}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
