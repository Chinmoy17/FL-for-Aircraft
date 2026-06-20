import { NavLink, Outlet } from "react-router-dom";

/**
 * AppShell — the global page chrome for the academic / paper-style frontend.
 *
 * Structure:
 *
 *   ┌─────────────┬─────────────────────────────────────────┐
 *   │             │                                         │
 *   │  Sidebar    │   <Outlet />   (full remaining width)   │
 *   │  (sticky    │                                         │
 *   │  TOC,       │                                         │
 *   │  256px)     │                                         │
 *   │             │                                         │
 *   └─────────────┴─────────────────────────────────────────┘
 *
 * The sidebar is the project's table of contents. It mirrors the
 * narrative arc of the work (overview → experiments → research
 * questions → follow-ups → artifacts) rather than a flat link list.
 *
 * Routes that don't exist yet are rendered as "soon" placeholders so
 * the planned IA is visible from day one — readers can see what's
 * coming next without each link breaking on click.
 *
 * Below the md breakpoint the sidebar collapses out of view; the
 * content area takes full width. A mobile menu can be added later
 * if anyone actually views this on a phone (academic audience is
 * desktop-first).
 */
export function AppShell() {
  return (
    <div className="min-h-screen flex bg-bg text-text">
      <Sidebar />
      <main className="flex-1 min-w-0">
        <Outlet />
      </main>
    </div>
  );
}

// ===========================================================================
// Sidebar data model
// ===========================================================================
type BadgeKind = "positive" | "negative" | "null";

type NavItem =
  | {
      kind: "internal";
      label: string;
      to: string;
      end?: boolean;
      badge?: BadgeKind;
    }
  | {
      kind: "external";
      label: string;
      href: string;
    }
  | {
      kind: "soon";
      label: string;
    };

type NavSection = { eyebrow: string; items: NavItem[] };

/**
 * Navigation grouped by narrative arc. Items mark the project's reading
 * order: start with overview, walk through the empirical phases, then
 * the research questions, then follow-ups, then artifacts.
 *
 * The "soon" items are intentional — they signal where the work is
 * heading without breaking the click. As each page lands they get
 * promoted to "internal" with the real `to` path.
 */
const SECTIONS: NavSection[] = [
  {
    eyebrow: "Overview",
    items: [
      { kind: "internal", label: "Cover", to: "/", end: true },
      { kind: "internal", label: "Abstract & contributions", to: "/abstract" },
    ],
  },
  {
    eyebrow: "Experiments",
    items: [
      { kind: "internal", label: "00 · EDA", to: "/experiments/00-eda" },
      { kind: "internal", label: "01 · Data pipeline", to: "/experiments/01-data" },
      { kind: "internal", label: "02 · Smoke run", to: "/experiments/02-smoke" },
      { kind: "internal", label: "03 · Centralized", to: "/experiments/03-centralized" },
      { kind: "internal", label: "04 · Local-only", to: "/experiments/04-local-only" },
      { kind: "internal", label: "05 · FedAvg IID", to: "/experiments/05-fedavg" },
      { kind: "internal", label: "06 · Non-IID baseline", to: "/experiments/06-non-iid" },
    ],
  },
  {
    eyebrow: "Research questions",
    items: [
      {
        kind: "internal",
        label: "RQ2 — Aggregation",
        to: "/rq2-story",
        badge: "negative",
      },
      {
        kind: "internal",
        label: "RQ3 — Interpretability",
        to: "/rq3-story",
        badge: "positive",
      },
      {
        kind: "internal",
        label: "RQ7 — Security",
        to: "/rq7-story",
        badge: "positive",
      },
      { kind: "soon", label: "RQ4 / RQ5 — Synthesis" },
    ],
  },
  {
    eyebrow: "RQ2 follow-ups",
    items: [
      { kind: "soon", label: "FedProx" },
      { kind: "soon", label: "FedRep" },
      { kind: "soon", label: "FedCCFA" },
    ],
  },
  {
    eyebrow: "Artifacts",
    items: [
      { kind: "internal", label: "Live demo", to: "/demo" },
      { kind: "internal", label: "Results browser", to: "/results" },
      { kind: "soon", label: "Technical reports" },
      {
        kind: "external",
        label: "GitHub →",
        href: "https://github.com/Chinmoy17/FL-for-Aircraft",
      },
    ],
  },
];

// ===========================================================================
// Sidebar
// ===========================================================================
function Sidebar() {
  return (
    <aside
      className="
        hidden md:flex flex-col
        w-64 shrink-0
        sticky top-0 h-screen
        border-r border-border
        bg-bg-subtle
        overflow-y-auto
      "
      aria-label="Project navigation"
    >
      <SidebarHeader />
      <nav className="flex-1 px-3 py-4 space-y-6">
        {SECTIONS.map((section) => (
          <SidebarSection key={section.eyebrow} section={section} />
        ))}
      </nav>
      <SidebarFooter />
    </aside>
  );
}

function SidebarHeader() {
  return (
    <div className="px-6 pt-6 pb-5 border-b border-border">
      <div className="font-display text-[22px] leading-[1.15] text-text">
        FL for Aircraft PHM
      </div>
      <div className="mt-2 text-[11px] text-text-muted uppercase tracking-[0.14em]">
        Chinmoy Mitra · 2026
      </div>
    </div>
  );
}

function SidebarFooter() {
  return (
    <div className="px-6 py-4 border-t border-border text-[10.5px] leading-snug text-text-muted">
      PhD research project · NASA C-MAPSS turbofan dataset
    </div>
  );
}

function SidebarSection({ section }: { section: NavSection }) {
  return (
    <div>
      <h3 className="px-3 mb-1.5 eyebrow">{section.eyebrow}</h3>
      <ul>
        {section.items.map((item, i) => (
          <li key={`${section.eyebrow}-${i}`}>
            <NavItemView item={item} />
          </li>
        ))}
      </ul>
    </div>
  );
}

// ===========================================================================
// Nav item renderers
// ===========================================================================
function NavItemView({ item }: { item: NavItem }) {
  if (item.kind === "external") {
    return (
      <a
        href={item.href}
        target="_blank"
        rel="noreferrer"
        className="
          block px-3 py-1.5 text-sm rounded
          text-text-dim hover:text-text hover:bg-bg
          transition-colors
        "
      >
        {item.label}
      </a>
    );
  }

  if (item.kind === "soon") {
    return (
      <div
        className="
          px-3 py-1.5 text-sm
          flex items-center justify-between
          text-text-muted/70 cursor-default
        "
        aria-disabled="true"
      >
        <span>{item.label}</span>
        <span className="text-[9px] uppercase tracking-[0.1em] text-text-muted/60">
          soon
        </span>
      </div>
    );
  }

  return (
    <NavLink
      to={item.to}
      end={item.end}
      className={({ isActive }) =>
        [
          "px-3 py-1.5 text-sm rounded",
          "flex items-center justify-between",
          "transition-colors",
          isActive
            ? "bg-accent-subtle text-accent font-medium"
            : "text-text-dim hover:text-text hover:bg-bg",
        ].join(" ")
      }
    >
      <span>{item.label}</span>
      {item.badge && <BadgeView kind={item.badge} />}
    </NavLink>
  );
}

function BadgeView({ kind }: { kind: BadgeKind }) {
  // Small symbolic badge so a glance at the sidebar gives the verdict
  // of each research question (positive / negative / null finding).
  const styles: Record<BadgeKind, { cls: string; symbol: string; aria: string }> = {
    positive: {
      cls: "text-good bg-good/10 border-good/20",
      symbol: "+",
      aria: "positive finding",
    },
    negative: {
      cls: "text-bad bg-bad/10 border-bad/20",
      symbol: "−",
      aria: "negative finding",
    },
    null: {
      cls: "text-text-muted bg-text-muted/10 border-text-muted/20",
      symbol: "○",
      aria: "null result",
    },
  };
  const s = styles[kind];
  return (
    <span
      className={`
        inline-flex items-center justify-center
        w-4 h-4 rounded-full border text-[10px] leading-none font-semibold
        ${s.cls}
      `}
      aria-label={s.aria}
    >
      {s.symbol}
    </span>
  );
}
