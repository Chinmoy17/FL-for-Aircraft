import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

/**
 * AppShell — the global page chrome for the academic / paper-style frontend.
 *
 * Structure on desktop (≥ md):
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
 * On mobile (< md) the sidebar collapses into a slide-in drawer behind a
 * hamburger button rendered in a thin top bar. Active sidebar items
 * auto-scroll into the visible portion of the TOC so the reader's
 * position is always discoverable.
 *
 * Routes that don't exist yet are rendered as "soon" placeholders so
 * the planned IA is visible from day one — readers can see what's
 * coming next without each link breaking on click.
 */
export function AppShell() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();

  // Close drawer on route change (mobile UX).
  useEffect(() => {
    setDrawerOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex bg-bg text-text">
      <RouteTitleManager />

      {/* Desktop sidebar (md+) */}
      <Sidebar className="hidden md:flex" />

      {/* Mobile top bar (< md) */}
      <MobileTopBar onOpenDrawer={() => setDrawerOpen(true)} />

      {/* Mobile drawer (< md) */}
      <MobileDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />

      <main className="flex-1 min-w-0 pt-12 md:pt-0">
        {/*
          Content is capped at max-w-screen-2xl (1536 px) and centered.
          This keeps line lengths comfortable on extended-display setups
          (1920 / 2560 / 3440 px screens) without making things feel
          cramped on standard laptops. The wrapper is INSIDE <main> so
          the sidebar still sits flush against the left edge.
        */}
        <div className="max-w-screen-2xl mx-auto w-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

// ===========================================================================
// Route → title map (drives document.title per page)
// ===========================================================================
const TITLES: Array<[RegExp, string]> = [
  [/^\/$/, "Cover"],
  [/^\/abstract$/, "Abstract & contributions"],
  [/^\/experiments\/00-eda$/, "00 · EDA"],
  [/^\/experiments\/01-data$/, "01 · Data pipeline"],
  [/^\/experiments\/02-smoke$/, "02 · Smoke run"],
  [/^\/experiments\/03-centralized$/, "03 · Centralized"],
  [/^\/experiments\/04-local-only$/, "04 · Local-only"],
  [/^\/experiments\/05-fedavg$/, "05 · FedAvg IID"],
  [/^\/experiments\/06-non-iid$/, "06 · Non-IID baseline"],
  [/^\/rq2-story$/, "RQ2 · Aggregation"],
  [/^\/rq3-story$/, "RQ3 · Interpretability"],
  [/^\/rq7-story$/, "RQ7 · Security"],
  [/^\/rq4-rq5-synthesis$/, "RQ4 / RQ5 · Synthesis"],
  [/^\/rq2-followups\/fedprox$/, "FedProx · RQ2 follow-up"],
  [/^\/rq2-followups\/fedrep$/, "FedRep · RQ2 follow-up"],
  [/^\/rq2-followups\/fedccfa$/, "FedCCFA · RQ2 follow-up"],
  [/^\/demo$/, "Live demo"],
  [/^\/reports$/, "Technical reports"],
];
const APP_NAME = "FL Aircraft PHM";

function RouteTitleManager() {
  const { pathname } = useLocation();
  useEffect(() => {
    const match = TITLES.find(([rx]) => rx.test(pathname));
    document.title = match ? `${match[1]} · ${APP_NAME}` : APP_NAME;
  }, [pathname]);
  return null;
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
      {
        kind: "internal",
        label: "RQ4 / RQ5 — Synthesis",
        to: "/rq4-rq5-synthesis",
      },
    ],
  },
  {
    eyebrow: "RQ2 follow-ups",
    items: [
      { kind: "internal", label: "FedProx", to: "/rq2-followups/fedprox", badge: "positive" },
      { kind: "internal", label: "FedRep", to: "/rq2-followups/fedrep", badge: "positive" },
      { kind: "internal", label: "FedCCFA", to: "/rq2-followups/fedccfa", badge: "null" },
    ],
  },
  {
    eyebrow: "Artifacts",
    items: [
      { kind: "internal", label: "Live demo", to: "/demo" },
      { kind: "internal", label: "Technical reports", to: "/reports" },
      {
        kind: "external",
        label: "GitHub →",
        href: "https://github.com/Chinmoy17/FL-for-Aircraft",
      },
    ],
  },
];

// ===========================================================================
// Sidebar (desktop) — sticky left rail
// ===========================================================================
function Sidebar({ className = "" }: { className?: string }) {
  return (
    <aside
      className={`
        ${className}
        flex-col
        w-64 shrink-0
        sticky top-0 h-screen
        border-r border-border
        bg-bg-subtle
        overflow-y-auto
      `}
      aria-label="Project navigation"
    >
      <SidebarContent />
    </aside>
  );
}

// Shared inner content used by both desktop sidebar and mobile drawer.
function SidebarContent() {
  const navRef = useRef<HTMLElement>(null);
  const { pathname } = useLocation();

  // Auto-scroll the active item into view inside the TOC. Important once
  // the Experiments section has 7+ entries and the active link might
  // start below the visible viewport.
  useEffect(() => {
    const active = navRef.current?.querySelector<HTMLElement>(
      'a[aria-current="page"]',
    );
    if (active) {
      active.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [pathname]);

  return (
    <>
      <SidebarHeader />
      <nav
        ref={navRef}
        className="flex-1 px-3 py-4 space-y-6"
      >
        {SECTIONS.map((section) => (
          <SidebarSection key={section.eyebrow} section={section} />
        ))}
      </nav>
      <SidebarFooter />
    </>
  );
}

// ===========================================================================
// Mobile top bar + drawer (< md)
// ===========================================================================
function MobileTopBar({ onOpenDrawer }: { onOpenDrawer: () => void }) {
  return (
    <div
      className="
        md:hidden fixed top-0 inset-x-0 z-30
        h-12 flex items-center gap-3 px-4
        border-b border-border bg-bg/95 backdrop-blur
      "
    >
      <button
        type="button"
        onClick={onOpenDrawer}
        aria-label="Open navigation"
        className="
          inline-flex items-center justify-center
          w-9 h-9 rounded-md
          text-text hover:bg-bg-subtle
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40
        "
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>
      <span className="font-display text-base text-text leading-none">
        FL Aircraft PHM
      </span>
    </div>
  );
}

function MobileDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  // Lock body scroll while the drawer is open. Restore on close.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="md:hidden fixed inset-0 z-40">
      {/* Backdrop */}
      <button
        type="button"
        aria-label="Close navigation"
        onClick={onClose}
        className="absolute inset-0 bg-black/30 backdrop-blur-[1px] cursor-default"
      />
      {/* Drawer panel — fixed full-height, no stickiness needed */}
      <aside
        className="
          absolute inset-y-0 left-0
          w-72 max-w-[85vw]
          flex flex-col
          bg-bg-subtle border-r border-border shadow-xl
          overflow-y-auto
        "
        aria-label="Project navigation"
      >
        <SidebarContent />
      </aside>
    </div>
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
