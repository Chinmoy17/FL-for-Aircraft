import { NavLink, Route, BrowserRouter, Routes } from "react-router-dom";
import { LiveDemoPage } from "./pages/LiveDemoPage";
import { ResultsPage } from "./pages/ResultsPage";
import { Rq2StoryPage } from "./pages/Rq2StoryPage";

export function App() {
  return (
    <BrowserRouter>
      <div className="min-h-full flex flex-col">
        <TopNav />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<LiveDemoPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/rq2-story" element={<Rq2StoryPage />} />
            <Route
              path="*"
              element={
                <div className="max-w-6xl mx-auto px-6 py-12 text-text-dim">
                  Not found.{" "}
                  <NavLink to="/" className="text-accent">
                    Back to live demo
                  </NavLink>
                </div>
              }
            />
          </Routes>
        </main>
        <SiteFooter />
      </div>
    </BrowserRouter>
  );
}

function TopNav() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    [
      "px-3 py-2 text-sm rounded-md transition-colors",
      isActive
        ? "bg-bg-subtle text-text font-medium"
        : "text-text-dim hover:text-text hover:bg-bg-subtle/60",
    ].join(" ");

  return (
    <header className="border-b border-border bg-bg/95 backdrop-blur sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        <NavLink
          to="/"
          end
          className="flex items-baseline gap-2 hover:no-underline"
        >
          <span className="text-text font-semibold tracking-tight">
            FL-Aircraft PHM
          </span>
          <span className="text-text-muted text-xs hidden sm:inline">
            federated learning · NASA C-MAPSS
          </span>
        </NavLink>
        <nav className="flex items-center gap-1" aria-label="Primary">
          <NavLink to="/" end className={linkClass}>
            Live demo
          </NavLink>
          <NavLink to="/results" className={linkClass}>
            Results
          </NavLink>
          <NavLink to="/rq2-story" className={linkClass}>
            RQ2 story
          </NavLink>
        </nav>
      </div>
    </header>
  );
}

function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border">
      <div className="max-w-6xl mx-auto px-6 py-6 text-xs text-text-muted">
        Research project · Federated Learning for Aircraft Engine PHM ·{" "}
        <a
          href="https://github.com/Chinmoy17/FL-for-Aircraft"
          target="_blank"
          rel="noreferrer"
        >
          github.com/Chinmoy17/FL-for-Aircraft
        </a>
      </div>
    </footer>
  );
}
