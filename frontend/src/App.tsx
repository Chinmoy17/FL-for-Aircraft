import { NavLink, Route, BrowserRouter, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AbstractPage } from "./pages/AbstractPage";
import { CoverPage } from "./pages/CoverPage";
import { ExperimentCentralizedPage } from "./pages/ExperimentCentralizedPage";
import { ExperimentDataPage } from "./pages/ExperimentDataPage";
import { ExperimentEdaPage } from "./pages/ExperimentEdaPage";
import { ExperimentFedavgPage } from "./pages/ExperimentFedavgPage";
import { ExperimentLocalOnlyPage } from "./pages/ExperimentLocalOnlyPage";
import { ExperimentNonIidPage } from "./pages/ExperimentNonIidPage";
import { ExperimentSmokePage } from "./pages/ExperimentSmokePage";
import { LiveDemoPage } from "./pages/LiveDemoPage";
import { ReportsPage } from "./pages/ReportsPage";
import { ResultsPage } from "./pages/ResultsPage";
import { Rq2StoryPage } from "./pages/Rq2StoryPage";
import { Rq3StoryPage } from "./pages/Rq3StoryPage";
import { Rq45SynthesisPage } from "./pages/Rq45SynthesisPage";
import { Rq7StoryPage } from "./pages/Rq7StoryPage";

/**
 * Top-level router.
 *
 * The single layout route mounts `AppShell` (sticky left sidebar +
 * full-width content area) and renders the active page through its
 * <Outlet />. The old top nav + site footer are gone — the sidebar
 * is the table of contents, the author byline, and the GitHub link
 * all in one.
 */
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<CoverPage />} />
          <Route path="/abstract" element={<AbstractPage />} />
          <Route path="/experiments/00-eda" element={<ExperimentEdaPage />} />
          <Route path="/experiments/01-data" element={<ExperimentDataPage />} />
          <Route path="/experiments/02-smoke" element={<ExperimentSmokePage />} />
          <Route path="/experiments/03-centralized" element={<ExperimentCentralizedPage />} />
          <Route path="/experiments/04-local-only" element={<ExperimentLocalOnlyPage />} />
          <Route path="/experiments/05-fedavg" element={<ExperimentFedavgPage />} />
          <Route path="/experiments/06-non-iid" element={<ExperimentNonIidPage />} />
          <Route path="/demo" element={<LiveDemoPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/rq2-story" element={<Rq2StoryPage />} />
          <Route path="/rq3-story" element={<Rq3StoryPage />} />
          <Route path="/rq7-story" element={<Rq7StoryPage />} />
          <Route path="/rq4-rq5-synthesis" element={<Rq45SynthesisPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

function NotFoundPage() {
  return (
    <div className="px-10 py-16 text-text-dim">
      <h1 className="font-display text-3xl text-text mb-2">Not found</h1>
      <p className="mb-4">That page doesn't exist (yet).</p>
      <NavLink to="/" className="text-accent">
        ← Back to the cover
      </NavLink>
    </div>
  );
}
