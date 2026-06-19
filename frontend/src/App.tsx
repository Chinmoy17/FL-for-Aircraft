import { LiveDemo } from "./components/LiveDemo";

export function App() {
  return (
    <div className="app">
      <header className="site-header">
        <h1>FL-Aircraft PHM — live demo</h1>
        <p className="subtitle">
          Federated Learning for Aircraft Engine Prognostics on the NASA
          C-MAPSS dataset. Pick a trained checkpoint and a test engine; the
          backend runs Integrated Gradients on demand and returns a sensor-
          level explanation grounded in a maintenance ontology.
        </p>
      </header>
      <LiveDemo />
    </div>
  );
}
