import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { BenchmarkApp } from "./BenchmarkApp";

const CurrentApp = window.location.pathname.startsWith("/benchmark") ? BenchmarkApp : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CurrentApp />
  </StrictMode>,
);
