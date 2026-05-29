import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { BenchmarkApp } from "./BenchmarkApp";

const normalizedBasePath = import.meta.env.BASE_URL.replace(/\/+$/u, "");
const pathAfterBase =
  normalizedBasePath.length === 0 ||
  !window.location.pathname.startsWith(normalizedBasePath)
    ? window.location.pathname
    : window.location.pathname.slice(normalizedBasePath.length);
const CurrentApp = pathAfterBase.startsWith("/benchmark") ? BenchmarkApp : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CurrentApp />
  </StrictMode>,
);
