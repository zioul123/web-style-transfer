import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { RouteApp } from "./RouteApp";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouteApp />
  </StrictMode>,
);
