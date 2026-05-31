import type { JSX } from "react";
import App from "./App";
import { BenchmarkApp } from "./BenchmarkApp";

const normalizedBasePath = import.meta.env.BASE_URL.replace(/\/+$/u, "");
const pathAfterBase =
  normalizedBasePath.length === 0 ||
  !window.location.pathname.startsWith(normalizedBasePath)
    ? window.location.pathname
    : window.location.pathname.slice(normalizedBasePath.length);

export const RouteApp = (): JSX.Element => {
  const CurrentApp = pathAfterBase.startsWith("/benchmark")
    ? BenchmarkApp
    : App;
  return <CurrentApp />;
};
