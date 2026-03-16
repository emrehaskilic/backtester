import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  Settings2,
  Database,
  Activity,
  Rocket,
  Trophy,
} from "lucide-react";
import { useEffect, useState } from "react";
import { getHealth } from "../api";

const NAV_ITEMS = [
  { path: "/", label: "Panel", icon: LayoutDashboard },
  { path: "/pipeline", label: "Pipeline", icon: Rocket },
  { path: "/backtest", label: "Backtest", icon: FlaskConical },
  { path: "/results", label: "Sonuçlar", icon: Trophy },
  { path: "/data", label: "Veri", icon: Database },
  { path: "/settings", label: "Ayarlar", icon: Settings2 },
];

const PAGE_TITLES: Record<string, string> = {
  "/": "Panel",
  "/pipeline": "Pipeline — Tam Otomatik Optimizasyon",
  "/backtest": "Backtest",
  "/results": "Optimizasyon Sonuçları",
  "/data": "Veri Yönetimi",
  "/settings": "Ayarlar",
};

export default function Layout() {
  const location = useLocation();
  const [apiConnected, setApiConnected] = useState(false);

  useEffect(() => {
    const check = () => {
      getHealth()
        .then(() => setApiConnected(true))
        .catch(() => setApiConnected(false));
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-surface-dark border-r border-border flex flex-col shrink-0">
        {/* Logo */}
        <div className="p-5 border-b border-border">
          <div className="flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-500" />
            <div>
              <h1 className="text-base font-bold text-white leading-none">
                OptiCanavar
              </h1>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Tam Otomatik Optimizer
              </p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              end={path === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                  isActive
                    ? "text-blue-400 bg-blue-500/10 border-r-2 border-blue-500"
                    : "text-slate-400 hover:text-slate-200 hover:bg-surface-hover"
                }`
              }
            >
              <Icon className="w-4 h-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* API Status */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2 text-xs">
            <div
              className={`w-2 h-2 rounded-full ${
                apiConnected ? "bg-profit" : "bg-loss"
              }`}
            />
            <span className="text-slate-500">
              API: {apiConnected ? "Bagli" : "Bagli degil"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-14 bg-surface border-b border-border flex items-center px-6 shrink-0">
          <h2 className="text-lg font-semibold text-white">
            {PAGE_TITLES[location.pathname] || ""}
          </h2>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
