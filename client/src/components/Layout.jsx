import { Outlet } from "react-router-dom";
import Logo from "./Logo.jsx";

export default function Layout() {
  return (
    <div className="relative z-10 min-h-dvh flex flex-col">
      <header className="sticky top-0 z-50 border-b border-slate-200/90 bg-white/90 backdrop-blur-xl supports-[backdrop-filter]:bg-white/80">
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-clinic-500/20 to-transparent pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-3.5 flex items-center justify-between gap-3">
          <Logo to="/" size="sm" variant="light" />
        </div>
      </header>

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-9 pb-14 sm:pb-12">
        <Outlet />
      </main>

      <footer className="border-t border-slate-200/90 mt-auto bg-white/60 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-6 text-xs text-slate-600">
          <p className="text-center sm:text-left max-w-lg leading-relaxed text-slate-600">
            MedAI provides automated screening assistance only — not a diagnosis. Always consult a licensed
            clinician for medical decisions.
          </p>
          <div className="flex items-center gap-6 text-slate-500">
            <span className="whitespace-nowrap">© {new Date().getFullYear()} MedAI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
