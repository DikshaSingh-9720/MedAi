import { Link } from "react-router-dom";

export default function Logo({ to = "/", size = "md", subtitle = true, variant = "light" }) {
  const sizes = {
    sm: { box: "w-10 h-10 text-sm", title: "text-[1.0625rem]", sub: "text-[11px]" },
    md: { box: "w-12 h-12 text-base", title: "text-lg", sub: "text-xs" },
    lg: { box: "w-14 h-14 text-lg", title: "text-xl", sub: "text-sm" },
  };
  const s = sizes[size] || sizes.md;
  const isLight = variant === "light";

  return (
    <Link
      to={to}
      className={`inline-flex items-center gap-3 group outline-none rounded-xl focus-visible:ring-2 focus-visible:ring-clinic-500/40 focus-visible:ring-offset-2 ${
        isLight ? "focus-visible:ring-offset-white" : "focus-visible:ring-offset-slate-900"
      }`}
    >
      <span
        className={`${s.box} flex items-center justify-center rounded-xl bg-gradient-to-br from-clinic-500 via-clinic-600 to-clinic-700 text-white font-bold shadow-md ring-1 ring-white/25 transition-transform duration-200 group-hover:scale-[1.02]`}
      >
        M
      </span>
      <div className="text-left">
        <p
          className={`font-sans ${s.title} font-semibold leading-tight tracking-tight ${
            isLight ? "text-slate-900" : "text-white"
          }`}
        >
          MedAI
        </p>
        {subtitle && (
          <p className={`${s.sub} font-medium ${isLight ? "text-slate-500" : "text-slate-400"}`}>
            Imaging intelligence
          </p>
        )}
      </div>
    </Link>
  );
}
