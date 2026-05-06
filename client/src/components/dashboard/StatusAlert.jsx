export default function StatusAlert({ tone = "info", message }) {
  if (!message) return null;

  const toneClass =
    tone === "error"
      ? "border-red-200 bg-red-50 text-red-800"
      : tone === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-900"
        : tone === "success"
          ? "border-emerald-200 bg-emerald-50 text-emerald-900"
          : "border-sky-200 bg-sky-50 text-sky-900";

  return (
    <div role="alert" className={`rounded-xl border text-sm px-4 py-3.5 flex items-start gap-3 ${toneClass}`}>
      <span className="mt-0.5 font-bold" aria-hidden>
        !
      </span>
      <span>{message}</span>
    </div>
  );
}
