export default function MedicalNotice({ className = "" }) {
  return (
    <div
      role="note"
      className={`rounded-xl border border-amber-200 bg-amber-50 px-4 py-3.5 ${className}`}
    >
      <p className="text-sm text-amber-950 leading-relaxed">
        <span className="font-semibold text-amber-800">Important:</span> MedAI provides decision support only. It
        is <span className="font-medium text-amber-900">not a medical device</span> and does not replace
        professional judgment or emergency care.
      </p>
    </div>
  );
}
