import { useEffect } from "react";
import { createPortal } from "react-dom";

function TrashIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
    </svg>
  );
}

function WarningIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.75}
        d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  );
}

/**
 * In-app delete confirmation (replaces window.confirm so production builds never show "localhost says").
 */
export default function DeleteReportDialog({ open, onCancel, onConfirm, isDeleting, fileHint }) {
  useEffect(() => {
    if (!open) return undefined;
    function onKey(e) {
      if (e.key === "Escape") onCancel();
    }
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onCancel]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 sm:p-6" role="presentation">
      <button
        type="button"
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-[3px] cursor-default border-0"
        aria-label="Close dialog"
        onClick={onCancel}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-report-dialog-title"
        className="relative w-full max-w-[420px] overflow-hidden rounded-2xl border border-slate-200/90 bg-white shadow-2xl shadow-slate-900/20 ring-1 ring-slate-100 animate-fade-up"
      >
        <div className="border-b border-slate-100 bg-gradient-to-br from-slate-50/90 to-white px-6 pt-6 pb-4">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-red-50 text-red-600 ring-1 ring-red-100">
              <TrashIcon className="h-5 w-5" />
            </div>
            <div className="min-w-0 pt-0.5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-clinic-600 mb-1">MedAI</p>
              <h2 id="delete-report-dialog-title" className="text-lg font-semibold text-slate-900 leading-snug">
                Delete this report?
              </h2>
            </div>
          </div>
        </div>

        <div className="px-6 py-5 space-y-4">
          <p className="text-sm text-slate-600 leading-relaxed">
            This removes the study record and the uploaded image from your workspace. You cannot undo this action.
          </p>

          {fileHint ? (
            <div className="rounded-xl border border-slate-200/90 bg-slate-50/80 px-3.5 py-2.5">
              <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500 mb-1">File</p>
              <p className="text-xs font-mono text-slate-700 truncate" title={fileHint}>
                {fileHint}
              </p>
            </div>
          ) : null}

          <div className="flex gap-3 rounded-xl border border-amber-200/80 bg-amber-50/80 px-3.5 py-3 text-amber-950">
            <WarningIcon className="h-5 w-5 shrink-0 text-amber-700 mt-0.5" />
            <p className="text-xs leading-relaxed">
              If you might need this image later, download or save a copy outside the app before deleting.
            </p>
          </div>
        </div>

        <div className="flex flex-col-reverse gap-2 border-t border-slate-100 bg-slate-50/50 px-6 py-4 sm:flex-row sm:justify-end sm:gap-3">
          <button
            type="button"
            className="btn-ghost px-4 py-2.5 text-sm font-semibold rounded-xl"
            onClick={onCancel}
            disabled={isDeleting}
          >
            Keep report
          </button>
          <button
            type="button"
            className="btn-danger inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-semibold min-w-[8rem] rounded-xl"
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting ? (
              <>
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" aria-hidden>
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path
                    className="opacity-90"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Deleting…
              </>
            ) : (
              <>
                <TrashIcon className="h-4 w-4" />
                Delete permanently
              </>
            )}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
