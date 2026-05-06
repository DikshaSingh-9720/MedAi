import { useCallback, useEffect, useMemo, useState } from "react";
import { deleteReport, fetchReports, uploadReport } from "../api.js";
import DeleteReportDialog from "../components/DeleteReportDialog.jsx";
import SimpleFindingDisplay from "../components/dashboard/SimpleFindingDisplay.jsx";
import StatusAlert from "../components/dashboard/StatusAlert.jsx";
import { filenameModalityMismatchMessage } from "../modalityUploadGuard.js";
import { displayFileName, modalityShort } from "../utils/clinicalRules.js";
import { formatDateTime, formatShortDateTime } from "../utils/dateFormat.js";

const STUDY_OPTIONS = [
  { id: "xray", label: "X-ray", hint: "Bone fracture screening" },
  { id: "ct", label: "CT Scan", hint: "Lung CT - lung cancer vs benign / normal" },
  { id: "ultrasound", label: "Ultrasound", hint: "Breast ultrasound - breast cancer vs not" },
  { id: "mri", label: "MRI", hint: "Brain MRI slice - brain tumor vs not" },
];

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

function UploadIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
      <path d="M24 8v20m0 0l-6-6m6 6l6-6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 32v4a4 4 0 004 4h12a4 4 0 004-4v-4" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

function HistorySkeleton() {
  return (
    <ul className="space-y-3 animate-pulse">
      {[1, 2, 3].map((i) => (
        <li key={i} className="rounded-2xl border border-slate-200/90 bg-white p-4 flex gap-4 shadow-sm">
          <div className="w-full sm:w-32 h-32 rounded-xl bg-slate-200/80 shrink-0" />
          <div className="flex-1 space-y-3 pt-1">
            <div className="h-5 bg-slate-200/80 rounded-lg w-2/5" />
            <div className="h-4 bg-slate-100 rounded w-3/5" />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function DashboardProfessional() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const [latest, setLatest] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [studyModality, setStudyModality] = useState("xray");
  const [backendDown, setBackendDown] = useState(false);

  const load = useCallback(async () => {
    setError("");
    try {
      const data = await fetchReports();
      setReports(data);
      setBackendDown(false);
    } catch (e) {
      setError(e.response?.data?.error || "Could not load reports");
      setBackendDown(!e.response);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const studyDescription = useMemo(() => STUDY_OPTIONS.map((s) => s.label).join(" • "), []);

  async function handleFiles(files) {
    const file = files?.[0];
    if (!file || !file.type.startsWith("image/")) {
      setError("Please choose an image file (PNG or JPEG).");
      return;
    }
    const mismatch = filenameModalityMismatchMessage(file.name, studyModality);
    if (mismatch) {
      setError(mismatch);
      return;
    }
    setUploading(true);
    setError("");
    setLatest(null);
    try {
      const result = await uploadReport(file, studyModality);
      setLatest(result);
      setBackendDown(false);
      await load();
    } catch (e) {
      setError(e.response?.data?.error || e.message || "Upload failed");
      setBackendDown(!e.response);
    } finally {
      setUploading(false);
    }
  }

  const confirmDeleteReport = useCallback(async () => {
    const reportId = pendingDelete?.id;
    if (!reportId) return;
    setError("");
    setDeletingId(reportId);
    try {
      await deleteReport(reportId);
      setBackendDown(false);
      if (latest?.id === reportId) setLatest(null);
      await load();
    } catch (e) {
      setError(e.response?.data?.error || e.message || "Could not delete report");
      setBackendDown(!e.response);
    } finally {
      setDeletingId(null);
      setPendingDelete(null);
    }
  }, [pendingDelete?.id, latest?.id, load]);

  return (
    <div className="space-y-7 sm:space-y-9 animate-fade-up">
      {backendDown && <StatusAlert tone="warning" message="Backend appears disconnected. Start the API server to run upload and report actions." />}
      {error && <StatusAlert tone="error" message={error} />}

      <section className="relative overflow-hidden rounded-2xl border border-slate-200/90 bg-gradient-to-br from-white via-slate-50/95 to-clinic-50/35 p-6 sm:p-8 shadow-md shadow-slate-200/40 ring-1 ring-slate-100">
        <div className="relative flex flex-col xl:flex-row xl:items-end xl:justify-between gap-6 sm:gap-8">
          <div className="max-w-2xl">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-clinic-600 mb-2">Patient workspace</p>
            <h1 className="font-sans text-[1.75rem] sm:text-3xl xl:text-[2.125rem] font-semibold text-slate-900 leading-tight tracking-tight mb-3">
              Upload & analyze imaging
            </h1>
            <p className="text-slate-600 text-[15px] sm:text-base leading-relaxed max-w-xl">
              Choose a study type, upload one image, and get a quick AI-assisted screening output.
            </p>
            <p className="mt-1.5 text-slate-500 text-sm max-w-xl">Supported studies: {studyDescription}. Results are decision support only, not diagnosis.</p>
            <div className="mt-5 flex flex-wrap items-center gap-3" role="group" aria-label="Study type">
              {STUDY_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setStudyModality(opt.id)}
                  title={opt.hint}
                  aria-pressed={studyModality === opt.id}
                  className={`study-type-btn ${studyModality === opt.id ? "study-type-btn-active" : ""}`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-1 xl:w-[min(100%,216px)] gap-2.5 sm:gap-3 shrink-0">
            <div className="rounded-xl border border-slate-200/95 bg-white px-4 py-3.5 shadow-sm">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-0.5">Reports</p>
              <p className="text-2xl font-semibold text-slate-900 tabular-nums font-mono tracking-tight">{reports.length}</p>
            </div>
            <div className="rounded-xl border border-slate-200/95 bg-white px-4 py-3.5 shadow-sm">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-0.5">Pipeline</p>
              <p className={`text-sm font-semibold ${backendDown ? "text-amber-700" : "text-emerald-700"}`}>{backendDown ? "Backend disconnected" : "Ready"}</p>
            </div>
            <div className="rounded-xl border border-slate-200/95 bg-white px-4 py-3.5 shadow-sm col-span-2 sm:col-span-1 xl:col-span-1">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-0.5">Last upload</p>
              <p className="text-sm text-slate-800 truncate font-mono tabular-nums">{formatShortDateTime(reports[0]?.createdAt)}</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div
          className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 overflow-hidden bg-white/70 ${
            dragOver ? "border-clinic-400 bg-clinic-50/90" : "border-slate-300/85 hover:border-clinic-400/60 hover:bg-clinic-50/25"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          <div className="relative px-5 py-9 sm:py-10 text-center">
            <input key={studyModality} type="file" accept="image/*" className="sr-only" id="file-upload" disabled={uploading} onChange={(e) => handleFiles(e.target.files)} />
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-clinic-100 text-clinic-600 ring-1 ring-clinic-200/90 mb-4">
              <UploadIcon className="w-7 h-7" />
            </div>
            <label htmlFor="file-upload" className={`cursor-pointer block ${uploading ? "pointer-events-none opacity-70" : ""}`}>
              <p className="text-lg sm:text-xl font-semibold text-slate-900 mb-1.5">{uploading ? "Analyzing your image..." : "Drop a study here"}</p>
              <p className="text-slate-500 text-sm mb-6 max-w-md mx-auto leading-relaxed">
                or browse - PNG or JPG, up to 15 MB. Current study: <span className="font-semibold text-slate-700">{modalityShort(studyModality)}</span>.
              </p>
              <span className="btn-primary inline-flex gap-2 min-h-[44px] px-8">{uploading ? "Processing" : "Select image"}</span>
            </label>
          </div>
        </div>
      </section>

      {latest && (
        <section className="rounded-2xl border border-slate-200/95 bg-white p-5 sm:p-7 shadow-md shadow-slate-200/30 ring-1 ring-slate-100">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
            <h2 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-clinic-600">Latest analysis</h2>
            <span className="text-xs text-slate-500 tabular-nums">{formatDateTime(latest.createdAt)}</span>
          </div>
          <div className="grid sm:grid-cols-[minmax(0,200px),1fr] gap-6 items-start">
            <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-100 aspect-square max-w-[220px] mx-auto sm:mx-0 shadow-inner">
              <img src={latest.imageUrl} alt="Uploaded study preview" className="w-full h-full object-cover" />
            </div>
            <SimpleFindingDisplay
              predictionLabel={latest.predictionLabel}
              classScores={latest.classScores}
              studyModality={latest.studyModality}
              screeningResult={latest.screeningResult}
            />
          </div>
        </section>
      )}

      <section className="pt-1">
        {loading ? (
          <HistorySkeleton />
        ) : (
          <ul className="space-y-3">
            {reports.map((r) => (
              <li key={r.id} className="group rounded-2xl border border-slate-200/95 bg-white p-4 sm:p-5 shadow-sm transition-all duration-200 hover:border-clinic-200/90 hover:shadow-md">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-stretch sm:gap-5">
                  <div className="relative w-full sm:w-36 h-40 sm:h-32 shrink-0 rounded-xl overflow-hidden border border-slate-200 bg-slate-100">
                    <img src={r.imageUrl} alt="Uploaded report preview" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex min-w-0 flex-1 flex-col justify-between gap-3 sm:flex-row sm:items-center">
                    <div className="min-w-0 space-y-2">
                      <SimpleFindingDisplay compact predictionLabel={r.predictionLabel} classScores={r.classScores} studyModality={r.studyModality} screeningResult={r.screeningResult} />
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-slate-500">
                        <time className="tabular-nums text-slate-600" dateTime={r.createdAt}>
                          {formatDateTime(r.createdAt)}
                        </time>
                        {r.originalName && (
                          <span className="font-mono text-xs text-slate-500 truncate max-w-[min(100%,280px)] sm:max-w-[340px]" title={r.originalName}>
                            {displayFileName(r.originalName)}
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      type="button"
                      title="Delete report"
                      aria-label={`Delete report ${r.originalName || r.id || ""}`}
                      onClick={() => setPendingDelete({ id: r.id, originalName: r.originalName || "" })}
                      disabled={deletingId === r.id}
                      className="group/del inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200/95 bg-slate-50/90 text-slate-500 transition-all duration-200 hover:border-red-200 hover:bg-red-50 hover:text-red-700 disabled:cursor-not-allowed disabled:opacity-50 sm:self-center"
                    >
                      {deletingId === r.id ? "..." : <TrashIcon className="h-[1.15rem] w-[1.15rem]" />}
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <DeleteReportDialog
        open={pendingDelete != null}
        onCancel={() => setPendingDelete(null)}
        onConfirm={confirmDeleteReport}
        isDeleting={pendingDelete != null && deletingId === pendingDelete.id}
        fileHint={pendingDelete?.originalName}
      />
    </div>
  );
}
