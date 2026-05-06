import { getSimpleAnswer, simpleAnswerBadgeClass } from "../../utils/clinicalRules.js";

export default function SimpleFindingDisplay({ predictionLabel, classScores, studyModality, screeningResult, compact }) {
  const { topic, answer } = getSimpleAnswer(predictionLabel, classScores, studyModality, screeningResult);
  const word = answer === "yes" ? "Yes" : answer === "no" ? "No" : "Unclear";
  const badgeClass = simpleAnswerBadgeClass(answer);

  if (compact) {
    return (
      <div className="flex flex-col gap-1.5 min-w-0">
        <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">{topic}</p>
        <span className={`inline-flex w-fit items-center rounded-full px-2.5 py-1 text-xs font-bold ring-1 tabular-nums ${badgeClass}`}>
          {word}
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-slate-200/90 bg-white px-4 py-4 shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 mb-3">{topic}</p>
      <p className={`inline-flex min-w-[5.5rem] items-center justify-center rounded-xl px-5 py-3 text-2xl font-bold tabular-nums ring-1 ${badgeClass}`}>
        {word}
      </p>
    </div>
  );
}
