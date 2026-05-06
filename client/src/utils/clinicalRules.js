function formatLabel(key) {
  return String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function binaryFractureTitle(predictionLabel) {
  const t = String(predictionLabel || "")
    .toLowerCase()
    .replace(/_/g, " ");
  if (t.startsWith("not ") || t.startsWith("no ") || /^not_|^no_/.test(t) || /^(normal|negative|healthy|intact|unfractured)\b/.test(t)) {
    return "No fracture detected";
  }
  if (t.includes("fractur")) return "Bone fracture suspected";
  return formatLabel(predictionLabel);
}

function isBinaryFractureModel(classScores) {
  return Object.keys(classScores || {}).length === 2;
}

const LUNG_CT_STAGES = ["Normal", "Benign", "Malignant"];

function lungCtOrderedScores(classScores) {
  const raw = classScores || {};
  const byLower = new Map(Object.entries(raw).map(([k, v]) => [String(k).toLowerCase(), [k, Number(v)]]));
  const rows = [];
  for (const stage of LUNG_CT_STAGES) {
    const hit = byLower.get(stage.toLowerCase());
    if (hit) rows.push(hit);
  }
  return rows.length === 3 ? rows : null;
}

function lungCtLayVerdict(classScores) {
  const rows = lungCtOrderedScores(classScores);
  if (!rows) return "unclear";

  const m = Object.fromEntries(rows.map(([k, v]) => [k.toLowerCase(), Number(v)]));
  const pN = m.normal ?? 0;
  const pB = m.benign ?? 0;
  const pM = m.malignant ?? 0;
  const topP = Math.max(pN, pB, pM);
  const topLabel = pN === topP ? "normal" : pB === topP ? "benign" : "malignant";

  const cancerPossible = topLabel === "malignant" || pM >= 0.41 || (pM >= 0.34 && pM + 0.04 >= topP);
  const cancerUnlikely = pM < 0.27 && topP >= pM + 0.1 && topLabel !== "malignant";

  if (cancerPossible) return "yes";
  if (cancerUnlikely) return "no";
  return "unclear";
}

function breastScreenTitle(predictionLabel) {
  const t = String(predictionLabel || "").toLowerCase().replace(/_/g, " ");
  if (t.includes("no breast cancer") || t.includes("not breast")) return "No breast cancer sign";
  if (t.includes("breast cancer")) return "Breast cancer";
  if (t.includes("not cancer") || (t.includes("no ") && t.includes("cancer"))) return "No breast cancer sign";
  if (t.includes("cancer")) return "Breast cancer";
  if (t.includes("malignant")) return "Breast cancer";
  if (t.includes("benign")) return "Benign";
  if (/\bnormal\b/.test(t)) return "Normal";
  return formatLabel(predictionLabel);
}

function brainMriScreenTitle(predictionLabel) {
  const t = String(predictionLabel || "").toLowerCase().replace(/_/g, " ");
  if (t.includes("notumor") || t.includes("no tumor")) return "No tumor";
  if (t.includes("glioma")) return "Glioma";
  if (t.includes("meningioma")) return "Meningioma";
  if (t.includes("pituitary")) return "Pituitary";
  if (t.includes("no brain tumor")) return "No brain tumor sign";
  if (t.includes("brain tumor") || (t.includes("tumor") && !t.includes("no "))) return "Brain tumor";
  if (t.includes("mass") || t.includes("lesion")) return "Brain tumor";
  return formatLabel(predictionLabel);
}

function ultrasoundCancerSummary(predictionLabel) {
  const title = breastScreenTitle(predictionLabel);
  const t = title.toLowerCase();
  const isCancer = t.includes("breast cancer") && !t.includes("no breast");
  return { isCancer };
}

function mriTumorSummary(predictionLabel) {
  const title = brainMriScreenTitle(predictionLabel);
  const t = title.toLowerCase();
  const isNeg = t === "no tumor" || t.includes("no brain") || t.includes("no tumor");
  const isTumorType = t === "glioma" || t === "meningioma" || t === "pituitary";
  const isTumor = isTumorType || ((t.includes("brain tumor") || t.includes("tumor") || t.includes("mass")) && !isNeg);
  return { isTumor: isTumor && !isNeg };
}

export function getSimpleAnswer(predictionLabel, classScores, studyModality, screeningResult) {
  const mod = studyModality || "xray";

  if (mod === "ct") {
    const sr = screeningResult && String(screeningResult).trim();
    if (sr) {
      const r = sr.toLowerCase();
      if (r.includes("no lung cancer") || r.includes("no cancer (normal)")) return { topic: "Lung cancer", answer: "no" };
      if (r.includes("malignant")) return { topic: "Lung cancer", answer: "yes" };
      if (r.includes("benign") || r.includes("unlikely")) return { topic: "Lung cancer", answer: "no" };
      return { topic: "Lung cancer", answer: "unclear" };
    }
    return { topic: "Lung cancer", answer: lungCtLayVerdict(classScores) };
  }

  if (mod === "ultrasound") {
    const sr = screeningResult && String(screeningResult).trim();
    if (sr) {
      const r = sr.toLowerCase();
      if (r.includes("inconclusive") || r.includes("retest") || r.includes("verify clinically")) {
        return { topic: "Breast cancer", answer: "unclear" };
      }
      if (r.includes("no breast cancer") || (r.includes("no cancer") && r.includes("ultrasound"))) {
        return { topic: "Breast cancer", answer: "no" };
      }
      if (r.includes("breast cancer detected")) return { topic: "Breast cancer", answer: "yes" };
      if (r.includes("breast cancer likely")) return { topic: "Breast cancer", answer: "unclear" };
      return { topic: "Breast cancer", answer: "unclear" };
    }
    return { topic: "Breast cancer", answer: ultrasoundCancerSummary(predictionLabel).isCancer ? "yes" : "no" };
  }

  if (mod === "mri") {
    const sr = screeningResult && String(screeningResult).trim();
    if (sr) {
      const r = sr.toLowerCase();
      if (r.includes("no brain tumor")) return { topic: "Brain tumor", answer: "no" };
      if (r.includes("brain tumor")) return { topic: "Brain tumor", answer: "yes" };
      return { topic: "Brain tumor", answer: "unclear" };
    }
    return { topic: "Brain tumor", answer: mriTumorSummary(predictionLabel).isTumor ? "yes" : "no" };
  }

  if (isBinaryFractureModel(classScores)) {
    const yes = binaryFractureTitle(predictionLabel) === "Bone fracture suspected";
    return { topic: "Fracture", answer: yes ? "yes" : "no" };
  }

  return { topic: "Finding", answer: "unclear" };
}

export function getFindingLabel(predictionLabel, classScores, studyModality, screeningResult) {
  const mod = studyModality || "xray";
  const sr = String(screeningResult || "").trim();
  const srLower = sr.toLowerCase();

  if (mod === "ct") {
    if (sr) return sr;
    const t = String(predictionLabel || "").toLowerCase();
    if (t.includes("normal")) return "NO LUNG CANCER (NORMAL)";
    if (t.includes("malig")) return "LUNG CANCER DETECTED - MALIGNANT";
    if (t.includes("benign") || t.includes("bengin")) return "LUNG CANCER DETECTED - BENIGN";
    return "Lung CT result unavailable";
  }

  if (mod === "mri") {
    if (srLower === "tumor" || srLower === "no tumor") return sr;
    const t = String(predictionLabel || "").toLowerCase();
    if (t.includes("notumor") || t.includes("no tumor")) return "No Tumor";
    if (t.includes("glioma") || t.includes("meningioma") || t.includes("pituitary") || t.includes("tumor")) return "Tumor";
    return sr || "MRI result unavailable";
  }

  if (mod === "ultrasound") {
    if (srLower === "cancer" || srLower === "not cancer") return sr;
    const t = String(predictionLabel || "").toLowerCase();
    if (t.includes("not cancer") || t.includes("no cancer") || t.includes("normal")) return "Not Cancer";
    if (t.includes("cancer") || t.includes("malignant")) return "Cancer";
    return sr || "Ultrasound result unavailable";
  }

  // xray
  if (sr) return sr;
  const t = String(predictionLabel || "").toLowerCase();
  if (t.includes("not fract") || t.includes("no fracture") || t.includes("normal")) return "Not Fractured";
  if (t.includes("fractur")) return "Fractured";
  return "X-ray result unavailable";
}

export function simpleAnswerBadgeClass(answer) {
  if (answer === "yes") return "bg-red-50 text-red-950 ring-red-200/90";
  if (answer === "no") return "bg-emerald-50 text-emerald-900 ring-emerald-200/70";
  return "bg-amber-50 text-amber-950 ring-amber-200/80";
}

export function modalityShort(m) {
  if (m === "ct") return "Lung CT";
  if (m === "mri") return "Brain MRI";
  if (m === "ultrasound") return "Ultrasound";
  return "X-ray";
}

export function displayFileName(name, max = 40) {
  const s = String(name || "").trim();
  if (!s) return "";
  if (s.length <= max) return s;
  const ext = s.includes(".") ? s.slice(s.lastIndexOf(".")) : "";
  const base = ext ? s.slice(0, s.length - ext.length) : s;
  const keep = max - ext.length - 3;
  return `${base.slice(0, Math.max(8, keep))}...${ext}`;
}
