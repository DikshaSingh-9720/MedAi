import axios from "axios";
import { v2 as cloudinary } from "cloudinary";
import express from "express";
import fs from "fs/promises";
import FormData from "form-data";
import mongoose from "mongoose";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";

import { guestUser } from "../middleware/auth.js";
import Report from "../models/Report.js";
import { filenameModalityMismatchMessage } from "../modalityUploadGuard.js";

const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 15 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) return cb(new Error("Only image uploads are allowed"));
    cb(null, true);
  },
});

const router = express.Router();
router.use(guestUser);

const ML_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";
const ALLOWED_MODALITIES = new Set(["xray", "ct", "ultrasound", "mri"]);
const isDbReady = () => mongoose.connection.readyState === 1;
const memoryReportsByUser = new Map();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uploadsRoot = path.join(__dirname, "..", "..", "uploads");

function getCloudinaryConfig() {
  const cloudName = process.env.CLOUDINARY_CLOUD_NAME || "";
  const apiKey = process.env.CLOUDINARY_API_KEY || "";
  const apiSecret = process.env.CLOUDINARY_API_SECRET || "";
  const enabled = Boolean(cloudName && apiKey && apiSecret);
  if (enabled) {
    cloudinary.config({ cloud_name: cloudName, api_key: apiKey, api_secret: apiSecret, secure: true });
  }
  return { enabled };
}

function normalizeModality(raw) {
  const m = String(raw || "xray").trim().toLowerCase();
  if (m === "ctscan" || m === "ct_scan") return "ct";
  return ALLOWED_MODALITIES.has(m) ? m : "xray";
}

function memoryListForUser(userId) {
  const key = String(userId || "guest");
  if (!memoryReportsByUser.has(key)) memoryReportsByUser.set(key, []);
  return memoryReportsByUser.get(key);
}

function uploadBufferToCloudinary(buffer, { originalname, studyModality, userId }) {
  return new Promise((resolve, reject) => {
    const safeBase = String(originalname || "image")
      .replace(/\.[^/.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .slice(0, 60);
    const publicId = `${String(userId)}_${Date.now()}_${safeBase || "scan"}`;
    const stream = cloudinary.uploader.upload_stream(
      { resource_type: "image", folder: `medai/${studyModality}`, public_id: publicId, overwrite: false },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    stream.end(buffer);
  });
}

async function saveBufferLocally(buffer, { originalname, studyModality, userId }) {
  const ext = path.extname(String(originalname || "")).toLowerCase() || ".jpg";
  const safeBase = String(originalname || "image")
    .replace(/\.[^/.]+$/, "")
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .slice(0, 60);
  const filename = `${String(userId)}_${Date.now()}_${safeBase || "scan"}${ext}`;
  const relDir = path.join(studyModality);
  const absDir = path.join(uploadsRoot, relDir);
  await fs.mkdir(absDir, { recursive: true });
  const absPath = path.join(absDir, filename);
  await fs.writeFile(absPath, buffer);
  return {
    secure_url: `/uploads/${studyModality}/${filename}`,
    public_id: "",
  };
}

function normalizeUltrasoundResult(classScores, fallbackResult = "") {
  const entries = Object.entries(classScores || {});
  if (entries.length !== 2) return fallbackResult || "";
  const lower = new Map(entries.map(([k, v]) => [String(k).toLowerCase(), Number(v)]));
  const pNot = Number(lower.get("not cancer") ?? lower.get("not_cancer") ?? -1);
  const pCan = Number(lower.get("cancer") ?? -1);
  if (pNot < 0 || pCan < 0) return fallbackResult || "";
  // Match Colab final prediction code:
  // if Cancer_score >= 0.85 -> "Cancer" else "Not Cancer"
  const t = Number(process.env.MEDAI_US_CANCER_THRESHOLD || 0.85);
  return pCan >= t ? "Cancer" : "Not Cancer";
}

function normalizeCtScanResult(predictionLabel, classScores, fallbackResult = "") {
  const raw = String(predictionLabel || "").toLowerCase().replace(/_/g, " ").trim();
  if (raw.includes("normal")) return "NO LUNG CANCER (NORMAL)";
  if (raw.includes("malig")) return "LUNG CANCER DETECTED - MALIGNANT";
  if (raw.includes("benign") || raw.includes("bengin")) return "LUNG CANCER DETECTED - BENIGN";

  // Fallback to class-scores argmax when label is noisy/unknown.
  const entries = Object.entries(classScores || {});
  if (!entries.length) return fallbackResult || "";
  const [topLabel] = entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0];
  const t = String(topLabel || "").toLowerCase().replace(/_/g, " ").trim();
  if (t.includes("normal")) return "NO LUNG CANCER (NORMAL)";
  if (t.includes("malig")) return "LUNG CANCER DETECTED - MALIGNANT";
  if (t.includes("benign") || t.includes("bengin")) return "LUNG CANCER DETECTED - BENIGN";
  return fallbackResult || "";
}

function normalizeMriResult(predictionLabel, classScores, fallbackResult = "") {
  const raw = String(predictionLabel || "").toLowerCase().replace(/_/g, " ").trim();
  if (raw.includes("notumor") || raw.includes("no tumor") || raw === "notumor") {
    return "No Tumor";
  }
  if (raw.includes("glioma") || raw.includes("meningioma") || raw.includes("pituitary")) {
    return "Tumor";
  }

  // Fallback to class-score argmax when label text is inconsistent.
  const entries = Object.entries(classScores || {});
  if (!entries.length) return fallbackResult || "";
  const [topLabel] = entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0];
  const t = String(topLabel || "").toLowerCase().replace(/_/g, " ").trim();
  if (t.includes("notumor") || t.includes("no tumor") || t === "notumor") return "No Tumor";
  if (t.includes("glioma") || t.includes("meningioma") || t.includes("pituitary")) return "Tumor";
  return fallbackResult || "";
}

function normalizeXrayResult(predictionLabel, classScores, fallbackResult = "") {
  const raw = String(predictionLabel || "").toLowerCase().replace(/_/g, " ").trim();
  const entries = Object.entries(classScores || {});
  const lower = new Map(entries.map(([k, v]) => [String(k).toLowerCase(), Number(v)]));

  // Colab-style threshold rule for binary models.
  const t = Number(process.env.MEDAI_XRAY_POS_THRESHOLD || 0.85);

  // If this xray model is fracture-oriented, return Fractured / Not Fractured.
  const pFrac = Number(lower.get("fractured") ?? lower.get("fracture") ?? -1);
  const pNotFrac = Number(lower.get("not fractured") ?? lower.get("not_fractured") ?? lower.get("normal") ?? -1);
  if (pFrac >= 0 || pNotFrac >= 0 || raw.includes("fractur")) {
    if (pFrac >= 0) return pFrac >= t ? "Fractured" : "Not Fractured";
    if (raw.includes("not fract") || raw.includes("no fracture") || raw.includes("normal")) return "Not Fractured";
    return "Fractured";
  }

  // If labels look cancer-oriented, use Cancer / Not Cancer style.
  const pCan = Number(lower.get("cancer") ?? -1);
  const pNotCan = Number(lower.get("not cancer") ?? lower.get("not_cancer") ?? -1);
  if (pCan >= 0 || pNotCan >= 0 || raw.includes("cancer")) {
    if (pCan >= 0) return pCan >= t ? "Cancer" : "Not Cancer";
    if (raw.includes("not cancer") || raw.includes("no cancer")) return "Not Cancer";
    return "Cancer";
  }

  // Generic binary fallback by top score.
  if (entries.length === 2) {
    const [topLabel, topScore] = entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0];
    return Number(topScore) >= t ? String(topLabel) : `Not ${String(topLabel)}`;
  }

  return fallbackResult || "";
}

router.get("/", async (req, res) => {
  if (!isDbReady()) return res.json(memoryListForUser(req.userId));
  try {
    const list = await Report.find({ userId: req.userId }).sort({ createdAt: -1 }).lean();
    return res.json(
      list.map((r) => ({
        id: r._id,
        imageUrl: r.imageUrl,
        originalName: r.originalName,
        studyModality: r.studyModality || "xray",
        predictionLabel: r.predictionLabel,
        screeningResult: r.screeningResult || "",
        confidence: r.confidence,
        classScores: r.classScores && typeof r.classScores === "object" ? r.classScores : {},
        disclaimer: r.disclaimer,
        modelSource: r.modelSource,
        createdAt: r.createdAt,
      }))
    );
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Failed to list reports" });
  }
});

router.post("/", upload.single("image"), async (req, res) => {
  const { enabled: cloudinaryEnabled } = getCloudinaryConfig();
  if (!req.file) return res.status(400).json({ error: "No image file (field name: image)" });

  const studyModality = normalizeModality(req.body?.modality);
  const nameMismatch = filenameModalityMismatchMessage(req.file.originalname, studyModality);
  if (nameMismatch) return res.status(400).json({ error: nameMismatch });

  try {
    const mlForm = new FormData();
    mlForm.append("file", req.file.buffer, { filename: req.file.originalname, contentType: req.file.mimetype });
    mlForm.append("modality", studyModality);
    const mlResponse = await axios.post(`${ML_URL}/predict`, mlForm, {
      headers: mlForm.getHeaders(),
      timeout: 300000,
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      validateStatus: () => true,
    });

    if (mlResponse.status >= 400) {
      const detail = mlResponse.data?.detail || mlResponse.data?.message || JSON.stringify(mlResponse.data || {});
      throw Object.assign(new Error(`ML service ${mlResponse.status}: ${detail}`), { response: mlResponse });
    }

    const data = mlResponse.data;
    let uploaded;
    if (cloudinaryEnabled) {
      try {
        uploaded = await uploadBufferToCloudinary(req.file.buffer, {
          originalname: req.file.originalname,
          studyModality,
          userId: req.userId,
        });
      } catch (e) {
        console.warn("Cloudinary upload failed; falling back to local uploads:", e?.message || e);
        uploaded = await saveBufferLocally(req.file.buffer, {
          originalname: req.file.originalname,
          studyModality,
          userId: req.userId,
        });
      }
    } else {
      uploaded = await saveBufferLocally(req.file.buffer, {
        originalname: req.file.originalname,
        studyModality,
        userId: req.userId,
      });
    }

    const classScores = data.class_scores || data.classScores || {};
    let screeningResult = typeof data.result === "string" ? data.result : "";
    if (studyModality === "ultrasound") {
      screeningResult = normalizeUltrasoundResult(classScores, screeningResult);
    } else if (studyModality === "ct") {
      screeningResult = normalizeCtScanResult(data.label, classScores, screeningResult);
    } else if (studyModality === "mri") {
      screeningResult = normalizeMriResult(data.label, classScores, screeningResult);
    } else if (studyModality === "xray") {
      screeningResult = normalizeXrayResult(data.label, classScores, screeningResult);
    }

    if (!isDbReady()) {
      const memoryReport = {
        id: `temp-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
        imageUrl: uploaded.secure_url,
        imagePublicId: uploaded.public_id,
        originalName: req.file.originalname,
        studyModality,
        predictionLabel: data.label,
        screeningResult,
        confidence: data.confidence,
        classScores,
        disclaimer: data.disclaimer || "",
        modelSource: data.model_source || data.modelSource || "",
        createdAt: new Date().toISOString(),
      };
      const list = memoryListForUser(req.userId);
      list.unshift(memoryReport);
      return res.status(201).json(memoryReport);
    }

    const report = await Report.create({
      userId: req.userId,
      imageUrl: uploaded.secure_url,
      imagePublicId: uploaded.public_id,
      originalName: req.file.originalname,
      mimeType: req.file.mimetype,
      studyModality,
      predictionLabel: data.label,
      screeningResult,
      confidence: data.confidence,
      classScores,
      disclaimer: data.disclaimer || "",
      modelSource: data.model_source || data.modelSource || "",
    });
    return res.status(201).json({
      id: report._id,
      imageUrl: report.imageUrl,
      originalName: report.originalName,
      studyModality: report.studyModality || "xray",
      predictionLabel: report.predictionLabel,
      screeningResult: report.screeningResult || "",
      confidence: report.confidence,
      classScores: report.classScores || {},
      disclaimer: report.disclaimer,
      modelSource: report.modelSource,
      createdAt: report.createdAt,
    });
  } catch (err) {
    console.error(err.response?.data || err.message);
    const detail = err.response?.data?.detail;
    const msg =
      err.code === "ECONNREFUSED"
        ? "ML service unavailable. Start the Python service on port 8000."
        : err.code === "ECONNRESET" || err.message?.includes("ECONNRESET")
          ? "Connection reset during analysis (often ML timeout or server reload). Retry once; ensure uvicorn on port 8000 is running."
          : typeof detail === "string"
            ? detail
            : err.response?.data?.message || err.message || "Prediction failed";
    return res.status(502).json({ error: msg });
  }
});

router.delete("/:id", async (req, res) => {
  try {
    const reportId = req.params.id;

    // DB check
    if (!isDbReady()) {
      const list = memoryListForUser(req.userId);
      const index = list.findIndex((r) => r.id === reportId);
      if (index === -1) {
        return res.status(404).json({ error: "Report not found" });
      }

      const removed = list.splice(index, 1)[0];
      return res.json({ success: true, removed });
    }

    // DB delete
    const report = await Report.findOneAndDelete({
      _id: reportId,
      userId: req.userId,
    });

    if (!report) {
      return res.status(404).json({ error: "Report not found" });
    }

    // 🔥 Cloudinary delete (important)
    if (report.imagePublicId) {
      try {
        await cloudinary.uploader.destroy(report.imagePublicId);
      } catch (e) {
        console.warn("Cloudinary delete failed:", e.message);
      }
    }

    return res.json({ success: true });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Delete failed" });
  }
});

export default router;
