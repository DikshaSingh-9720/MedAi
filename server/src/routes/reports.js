import axios from "axios";
import { v2 as cloudinary } from "cloudinary";
import express from "express";
import FormData from "form-data";
import mongoose from "mongoose";
import multer from "multer";

import { guestUser } from "../middleware/auth.js";
import Report from "../models/Report.js";
import { filenameModalityMismatchMessage } from "../modalityUploadGuard.js";

function getCloudinaryConfig() {
  const cloudName = process.env.CLOUDINARY_CLOUD_NAME || "";
  const apiKey = process.env.CLOUDINARY_API_KEY || "";
  const apiSecret = process.env.CLOUDINARY_API_SECRET || "";
  const enabled = Boolean(cloudName && apiKey && apiSecret);
  if (enabled) {
    cloudinary.config({
      cloud_name: cloudName,
      api_key: apiKey,
      api_secret: apiSecret,
      secure: true,
    });
  }
  return { enabled };
}

const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: { fileSize: 15 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Only image uploads are allowed"));
    }
    cb(null, true);
  },
});

const router = express.Router();
router.use(guestUser);

const ML_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

const ALLOWED_MODALITIES = new Set(["xray", "ct", "ultrasound", "mri"]);
const isDbReady = () => mongoose.connection.readyState === 1;
const memoryReportsByUser = new Map();

function normalizeModality(raw) {
  const m = String(raw || "xray")
    .trim()
    .toLowerCase();
  if (m === "ctscan" || m === "ct_scan") return "ct";
  return ALLOWED_MODALITIES.has(m) ? m : "xray";
}

function normalizeUltrasoundResult(classScores, fallbackResult = "") {
  const entries = Object.entries(classScores || {});
  if (entries.length !== 2) return fallbackResult || "";
  const lower = new Map(entries.map(([k, v]) => [String(k).toLowerCase(), Number(v)]));
  const pNot = Number(lower.get("not cancer") ?? lower.get("not_cancer") ?? -1);
  const pCan = Number(lower.get("cancer") ?? -1);
  if (pNot < 0 || pCan < 0) return fallbackResult || "";

  const t = Number(process.env.MEDAI_US_RULE_THRESHOLD || 0.9999);
  const margin = Number(process.env.MEDAI_US_RULE_MARGIN || 0.2);
  if (pCan >= t && pCan - pNot >= margin) return "Breast Cancer Detected (Ultrasound)";
  if (pNot >= t && pNot - pCan >= margin) return "No Breast Cancer Detected (Ultrasound)";
  return "Ultrasound Result Inconclusive - Please retest/verify clinically";
}

function memoryListForUser(userId) {
  const key = String(userId || "guest");
  if (!memoryReportsByUser.has(key)) {
    memoryReportsByUser.set(key, []);
  }
  return memoryReportsByUser.get(key);
}

function uploadBufferToCloudinary(buffer, { originalname, mimetype, studyModality, userId }) {
  return new Promise((resolve, reject) => {
    const safeBase = String(originalname || "image")
      .replace(/\.[^/.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]/g, "_")
      .slice(0, 60);
    const publicId = `${String(userId)}_${Date.now()}_${safeBase || "scan"}`;
    const stream = cloudinary.uploader.upload_stream(
      {
        resource_type: "image",
        folder: `medai/${studyModality}`,
        public_id: publicId,
        overwrite: false,
      },
      (error, result) => {
        if (error) return reject(error);
        resolve(result);
      }
    );
    stream.end(buffer);
  });
}

router.delete("/:id", async (req, res) => {
  const { enabled: cloudinaryEnabled } = getCloudinaryConfig();
  const { id } = req.params;
  if (!isDbReady()) {
    const list = memoryListForUser(req.userId);
    const idx = list.findIndex((r) => String(r.id) == String(id));
    if (idx < 0) {
      return res.status(404).json({ error: "Report not found" });
    }
    const deleted = list[idx];
    list.splice(idx, 1);
    if (cloudinaryEnabled && deleted.imagePublicId) {
      try {
        await cloudinary.uploader.destroy(deleted.imagePublicId, { resource_type: "image" });
      } catch (e) {
        console.warn("Cloudinary delete failed:", e?.message || e);
      }
    }
    return res.json({ ok: true });
  }
  if (!mongoose.Types.ObjectId.isValid(id)) {
    return res.status(400).json({ error: "Invalid report id" });
  }
  try {
    const deleted = await Report.findOneAndDelete({ _id: id, userId: req.userId });
    if (!deleted) {
      return res.status(404).json({ error: "Report not found" });
    }
    if (cloudinaryEnabled && deleted.imagePublicId) {
      try {
        await cloudinary.uploader.destroy(deleted.imagePublicId, { resource_type: "image" });
      } catch (e) {
        console.warn("Cloudinary delete failed:", e?.message || e);
      }
    }
    res.json({ ok: true });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Failed to delete report" });
  }
});

router.get("/", async (req, res) => {
  if (!isDbReady()) {
    return res.json(memoryListForUser(req.userId));
  }
  try {
    const list = await Report.find({ userId: req.userId })
      .sort({ createdAt: -1 })
      .lean();
    res.json(
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
    res.status(500).json({ error: "Failed to list reports" });
  }
});

function uploadSingle(req, res, next) {
  upload.single("image")(req, res, (err) => {
    if (err) {
      return res.status(400).json({ error: err.message || "Upload failed" });
    }
    next();
  });
}

router.post("/", uploadSingle, async (req, res) => {
  const { enabled: cloudinaryEnabled } = getCloudinaryConfig();
  if (!cloudinaryEnabled) {
    return res.status(503).json({
      error: "Cloudinary not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET.",
    });
  }
  if (!req.file) {
    return res.status(400).json({ error: "No image file (field name: image)" });
  }

  const studyModality = normalizeModality(req.body?.modality);

  const nameMismatch = filenameModalityMismatchMessage(req.file.originalname, studyModality);
  if (nameMismatch) {
    return res.status(400).json({ error: nameMismatch });
  }

  try {
    const mlForm = new FormData();
    mlForm.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    mlForm.append("modality", studyModality);

    const [uploaded, mlResponse] = await Promise.all([
      uploadBufferToCloudinary(req.file.buffer, {
        originalname: req.file.originalname,
        mimetype: req.file.mimetype,
        studyModality,
        userId: req.userId,
      }),
      axios.post(`${ML_URL}/predict`, mlForm, {
        headers: mlForm.getHeaders(),
        timeout: 300000,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        validateStatus: () => true,
      }),
    ]);

    if (mlResponse.status >= 400) {
      const detail =
        mlResponse.data?.detail ||
        mlResponse.data?.message ||
        JSON.stringify(mlResponse.data || {});
      throw Object.assign(new Error(`ML service ${mlResponse.status}: ${detail}`), {
        response: mlResponse,
      });
    }

    const data = mlResponse.data;
    const classScores = data.class_scores || data.classScores || {};
    let screeningResult = typeof data.result === "string" ? data.result : "";
    if (studyModality === "ultrasound") {
      screeningResult = normalizeUltrasoundResult(classScores, screeningResult);
    }

    if (!isDbReady()) {
      const memoryReport = {
        id: `temp-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
        imageUrl: uploaded.secure_url,
        imagePublicId: uploaded.public_id,
        originalName: req.file.originalname,
        studyModality,
        predictionLabel: data.label,
        screeningResult: screeningResult || "",
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
      screeningResult: screeningResult || "",
      confidence: data.confidence,
      classScores,
      disclaimer: data.disclaimer || "",
      modelSource: data.model_source || data.modelSource || "",
    });

    res.status(201).json({
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
    res.status(502).json({ error: msg });
  }
});

export default router;
