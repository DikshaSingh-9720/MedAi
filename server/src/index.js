import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import mongoose from "mongoose";
import path from "path";
import { fileURLToPath } from "url";

import authRoutes from "./routes/auth.js";
import reportRoutes from "./routes/reports.js";



dotenv.config();

if (!process.env.JWT_SECRET) {
  process.env.JWT_SECRET = "medai-dev-only-change-in-production";
  console.warn("JWT_SECRET missing — using insecure dev default.");
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 5000;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || "http://localhost:5173";
let dbConnected = false;

app.use(
  cors({
    origin: CLIENT_ORIGIN.split(",").map((s) => s.trim()),
    credentials: true,
  })
);
app.use(express.json());

const uploadsDir = path.join(__dirname, "..", "uploads");
app.use("/uploads", express.static(uploadsDir));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, service: "medai-server", db: dbConnected ? "connected" : "disconnected" });
});

app.use((req, res, next) => {
  if (!dbConnected && req.path.startsWith("/api/auth")) {
    return res.status(503).json({
      error: "Database unavailable. Start MongoDB and restart the server.",
    });
  }
  next();
});

app.use("/api/auth", authRoutes);
app.use("/api/reports", reportRoutes);

app.use((err, _req, res, _next) => {
  const isBadJson =
    err?.type === "entity.parse.failed" ||
    (err instanceof SyntaxError && "body" in err);
  if (isBadJson) {
    return res.status(400).json({ error: "Invalid JSON body" });
  }
  console.error(err);
  const status =
    typeof err.status === "number" && err.status >= 400 && err.status < 600
      ? err.status
      : 500;
  const expose =
    status < 500 || process.env.NODE_ENV !== "production"
      ? err.message || "Request error"
      : "Internal server error";
  res.status(status).json({ error: expose });
});

const mongoUri = process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/medai";
let started = false;

function startServer() {
  if (started) return;
  started = true;
  const server = app.listen(PORT, () => {
    console.log(`MedAI server http://localhost:${PORT}`);
    if (!dbConnected) {
      console.warn("Running without database; /api/auth and /api/reports return 503.");
    }
  });
  server.timeout = 300000;
  server.keepAliveTimeout = 65000;
  server.headersTimeout = 66000;
  server.on("error", (err) => {
    if (err?.code === "EADDRINUSE") {
      console.error(`Port ${PORT} is already in use. Another server instance is already running.`);
      return;
    }
    throw err;
  });
}

mongoose
  .connect(mongoUri)
  .then(() => {
    dbConnected = true;
    startServer();
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
    dbConnected = false;
    startServer();
  });
