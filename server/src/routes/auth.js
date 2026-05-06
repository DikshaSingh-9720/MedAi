import bcrypt from "bcryptjs";
import express from "express";
import jwt from "jsonwebtoken";
import mongoose from "mongoose";

import User from "../models/User.js";

const router = express.Router();

function sign(user) {
  const payload = { sub: String(user._id), email: user.email };
  const secret = process.env.JWT_SECRET || "medai-dev-only-change-in-production";
  const expiresIn = process.env.JWT_EXPIRES_IN || "7d";
  return jwt.sign(payload, secret, { expiresIn });
}

function sanitize(user) {
  return { id: String(user._id), email: user.email, fullName: user.fullName || "" };
}

router.post("/register", async (req, res) => {
  if (mongoose.connection.readyState !== 1) {
    return res.status(503).json({ error: "Database unavailable. Please try again later." });
  }
  const email = String(req.body?.email || "").trim().toLowerCase();
  const password = String(req.body?.password || "");
  const fullName = String(req.body?.fullName || "").trim();
  if (!email || !password) return res.status(400).json({ error: "Email and password are required" });
  if (password.length < 6) return res.status(400).json({ error: "Password must be at least 6 characters" });

  try {
    const existing = await User.findOne({ email }).lean();
    if (existing) return res.status(409).json({ error: "Email already registered" });
    const passwordHash = await bcrypt.hash(password, 10);
    const user = await User.create({ email, passwordHash, fullName });
    const token = sign(user);
    return res.status(201).json({ token, user: sanitize(user) });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Failed to register user" });
  }
});

router.post("/login", async (req, res) => {
  if (mongoose.connection.readyState !== 1) {
    return res.status(503).json({ error: "Database unavailable. Please try again later." });
  }
  const email = String(req.body?.email || "").trim().toLowerCase();
  const password = String(req.body?.password || "");
  if (!email || !password) return res.status(400).json({ error: "Email and password are required" });

  try {
    const user = await User.findOne({ email });
    if (!user) return res.status(401).json({ error: "Invalid email or password" });
    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ error: "Invalid email or password" });
    const token = sign(user);
    return res.json({ token, user: sanitize(user) });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Failed to login" });
  }
});

export default router;
