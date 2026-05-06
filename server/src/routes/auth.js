import bcrypt from "bcryptjs";
import express from "express";
import jwt from "jsonwebtoken";

const router = express.Router();
const SALT_ROUNDS = 10;

function signToken(userId) {
  return jwt.sign({ sub: userId }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRES_IN || "7d",
  });
}

function isDuplicateKey(err) {
  return err && err.code === 11000;
}

router.post("/register", async (req, res) => {
  try {
    const { email, password, fullName } = req.body || {};
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }
    if (password.length < 8) {
      return res.status(400).json({ error: "Password must be at least 8 characters" });
    }
    const normalizedEmail = String(email).toLowerCase().trim();
    const existing = await User.findOne({ email: normalizedEmail });
    if (existing) {
      return res.status(409).json({ error: "Email already registered" });
    }
    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);
    const user = await User.create({
      email: normalizedEmail,
      passwordHash,
      fullName: fullName ? String(fullName).trim() : "",
    });
    const token = signToken(user._id.toString());
    const uid = user._id.toString();
    res.status(201).json({
      token,
      user: { id: uid, email: user.email, fullName: user.fullName },
    });
  } catch (e) {
    if (isDuplicateKey(e)) {
      return res.status(409).json({ error: "Email already registered" });
    }
    console.error(e);
    res.status(500).json({
      error:
        process.env.NODE_ENV === "production"
          ? "Registration failed"
          : e.message || "Registration failed",
    });
  }
});

router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }
    const user = await User.findOne({ email: email.toLowerCase().trim() });
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }
    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) {
      return res.status(401).json({ error: "Invalid credentials" });
    }
    const token = signToken(user._id.toString());
    res.json({
      token,
      user: { id: user._id.toString(), email: user.email, fullName: user.fullName },
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({
      error:
        process.env.NODE_ENV === "production" ? "Login failed" : e.message || "Login failed",
    });
  }
});

export default router;
