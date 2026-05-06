import jwt from "jsonwebtoken";
import mongoose from "mongoose";

/** Fixed user id for anonymous / no-login mode (all reports share this workspace). */
const GUEST_USER_ID = process.env.MEDAI_GUEST_USER_ID || "000000000000000000000001";

export function guestUser(req, res, next) {
  try {
    req.userId = new mongoose.Types.ObjectId(GUEST_USER_ID);
  } catch {
    return res.status(500).json({ error: "Invalid MEDAI_GUEST_USER_ID" });
  }
  next();
}

export function authRequired(req, res, next) {
  const header = req.headers.authorization;
  if (!header?.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Missing or invalid Authorization header" });
  }
  const token = header.slice(7);
  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = payload.sub;
    next();
  } catch {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}
