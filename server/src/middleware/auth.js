import jwt from "jsonwebtoken";

export function guestUser(req, _res, next) {
  const auth = req.headers.authorization || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  if (!token) {
    req.userId = "000000000000000000000001";
    return next();
  }
  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET || "medai-dev-only-change-in-production");
    req.userId = String(payload?.sub || payload?.id || "000000000000000000000001");
    return next();
  } catch {
    req.userId = "000000000000000000000001";
    return next();
  }
}

export function requireAuth(req, res, next) {
  const auth = req.headers.authorization || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
  if (!token) return res.status(401).json({ error: "Missing bearer token" });
  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET || "medai-dev-only-change-in-production");
    req.userId = String(payload?.sub || payload?.id);
    if (!req.userId) return res.status(401).json({ error: "Invalid token" });
    return next();
  } catch {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
}
