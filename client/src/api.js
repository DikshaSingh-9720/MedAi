import axios from "axios";

/** Do not set global Content-Type — it breaks multipart uploads (needs boundary). */
const BASE_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const api = axios.create({
  baseURL: `${BASE_URL}/api`,
});

export function setAuthToken(token) {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
    localStorage.setItem("medai_token", token);
  } else {
    delete api.defaults.headers.common.Authorization;
    localStorage.removeItem("medai_token");
  }
}


export async function register(body) {
  const { data } = await api.post("/auth/register", body, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
}

export async function login(body) {
  const { data } = await api.post("/auth/login", body, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
}

export async function fetchReports() {
  const { data } = await api.get("/reports");
  return data;
}

export async function uploadReport(file, modality = "xray") {
  const form = new FormData();
  form.append("image", file);
  form.append("modality", modality);
  const { data } = await api.post("/reports", form, {
    timeout: 300000,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  });
  return data;
}

export async function deleteReport(reportId) {
  const { data } = await api.delete(`/reports/${reportId}`);
  return data;
}

export default api;
