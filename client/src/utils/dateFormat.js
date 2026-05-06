export function formatDateTime(value) {
  if (!value) return "";
  return new Date(value).toLocaleString();
}

export function formatShortDateTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
