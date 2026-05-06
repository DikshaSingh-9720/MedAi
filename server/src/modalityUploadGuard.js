/**
 * Filename hints that suggest a *different* imaging modality than selected.
 * Not clinically reliable — reduces accidental wrong-type uploads; images are still 2D bitmaps.
 */
const FORBIDDEN_SUBSTRINGS = {
  xray: [
    "mri",
    "fmri",
    "ctscan",
    "ct_scan",
    "lung_ct",
    "lung-ct",
    "chest_ct",
    "chest-ct",
    "chestct",
    "ultrasound",
    "sonograph",
    "usg",
    "busi",
    "brats",
    "brain_",
    "brain-",
  ],
  ct: [
    "mri",
    "brain_mri",
    "brainmri",
    "ultrasound",
    "sonograph",
    "xray",
    "xr_",
    "_xr",
    "radiograph",
    "fracture",
    "bone_xray",
    "wrist_xray",
    "busi",
  ],
  ultrasound: [
    "mri",
    "ctscan",
    "ct_scan",
    "lung_ct",
    "lung-ct",
    "lungct",
    "chest_ct",
    "xray",
    "xr_",
    "radiograph",
    "fracture",
    "brain_",
    "brats",
  ],
  mri: [
    "xray",
    "radiograph",
    "fracture",
    "ultrasound",
    "sonograph",
    "busi",
    "ctscan",
    "ct_scan",
    "lung_ct",
    "lung-ct",
    "chest_ct",
    "chest-ct",
  ],
};

const MODALITY_LABEL = {
  xray: "X-ray (bone fracture)",
  ct: "Lung CT",
  ultrasound: "Breast ultrasound",
  mri: "Brain MRI slice",
};

/**
 * @param {string} originalName
 * @param {string} modality - xray | ct | ultrasound | mri
 * @returns {string|null} error message or null if allowed
 */
export function filenameModalityMismatchMessage(originalName, modality) {
  const m = String(modality || "xray").toLowerCase();
  const list = FORBIDDEN_SUBSTRINGS[m];
  if (!list) return null;
  const norm = String(originalName || "")
    .toLowerCase()
    .replace(/\\/g, "/");
  const leaf = norm.split("/").pop() || norm;
  for (const hint of list) {
    if (leaf.includes(hint)) {
      const want = MODALITY_LABEL[m] || m;
      return `This file name looks like a different study type (“${hint}”). You selected ${want}. Rename the file to remove modality hints from other types, or switch the study button above to match your image.`;
    }
  }
  return null;
}
