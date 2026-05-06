import { describe, expect, it } from "vitest";
import { getSimpleAnswer } from "./clinicalRules.js";

describe("getSimpleAnswer", () => {
  it("maps CT malignant screening result to yes", () => {
    const out = getSimpleAnswer("malignant", null, "ct", "Malignant - needs review");
    expect(out).toEqual({ topic: "Lung cancer", answer: "yes" });
  });

  it("maps CT no-cancer screening result to no", () => {
    const out = getSimpleAnswer("normal", null, "ct", "No lung cancer");
    expect(out).toEqual({ topic: "Lung cancer", answer: "no" });
  });

  it("uses fallback class scores for CT when screening result missing", () => {
    const out = getSimpleAnswer("malignant", { Normal: 0.1, Benign: 0.2, Malignant: 0.7 }, "ct", "");
    expect(out).toEqual({ topic: "Lung cancer", answer: "yes" });
  });

  it("maps MRI no brain tumor to no", () => {
    const out = getSimpleAnswer("notumor", {}, "mri", "No brain tumor");
    expect(out).toEqual({ topic: "Brain tumor", answer: "no" });
  });

  it("maps ultrasound breast cancer prediction to yes", () => {
    const out = getSimpleAnswer("breast_cancer", {}, "ultrasound", "");
    expect(out).toEqual({ topic: "Breast cancer", answer: "yes" });
  });

  it("maps binary fracture labels correctly", () => {
    const out = getSimpleAnswer("fracture_detected", { no_fracture: 0.2, fracture: 0.8 }, "xray", "");
    expect(out).toEqual({ topic: "Fracture", answer: "yes" });
  });
});
