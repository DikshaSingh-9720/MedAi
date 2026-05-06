import mongoose from "mongoose";

const reportSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    imageUrl: { type: String, required: true },
    imagePublicId: { type: String, default: "" },
    originalName: { type: String, default: "" },
    mimeType: { type: String, default: "" },
    studyModality: { type: String, default: "xray" },
    predictionLabel: { type: String, required: true },
    screeningResult: { type: String, default: "" },
    confidence: { type: Number, required: true },
    classScores: { type: Object, default: {} },
    disclaimer: { type: String, default: "" },
    modelSource: { type: String, default: "" },
  },
  { timestamps: true }
);

export default mongoose.model("Report", reportSchema);
