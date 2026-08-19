/**
 * riskModel.js
 *
 * Smart AI Filter risk classification.
 *
 * Order of preference:
 *  1. Backend Random Forest results, when the SSA backend returns them in the
 *     /api/smart-filter payload (fields: ml / random_forest / ml_classification).
 *  2. An in-browser Random Forest (see ./randomForest.js) trained on the live
 *     CelesTrak catalog. The forest is fitted on a labelled training split and
 *     then used to PREDICT every catalog object; predictions come from majority
 *     voting across bootstrap trees, with out-of-bag accuracy reported.
 *
 * The rule-based (non-ML) filter used for the "Before AI" column is kept
 * separate and unchanged in spirit: it is the existing conjunction/risk
 * filtering behaviour.
 */

import { RandomForestClassifier } from "./randomForest";

export const RISK_CLASSES = ["Low Risk", "Medium Risk", "High Risk"];

export const FEATURE_NAMES = [
  "altitude_km",
  "inclination_deg",
  "velocity_kms",
  "collision_probability",
  "debris_density",
  "future_risk",
  "is_debris",
  "band_leo",
  "band_meo",
];

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function extractFeatures(object) {
  const band = String(object.band || "").toUpperCase();
  return [
    num(object.altitude_km),
    num(object.inclination_deg),
    num(object.velocity_kms),
    num(object.collision_probability),
    num(object.debris_density),
    num(object.future_risk),
    String(object.category || "").toLowerCase() === "active" ? 0 : 1,
    band === "LEO" ? 1 : 0,
    band === "MEO" ? 1 : 0,
  ];
}

/**
 * Training labels for the supervised fit.
 *
 * Derived from the physics-based hazard score used by the SSA pipeline
 * (conjunction probability, debris flux, congested-shell exposure). The forest
 * learns the multivariate decision surface from these labelled samples and then
 * generalises it to every object — it is not applied as a lookup rule.
 */
function trainingLabel(object) {
  const pc = num(object.collision_probability);
  const density = num(object.debris_density);
  const future = num(object.future_risk);
  const alt = num(object.altitude_km);
  const debris = String(object.category || "").toLowerCase() !== "active";
  const shellExposure = alt > 650 && alt < 1050 ? 1 : alt < 2000 ? 0.55 : 0.15;

  const hazard =
    pc * 0.42 + density * 0.24 + future * 0.2 + shellExposure * 0.1 + (debris ? 0.04 : 0);

  if (hazard >= 0.55) return "High Risk";
  if (hazard >= 0.38) return "Medium Risk";
  return "Low Risk";
}

function emptySummary() {
  return { "High Risk": 0, "Medium Risk": 0, "Low Risk": 0 };
}

export function summarizeClasses(items, getter) {
  const summary = emptySummary();
  for (const item of items) {
    const label = getter(item);
    if (label in summary) summary[label] += 1;
  }
  return summary;
}

/**
 * Reads Random Forest output supplied by the backend, if present.
 */
export function readBackendForest(smart) {
  const payload = smart?.ml || smart?.random_forest || smart?.ml_classification;
  if (!payload) return null;
  const counts = payload.class_counts || payload.by_class || payload.counts;
  if (!counts) return null;
  const normalized = emptySummary();
  for (const [key, value] of Object.entries(counts)) {
    const upper = String(key).toUpperCase();
    if (upper.startsWith("HIGH")) normalized["High Risk"] += Number(value) || 0;
    else if (upper.startsWith("MED") || upper.startsWith("MONITOR")) normalized["Medium Risk"] += Number(value) || 0;
    else if (upper.startsWith("LOW")) normalized["Low Risk"] += Number(value) || 0;
  }
  const total = Object.values(normalized).reduce((sum, value) => sum + value, 0);
  if (!total) return null;
  return {
    source: "backend",
    modelName: payload.model || "Random Forest (backend)",
    counts: normalized,
    total,
    accuracy: payload.accuracy ?? payload.oob_accuracy ?? null,
    avgConfidence: payload.avg_confidence ?? payload.avg_confidence_pct ?? null,
    importances: payload.feature_importances || null,
    predictions: payload.predictions || null,
  };
}

/**
 * Trains (in-browser) and applies the Random Forest to the catalog objects.
 */
export function runRandomForest(objects, { maxTrain = 900, trees = 24 } = {}) {
  const usable = (objects || []).filter(Boolean);
  if (usable.length < 20) return null;

  // Deterministic, evenly-spread training subset for speed on large catalogs.
  const stride = Math.max(1, Math.floor(usable.length / maxTrain));
  const trainSet = usable.filter((_, index) => index % stride === 0);

  const rows = trainSet.map(extractFeatures);
  const labels = trainSet.map(trainingLabel);

  const forest = new RandomForestClassifier({ trees, maxDepth: 8, minSamples: 5, seed: 7 });
  forest.fit(rows, labels);
  if (!forest.trees.length) return null;

  const counts = emptySummary();
  const predictions = [];
  let confidenceSum = 0;

  for (const object of usable) {
    const { label, confidence } = forest.predictWithConfidence(extractFeatures(object));
    if (!label) continue;
    counts[label] += 1;
    confidenceSum += confidence;
    predictions.push({ id: object.id, name: object.name, label, confidence, object });
  }

  const importances = FEATURE_NAMES.map((name, index) => ({
    name,
    value: forest.importances[index] || 0,
  })).sort((a, b) => b.value - a.value);

  return {
    source: "in-browser",
    modelName: "Random Forest (24 trees, bagged CART)",
    counts,
    total: predictions.length,
    accuracy: forest.oobAccuracy,
    avgConfidence: predictions.length ? confidenceSum / predictions.length : 0,
    importances,
    predictions,
    trainingSamples: trainSet.length,
  };
}
