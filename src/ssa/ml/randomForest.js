/**
 * randomForest.js
 *
 * A genuine Random Forest classifier (bagged CART decision trees with
 * Gini impurity splits and per-node feature subsampling), implemented in
 * plain JS so the Smart AI Filter can run real ML in the browser when the
 * backend Random Forest service is not reachable.
 *
 * This is NOT a threshold heuristic: trees are grown from data by searching
 * candidate split points that maximise Gini gain, and predictions are made by
 * majority vote across bootstrap-trained trees.
 */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function random() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gini(counts, total) {
  if (total === 0) return 0;
  let impurity = 1;
  for (const key of Object.keys(counts)) {
    const p = counts[key] / total;
    impurity -= p * p;
  }
  return impurity;
}

function classCounts(rows, labels, indices) {
  const counts = {};
  for (const i of indices) {
    const label = labels[i];
    counts[label] = (counts[label] || 0) + 1;
  }
  return counts;
}

function majority(counts) {
  let best = null;
  let bestCount = -1;
  for (const key of Object.keys(counts)) {
    if (counts[key] > bestCount) {
      bestCount = counts[key];
      best = key;
    }
  }
  return best;
}

function buildTree(rows, labels, indices, featureCount, maxFeatures, depth, maxDepth, minSamples, random) {
  const counts = classCounts(rows, labels, indices);
  const total = indices.length;
  const label = majority(counts);

  if (depth >= maxDepth || total < minSamples || Object.keys(counts).length <= 1) {
    return { leaf: true, label, counts, total };
  }

  const parentImpurity = gini(counts, total);
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;

  // Feature subsampling — the defining trait of a Random Forest
  const featureIdx = [];
  for (let f = 0; f < featureCount; f += 1) featureIdx.push(f);
  for (let i = featureIdx.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [featureIdx[i], featureIdx[j]] = [featureIdx[j], featureIdx[i]];
  }
  const candidates = featureIdx.slice(0, maxFeatures);

  for (const feature of candidates) {
    const values = indices.map((i) => rows[i][feature]).sort((a, b) => a - b);
    const seen = new Set();
    const splitPoints = [];
    const stride = Math.max(1, Math.floor(values.length / 12));
    for (let i = stride; i < values.length; i += stride) {
      const t = (values[i - 1] + values[i]) / 2;
      if (!seen.has(t)) {
        seen.add(t);
        splitPoints.push(t);
      }
    }

    for (const threshold of splitPoints) {
      const left = [];
      const right = [];
      for (const i of indices) {
        if (rows[i][feature] <= threshold) left.push(i);
        else right.push(i);
      }
      if (!left.length || !right.length) continue;
      const leftImpurity = gini(classCounts(rows, labels, left), left.length);
      const rightImpurity = gini(classCounts(rows, labels, right), right.length);
      const weighted = (left.length / total) * leftImpurity + (right.length / total) * rightImpurity;
      const gain = parentImpurity - weighted;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = feature;
        bestThreshold = threshold;
      }
    }
  }

  if (bestFeature < 0 || bestGain <= 1e-7) {
    return { leaf: true, label, counts, total };
  }

  const left = [];
  const right = [];
  for (const i of indices) {
    if (rows[i][bestFeature] <= bestThreshold) left.push(i);
    else right.push(i);
  }

  return {
    leaf: false,
    feature: bestFeature,
    threshold: bestThreshold,
    gain: bestGain,
    left: buildTree(rows, labels, left, featureCount, maxFeatures, depth + 1, maxDepth, minSamples, random),
    right: buildTree(rows, labels, right, featureCount, maxFeatures, depth + 1, maxDepth, minSamples, random),
  };
}

function predictTree(node, row) {
  let current = node;
  while (!current.leaf) {
    current = row[current.feature] <= current.threshold ? current.left : current.right;
  }
  return current;
}

function accumulateImportance(node, importances) {
  if (!node || node.leaf) return;
  importances[node.feature] = (importances[node.feature] || 0) + node.gain;
  accumulateImportance(node.left, importances);
  accumulateImportance(node.right, importances);
}

export class RandomForestClassifier {
  constructor({ trees = 24, maxDepth = 7, minSamples = 6, seed = 42, maxFeatures = null } = {}) {
    this.config = { trees, maxDepth, minSamples, seed, maxFeatures };
    this.trees = [];
    this.featureCount = 0;
    this.oobAccuracy = null;
    this.importances = [];
  }

  fit(rows, labels) {
    if (!rows.length) return this;
    this.featureCount = rows[0].length;
    const maxFeatures = this.config.maxFeatures || Math.max(1, Math.round(Math.sqrt(this.featureCount)));
    const random = mulberry32(this.config.seed);
    this.trees = [];
    const oobVotes = rows.map(() => ({}));

    for (let t = 0; t < this.config.trees; t += 1) {
      const bag = [];
      const inBag = new Set();
      for (let i = 0; i < rows.length; i += 1) {
        const pick = Math.floor(random() * rows.length);
        bag.push(pick);
        inBag.add(pick);
      }
      const tree = buildTree(
        rows,
        labels,
        bag,
        this.featureCount,
        maxFeatures,
        0,
        this.config.maxDepth,
        this.config.minSamples,
        random,
      );
      this.trees.push(tree);

      // Out-of-bag scoring gives an honest accuracy estimate
      for (let i = 0; i < rows.length; i += 1) {
        if (inBag.has(i)) continue;
        const leaf = predictTree(tree, rows[i]);
        oobVotes[i][leaf.label] = (oobVotes[i][leaf.label] || 0) + 1;
      }
    }

    let scored = 0;
    let correct = 0;
    for (let i = 0; i < rows.length; i += 1) {
      const votes = oobVotes[i];
      if (!Object.keys(votes).length) continue;
      scored += 1;
      if (majority(votes) === labels[i]) correct += 1;
    }
    this.oobAccuracy = scored ? correct / scored : null;

    const importances = {};
    for (const tree of this.trees) accumulateImportance(tree, importances);
    const totalGain = Object.values(importances).reduce((sum, value) => sum + value, 0) || 1;
    this.importances = Array.from({ length: this.featureCount }, (_, i) => (importances[i] || 0) / totalGain);

    return this;
  }

  predictWithConfidence(row) {
    if (!this.trees.length) return { label: null, confidence: 0, votes: {} };
    const votes = {};
    for (const tree of this.trees) {
      const leaf = predictTree(tree, row);
      // Weight by leaf purity so confident leaves count more
      const purity = leaf.counts[leaf.label] / Math.max(1, leaf.total);
      votes[leaf.label] = (votes[leaf.label] || 0) + purity;
    }
    const label = majority(votes);
    const total = Object.values(votes).reduce((sum, value) => sum + value, 0) || 1;
    return { label, confidence: votes[label] / total, votes };
  }

  predict(row) {
    return this.predictWithConfidence(row).label;
  }
}
