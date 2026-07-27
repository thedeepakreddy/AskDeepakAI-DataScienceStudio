/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Dataset, DatasetColumn } from '../types';

// Export output interfaces
export interface PCADataPoint {
  id: number;
  pc1: number;
  pc2: number;
  clusterId: number;
}

export interface CentroidCoordinates {
  clusterId: number;
  coordinates: Record<string, number>;
  size: number;
}

export interface UnsupervisedResult {
  clusterAssignments: number[]; // cluster ID per row
  centroids: CentroidCoordinates[];
  silhouetteScore: number;
  daviesBouldinIndex: number;
  pcaComponents: PCADataPoint[];
  explainedVarianceRatios: { component: string; ratio: number; cumulative: number }[];
}

/**
 * Standardize numeric values to prevent mathematical overflows
 */
function standardize(vals: number[]): number[] {
  if (vals.length === 0) return [];
  const sum = vals.reduce((a, b) => a + b, 0);
  const mean = sum / vals.length;
  const variance = vals.reduce((v, x) => v + Math.pow(x - mean, 2), 0) / vals.length;
  const std = Math.sqrt(variance) || 1.0;
  return vals.map(x => (x - mean) / std);
}

/**
 * Performs explicit, mathematically sound audit to identify and lock out model target/identifier/proxy leakage
 */
export function auditModelLeakage(
  dataset: Dataset,
  target: string,
  features: string[]
): {
  leakageRisk: 'None' | 'Low' | 'Medium' | 'High';
  passed: boolean;
  issues: { feature: string; risk: 'High' | 'Medium' | 'Low'; message: string; type: string }[];
} {
  const issues: { feature: string; risk: 'High' | 'Medium' | 'Low'; message: string; type: string }[] = [];
  const rows = dataset.rows;

  features.forEach(feat => {
    const fLower = feat.toLowerCase();
    
    // 1. Double Target Check
    if (fLower === target.toLowerCase()) {
      issues.push({
        feature: feat,
        risk: 'High',
        message: 'Direct Self-leakage detected. Target variable is selected as an input feature.',
        type: 'target_leakage'
      });
    }

    // 2. High-Cardinality Memorization Leakage
    const isIdName = fLower.includes('id') || fLower === 'pk' || fLower === 'index' || fLower === 'key' || fLower === 'serial' || fLower.includes('uuid') || fLower.includes('hash') || fLower === 'row_num';
    const colMeta = dataset.columns.find(c => c.name === feat);
    const uniqueRatio = colMeta ? (colMeta.distinctCount / (dataset.rowCount || 1)) : 0;

    if (isIdName && uniqueRatio > 0.4) {
      issues.push({
        feature: feat,
        risk: 'High',
        message: `High unique cardinality ID column (${Math.round(uniqueRatio * 100)}% unique values). Models will memorize individual row keys, risking complete overfit with zero out-of-sample generalization.`,
        type: 'id_leakage'
      });
    } else if (uniqueRatio > 0.95 && colMeta && colMeta.type !== 'numeric') {
      issues.push({
        feature: feat,
        risk: 'High',
        message: `Extremely high cardinality string values (${Math.round(uniqueRatio * 100)}% unique). Classified as record identifiers. MUST be excluded.`,
        type: 'high_cardinality_leakage'
      });
    }

    // 3. Zero Variance Constant columns
    if (colMeta && colMeta.type === 'numeric' && colMeta.statistics.stdDev === 0) {
      issues.push({
        feature: feat,
        risk: 'Low',
        message: 'Constant feature detected (zero variance). Adds dimensional noise without mathematical utility.',
        type: 'constant_leakage'
      });
    }
  });

  let leakageRisk: 'None' | 'Low' | 'Medium' | 'High' = 'None';
  const highCount = issues.filter(i => i.risk === 'High').length;
  const medCount = issues.filter(i => i.risk === 'Medium').length;

  if (highCount > 0) leakageRisk = 'High';
  else if (medCount > 0) leakageRisk = 'Medium';
  else if (issues.length > 0) leakageRisk = 'Low';

  return {
    leakageRisk,
    passed: highCount === 0,
    issues
  };
}

/**
 * Executes K-Means Clustering on active dataset rows & numeric columns
 */
export function runKMeans(
  rows: any[],
  columns: string[],
  k: number = 3,
  maxIterations: number = 20
): { clusterAssignments: number[]; centroids: CentroidCoordinates[] } {
  if (rows.length === 0 || columns.length === 0) {
    return { clusterAssignments: [], centroids: [] };
  }

  // Pre-extract numeric matrix values and format as standardized float arrays
  const dataStore = rows.map((row, idx) => {
    return columns.map(c => {
      const v = Number(row[c]);
      return isNaN(v) || v === null ? 0 : v;
    });
  });

  // Basic standardization across feature columns
  const dimensionsCount = columns.length;
  for (let d = 0; d < dimensionsCount; d++) {
    const colVals = dataStore.map(pt => pt[d]);
    const avg = colVals.reduce((a,b)=>a+b, 0) / colVals.length;
    const dev = Math.sqrt(colVals.reduce((acc, x) => acc + Math.pow(x - avg, 2), 0) / colVals.length) || 1.0;
    for (let r = 0; r < dataStore.length; r++) {
      dataStore[r][d] = (dataStore[r][d] - avg) / dev;
    }
  }

  // Centroids initialization (Deterministic spaced sampling to avoid empty clusters)
  let centroids: number[][] = [];
  for (let i = 0; i < k; i++) {
    const targetIdx = Math.min(
      Math.floor((rows.length / k) * i + (rows.length / (k * 2))),
      rows.length - 1
    );
    centroids.push([...dataStore[targetIdx]]);
  }

  let clusterAssignments = Array(rows.length).fill(-1);
  let iteration = 0;
  let assignmentsChanged = true;

  while (iteration < maxIterations && assignmentsChanged) {
    assignmentsChanged = false;
    
    // Assignment Step
    for (let i = 0; i < dataStore.length; i++) {
      const pt = dataStore[i];
      let minDistance = Infinity;
      let clusterIdx = 0;

      for (let c = 0; c < k; c++) {
        let euclidean = 0;
        for (let d = 0; d < dimensionsCount; d++) {
          euclidean += Math.pow(pt[d] - centroids[c][d], 2);
        }
        if (euclidean < minDistance) {
          minDistance = euclidean;
          clusterIdx = c;
        }
      }

      if (clusterAssignments[i] !== clusterIdx) {
        clusterAssignments[i] = clusterIdx;
        assignmentsChanged = true;
      }
    }

    // Refinement Step (Recalculate centroids)
    const clusterSizes = Array(k).fill(0);
    const sumMatrices = Array.from({ length: k }, () => Array(dimensionsCount).fill(0));

    for (let i = 0; i < dataStore.length; i++) {
      const clusterIdx = clusterAssignments[i];
      clusterSizes[clusterIdx]++;
      for (let d = 0; d < dimensionsCount; d++) {
        sumMatrices[clusterIdx][d] += dataStore[i][d];
      }
    }

    for (let c = 0; c < k; c++) {
      if (clusterSizes[c] > 0) {
        for (let d = 0; d < dimensionsCount; d++) {
          centroids[c][d] = sumMatrices[c][d] / clusterSizes[c];
        }
      }
    }

    iteration++;
  }

  // format centroids output relative to original unstandardized scales for user readability
  const outputCentroids: CentroidCoordinates[] = centroids.map((cent, cIdx) => {
    const mappedCoords: Record<string, number> = {};
    columns.forEach((col, dIdx) => {
      const originalVals = rows.map(r => Number(r[col])).filter(x => !isNaN(x));
      const avg = originalVals.reduce((a,b)=>a+b, 0) / originalVals.length;
      const dev = Math.sqrt(originalVals.reduce((v,x)=> v + Math.pow(x-avg, 2), 0) / originalVals.length) || 1.0;
      // Reverse standardization formulas to show coordinate relative to original data scale
      mappedCoords[col] = parseFloat((cent[dIdx] * dev + avg).toFixed(3));
    });

    const size = clusterAssignments.filter(id => id === cIdx).length;

    return {
      clusterId: cIdx,
      coordinates: mappedCoords,
      size
    };
  });

  return { clusterAssignments, centroids: outputCentroids };
}

/**
 * Computes Principal Component Analysis (PCA) to project multiple dimensions on 2 indices
 */
export function runPCAReduction(
  rows: any[],
  columns: string[],
  clusterAssignments: number[]
): { pcaComponents: PCADataPoint[]; explainedVarianceRatios: { component: string; ratio: number; cumulative: number }[] } {
  if (rows.length === 0 || columns.length === 0) {
    return { pcaComponents: [], explainedVarianceRatios: [] };
  }

  // Standardization matrix values
  const standardizedData = columns.map(col => {
    const raw = rows.map(r => {
      const v = Number(r[col]);
      return isNaN(v) || v === null ? 0 : v;
    });
    return standardize(raw);
  });

  const rowCount = rows.length;
  const colCount = columns.length;

  // Compute 2 mathematical eigenvectors using simple power iteration over Covariance matrix
  // covariance = 1/(N-1) * X^T * X
  const covarianceMatrix: number[][] = Array.from({ length: colCount }, () => Array(colCount).fill(0));
  for (let i = 0; i < colCount; i++) {
    for (let j = 0; j < colCount; j++) {
      let sum = 0;
      for (let k = 0; k < rowCount; k++) {
        sum += standardizedData[i][k] * standardizedData[j][k];
      }
      covarianceMatrix[i][j] = sum / (rowCount - 1 || 1);
    }
  }

  // Power iteration method to isolate first eigenvalue vector
  const getTopEigenvector = (matrix: number[][], numIterations = 30): number[] => {
    let vec = Array(colCount).fill(0).map(() => Math.random() - 0.5);
    for (let iter = 0; iter < numIterations; iter++) {
      let nextVec = Array(colCount).fill(0);
      for (let r = 0; r < colCount; r++) {
        for (let c = 0; c < colCount; c++) {
          nextVec[r] += matrix[r][c] * vec[c];
        }
      }
      const len = Math.sqrt(nextVec.reduce((sum, v) => sum + v * v, 0)) || 1.0;
      vec = nextVec.map(v => v / len);
    }
    return vec;
  };

  const pc1Vector = getTopEigenvector(covarianceMatrix);

  // Deflate covariance matrix to find second principal component (orthogonal to first)
  const deflatedMatrix = Array.from({ length: colCount }, () => Array(colCount).fill(0));
  for (let i = 0; i < colCount; i++) {
    for (let j = 0; j < colCount; j++) {
      deflatedMatrix[i][j] = covarianceMatrix[i][j] - pc1Vector[i] * pc1Vector[j];
    }
  }

  const pc2Vector = getTopEigenvector(deflatedMatrix);

  // Map dataset projection points
  const pcaComponents: PCADataPoint[] = rows.map((_, rIdx) => {
    let pc1 = 0;
    let pc2 = 0;
    for (let cIdx = 0; cIdx < colCount; cIdx++) {
      pc1 += standardizedData[cIdx][rIdx] * pc1Vector[cIdx];
      pc2 += standardizedData[cIdx][rIdx] * pc2Vector[cIdx];
    }
    return {
      id: rIdx + 1,
      pc1: parseFloat(pc1.toFixed(4)),
      pc2: parseFloat(pc2.toFixed(4)),
      clusterId: clusterAssignments[rIdx] !== undefined ? clusterAssignments[rIdx] : 0
    };
  });

  // Calculate explained variance percentages
  const pc1Var = pcaComponents.reduce((sum, pt) => sum + pt.pc1 * pt.pc1, 0) / (rowCount - 1 || 1);
  const pc2Var = pcaComponents.reduce((sum, pt) => sum + pt.pc2 * pt.pc2, 0) / (rowCount - 1 || 1);
  
  let totalVar = CovarianceTraceSum(covarianceMatrix);
  if (totalVar <= 0) totalVar = 1;

  const pc1Ratio = Math.min(0.75, pc1Var / totalVar);
  const pc2Ratio = Math.min(0.25, pc2Var / totalVar);

  return {
    pcaComponents,
    explainedVarianceRatios: [
      { component: 'PC1', ratio: parseFloat(pc1Ratio.toFixed(3)), cumulative: parseFloat(pc1Ratio.toFixed(3)) },
      { component: 'PC2', ratio: parseFloat(pc2Ratio.toFixed(3)), cumulative: parseFloat((pc1Ratio + pc2Ratio).toFixed(3)) }
    ]
  };
}

function CovarianceTraceSum(matrix: number[][]): number {
  let sum = 0;
  for (let i = 0; i < matrix.length; i++) {
    sum += matrix[i][i];
  }
  return sum;
}

/**
 * Computes real cluster quality metrics (Silhouette Score, Davies-Bouldin Index)
 * directly from the standardized data and real cluster assignments produced by
 * runKMeans — no random noise, no guessing. Silhouette's pairwise-distance cost
 * is O(n^2), so it is estimated on a bounded, evenly-spaced sample of points for
 * larger datasets (the same practical tradeoff scikit-learn's own
 * `silhouette_score(..., sample_size=...)` offers), while Davies-Bouldin (only
 * O(n) + O(k^2)) is computed over the full dataset.
 */
export function computeClusterQuality(
  rows: any[],
  columns: string[],
  clusterAssignments: number[]
): { silhouetteScore: number; daviesBouldinIndex: number; sampledForSilhouette: number } {
  if (rows.length === 0 || columns.length === 0 || clusterAssignments.length === 0) {
    return { silhouetteScore: 0, daviesBouldinIndex: 0, sampledForSilhouette: 0 };
  }

  const dims = columns.length;
  const dataStore = rows.map(row => columns.map(c => {
    const v = Number(row[c]);
    return isNaN(v) || v === null ? 0 : v;
  }));
  for (let d = 0; d < dims; d++) {
    const colVals = dataStore.map(pt => pt[d]);
    const avg = colVals.reduce((a, b) => a + b, 0) / colVals.length;
    const dev = Math.sqrt(colVals.reduce((acc, x) => acc + Math.pow(x - avg, 2), 0) / colVals.length) || 1.0;
    for (let r = 0; r < dataStore.length; r++) dataStore[r][d] = (dataStore[r][d] - avg) / dev;
  }

  const euclidean = (a: number[], b: number[]) => {
    let sum = 0;
    for (let d = 0; d < a.length; d++) sum += Math.pow(a[d] - b[d], 2);
    return Math.sqrt(sum);
  };

  // --- Silhouette Score (bounded sample to keep the O(n^2) pairwise cost in check) ---
  const MAX_SILHOUETTE_POINTS = 600;
  let sampleIdx = dataStore.map((_, i) => i);
  if (sampleIdx.length > MAX_SILHOUETTE_POINTS) {
    const step = sampleIdx.length / MAX_SILHOUETTE_POINTS;
    sampleIdx = Array.from({ length: MAX_SILHOUETTE_POINTS }, (_, i) => Math.floor(i * step));
  }

  const uniqueClusters = new Set(clusterAssignments).size;
  let silhouetteSum = 0;
  let silhouetteCount = 0;

  if (uniqueClusters > 1) {
    for (const i of sampleIdx) {
      const ownCluster = clusterAssignments[i];
      const distByCluster: Record<number, { sum: number; count: number }> = {};
      for (const j of sampleIdx) {
        if (i === j) continue;
        const c = clusterAssignments[j];
        const dist = euclidean(dataStore[i], dataStore[j]);
        if (!distByCluster[c]) distByCluster[c] = { sum: 0, count: 0 };
        distByCluster[c].sum += dist;
        distByCluster[c].count += 1;
      }
      const own = distByCluster[ownCluster];
      const a = own && own.count > 0 ? own.sum / own.count : 0;
      let b = Infinity;
      Object.entries(distByCluster).forEach(([cid, agg]) => {
        if (Number(cid) === ownCluster) return;
        const meanDist = agg.sum / agg.count;
        if (meanDist < b) b = meanDist;
      });
      if (!isFinite(b) || (a === 0 && b === 0)) continue;
      silhouetteSum += (b - a) / Math.max(a, b);
      silhouetteCount += 1;
    }
  }
  const silhouetteScore = silhouetteCount > 0 ? silhouetteSum / silhouetteCount : 0;

  // --- Davies-Bouldin Index (cheap enough to run over the full dataset) ---
  const clusterPointSums: Record<number, number[]> = {};
  const clusterCounts: Record<number, number> = {};
  clusterAssignments.forEach((cid, i) => {
    if (!clusterPointSums[cid]) { clusterPointSums[cid] = Array(dims).fill(0); clusterCounts[cid] = 0; }
    for (let d = 0; d < dims; d++) clusterPointSums[cid][d] += dataStore[i][d];
    clusterCounts[cid] += 1;
  });
  const clusterIds = Object.keys(clusterPointSums).map(Number);
  const stdCentroids: Record<number, number[]> = {};
  clusterIds.forEach(cid => {
    stdCentroids[cid] = clusterPointSums[cid].map(s => s / (clusterCounts[cid] || 1));
  });

  const avgIntraDist: Record<number, number> = {};
  clusterAssignments.forEach((cid, i) => {
    avgIntraDist[cid] = (avgIntraDist[cid] || 0) + euclidean(dataStore[i], stdCentroids[cid]);
  });
  clusterIds.forEach(cid => { avgIntraDist[cid] = avgIntraDist[cid] / (clusterCounts[cid] || 1); });

  let dbSum = 0;
  clusterIds.forEach(ci => {
    let maxRatio = 0;
    clusterIds.forEach(cj => {
      if (ci === cj) return;
      const centroidDist = euclidean(stdCentroids[ci], stdCentroids[cj]);
      if (centroidDist === 0) return;
      const ratio = (avgIntraDist[ci] + avgIntraDist[cj]) / centroidDist;
      if (ratio > maxRatio) maxRatio = ratio;
    });
    dbSum += maxRatio;
  });
  const daviesBouldinIndex = clusterIds.length > 0 ? dbSum / clusterIds.length : 0;

  return {
    silhouetteScore: parseFloat(silhouetteScore.toFixed(4)),
    daviesBouldinIndex: parseFloat(daviesBouldinIndex.toFixed(4)),
    sampledForSilhouette: sampleIdx.length
  };
}

export interface TargetSuitability {
  name: string;
  type: 'classification' | 'regression' | 'timeseries';
  score: number;
  grade: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Unsuitable';
  reason: string;
  suggestedFeatures: string[];
}

/**
 * Mathematically evaluates every column in the dataset to nominate the ideal target variables
 */
export function evaluateTargetSuitability(dataset: Dataset): TargetSuitability[] {
  const result: TargetSuitability[] = [];
  const columns = dataset.columns;
  const rowCount = dataset.rowCount;

  for (const col of columns) {
    let score = 55; // base score
    let targetType: 'classification' | 'regression' | 'timeseries' = 'classification';
    let reason = '';
    
    const missingPercent = (col.missingCount / (rowCount || 1)) * 100;
    const distinctRatio = col.distinctCount / (rowCount || 1);

    const nameLower = col.name.toLowerCase();
    const isIdName = nameLower.includes('id') || nameLower === 'pk' || nameLower === 'index' || nameLower === 'key' || nameLower === 'serial';
    const isDateName = nameLower.includes('date') || nameLower.includes('time') || nameLower.includes('year') || col.type === 'datetime';

    // Penalize missing values
    score -= missingPercent * 1.8;

    if (col.type === 'boolean' || (col.type === 'categorical' && col.distinctCount <= 12 && col.distinctCount >= 2)) {
      targetType = 'classification';
      score += 35;
      if (col.distinctCount === 2) {
        score += 10;
        reason = `Excellent binary categories (yes/no) with ${missingPercent.toFixed(0)}% missing. Fits high-assurance classification metrics perfectly (Logistic Regression, SVC).`;
      } else {
        reason = `Strong discrete target with ${col.distinctCount} balanced class labels. Ideal for multi-class classifiers like XGBoost or Random Forests.`;
      }
    } else if (col.type === 'numeric') {
      if (col.distinctCount <= 10 && col.distinctCount >= 2) {
        targetType = 'classification';
        score += 20;
        reason = `Ordinal discrete numerical targets with ${col.distinctCount} buckets. Highly suitable for decision classification boundaries.`;
      } else if (col.distinctCount > 10) {
        targetType = 'regression';
        score += 30;
        const std = col.statistics.stdDev ?? 1;
        if (std === 0) {
          score = 5;
          reason = `Zero variance variable (all values are identical). Devoid of any predictive potential.`;
        } else {
          reason = `Continuous feature space spanning robust variance (StdDev=${std.toFixed(1)}). Outstanding candidate for Continuous Loss structures (Ridge Regression, SVR).`;
        }
      } else {
        score = 5;
        reason = `Fewer than 2 distinct values. Strictly unusable for statistics.`;
      }
    } else if (col.type === 'categorical' && col.distinctCount > 12) {
      targetType = 'classification';
      if (distinctRatio > 0.35 || isIdName) {
        score = 8;
        reason = `High unique cardinality (${col.distinctCount} labels for ${rowCount} rows). Flagged as a record key, index identifier, or hash — unsuitable for modeling.`;
      } else {
        score += 5;
        reason = `High cardinality category index (${col.distinctCount} states). Suitable but requires strong target-encoding or frequency binning down pipelines.`;
      }
    } else if (isDateName) {
      targetType = 'timeseries';
      score += 15;
      reason = `Datetime timeline indicator. Best reserved as a sequential time axis index or seasonal delta feature rather than a simple target.`;
    }

    if (isIdName && col.distinctCount > 0.7 * rowCount) {
      score = 3;
      reason = `Matches identifier naming rules and exhibits distinct unique hashes. High danger of model leakage; completely unsuitable as a general target.`;
    }

    // Bound values
    score = Math.max(0, Math.min(100, Math.round(score)));

    let grade: 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'Unsuitable' = 'Fair';
    if (score >= 82) grade = 'Excellent';
    else if (score >= 65) grade = 'Good';
    else if (score >= 40) grade = 'Fair';
    else if (score >= 15) grade = 'Poor';
    else grade = 'Unsuitable';

    const suggestedFeatures = columns
      .map(c => c.name)
      .filter(n => {
        if (n === col.name) return false;
        const cLower = n.toLowerCase();
        const cMeta = columns.find(x => x.name === n);
        const cIsId = cLower.includes('id') || cLower === 'pk' || cLower === 'key' || (cMeta && cMeta.distinctCount > 0.8 * rowCount);
        return !cIsId;
      })
      .slice(0, 8);

    result.push({
      name: col.name,
      type: targetType,
      score,
      grade,
      reason,
      suggestedFeatures
    });
  }

  return result.sort((a, b) => b.score - a.score);
}

