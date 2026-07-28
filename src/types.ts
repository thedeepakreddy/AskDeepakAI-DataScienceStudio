/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface DatasetColumn {
  name: string;
  type: 'numeric' | 'categorical' | 'boolean' | 'datetime';
  missingCount: number;
  distinctCount: number;
  statistics: {
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    stdDev?: number;
    mostCommon?: { value: string; count: number }[];
  };
}

export interface Dataset {
  filename: string;
  rows: Record<string, any>[];
  columns: DatasetColumn[];
  rowCount: number;
  originalRowCount: number;
}

export interface CleaningOperation {
  id: string;
  type: 'drop_column' | 'fill_missing' | 'type_convert' | 'filter_rows';
  column: string;
  params: Record<string, any>;
}

export interface MLConfig {
  targetColumn: string;
  featureColumns: string[];
  modelType: 'classification' | 'regression' | 'timeseries';
  modelAlgorithm: string;
  trainRatio: number; // e.g., 0.8 for 80/20 train/test split
  hyperparameters: Record<string, string | number | boolean>;
}

export interface MLResult {
  modelType: 'classification' | 'regression' | 'timeseries';
  modelAlgorithm: string;
  modelId?: string; // real trained-model id in the Python service, used for /predict and /download
  hyperparameters: Record<string, any>;
  metrics: {
    accuracy?: number; // classification
    precision?: number;
    recall?: number;
    f1Score?: number;
    rocAuc?: number; // real ROC-AUC, binary classification only
    r2Score?: number; // regression
    mse?: number;
    mae?: number;
    rmse?: number;
    mape?: number; // timeseries
  };
  featureImportance: { feature: string; score: number }[];
  tuningHistory: { iteration: number; score: number; params: string }[];
  predictions: {
    id: number;
    actual: number | string;
    predicted: number | string;
    residual?: number;
    confidence?: number | null; // real predicted-class probability (%) for classification, from the trained model
    features: Record<string, any>;
  }[];
  modelReportMarkdown: string;
  markdownReport?: string;
  risks?: { title: string; riskLevel: 'High' | 'Medium' | 'Low'; description: string }[];
  recommendations?: { title: string; impact: 'High' | 'Medium' | 'Low'; details: string }[];
  scientistCallout?: {
    focusColumns: string[];
    justification: string;
    pathways: string[];
  };

  // --- Real, server-computed fields (populated by the Python training service) ---
  confusionMatrix?:
    | { tp: number; tn: number; fp: number; fn: number; positiveLabel?: string }
    | { matrix: number[][]; labels: string[] };
  cv?: {
    scores: number[];
    mean: number;
    std: number;
    folds: number;
    metric: string;
    error?: string;
  };
  comparison?: {
    modelKey: string;
    modelName: string;
    primaryMetric: string;
    metricValue: number;
    executionTimeMs: number;
    cv?: { scores: number[]; mean: number; std: number; folds: number; metric: string; error?: string } | null;
  }[];
  // Explainable AutoML selection rule, spelled out in plain language with the
  // real numbers plugged in (never hidden from the UI).
  selectionReason?: string;
  // Real SHAP feature importance for the champion model (mean |SHAP value|
  // aggregated back to original columns). Absent/undefined if SHAP couldn't
  // explain this particular model — never fabricated as a substitute.
  shapImportance?: { feature: string; score: number }[] | null;
  estimators?: {
    id: number;
    name: string;
    splitFeature: string;
    splitValue: number;
    sampleCount: number;
    impurity: number;
    treeDepth: number;
    leafCount: number;
  }[];
  oobScore?: number | null; // real Random Forest out-of-bag score, RF only
  deepLearning?: {
    trainingLogs: { epoch: number; trainingLoss: number }[];
    validationScores?: number[] | null; // real per-epoch validation accuracy/R2 (early_stopping=True)
    validationMetric?: 'accuracy' | 'r2';
    finalValidationScore?: number | null;
    hiddenLayerSizes: number[];
    nLayers: number;
    nIterations: number;
    totalTrainableParams: number;
  } | null;
  trainRows?: number;
  testRows?: number;
  trainRatio?: number;
}

export interface StakeholderInsightReport {
  summary: string;
  potentialRisks: { title: string; riskLevel: 'High' | 'Medium' | 'Low'; description: string }[];
  strategicRecommendations: { title: string; impact: 'High' | 'Medium' | 'Low'; details: string }[];
  scientistCallout: {
    focusColumns: string[];
    justification: string;
    potentialAnalysisPathways: string[];
  };
}

export interface DashboardFilterState {
  slicers: Record<string, string[]>; // column -> list of selected categorical values
  rangeFilters: Record<string, { min: number; max: number; currentMin: number; currentMax: number }>; // numeric filter ranges
}

// Real, server-computed EDA report from the Python service's /eda endpoint —
// every number here is genuine pandas/numpy computation, not narrated.
export interface EdaReport {
  rowCount: number;
  columnCount: number;
  numericSummaries: {
    column: string;
    count: number;
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    q1: number;
    q3: number;
    skew: number;
  }[];
  outliers: {
    column: string;
    q1: number;
    q3: number;
    iqr: number;
    lowerBound: number;
    upperBound: number;
    outlierCount: number;
    outlierPercent: number;
    sampleOutliers: { rowIndex: number; value: number }[];
  }[];
  correlation: { columns: string[]; matrix: (number | null)[][] } | null;
  missingReport: { column: string; missingCount: number; missingPercent: number }[];
  categoricalSummaries: {
    column: string;
    distinctCount: number;
    topValues: { value: string; count: number; percent: number }[];
  }[];
  // Gemini narrative (or honest template fallback) built strictly from the
  // real numbers above — no numeric fields in this object, by design.
  narrative?: {
    summary: string;
    dataQualityFlags: { column: string; issue: string; severity: 'High' | 'Medium' | 'Low' }[];
    recommendedNextSteps: string[];
  };
}

// Phase 2: agentic analysis mode. Gemini proposes each step via real
// function-calling (never free-text parsing); every `result` field below is
// the genuine response from the corresponding mlops_service /agent/* call —
// `reasoning` is the only LLM-authored text in a step, and it is never used
// as the source of a number shown elsewhere.
export interface AgentStep {
  step: number;
  tool?: 'retrain_with_transform' | 'drop_feature' | 'check_multicollinearity' | 'declare_done';
  reasoning?: string;
  args?: Record<string, any>;
  result?: {
    modelKey?: string;
    modelName?: string;
    modelId?: string;
    primaryMetric?: string;
    primaryMetricValue?: number;
    transformApplied?: { column: string; method: string };
    targetWasTransformed?: boolean;
    droppedFeature?: string;
    remainingFeatures?: string[];
    vifScores?: { feature: string; vif: number | null; rSquared: number; highRisk: boolean }[];
    highRiskFeatures?: string[];
  };
  metricBefore?: number;
  metricAfter?: number;
  improved?: boolean;
  error?: string;
}

export interface AgentRunResult {
  steps: AgentStep[];
  startingMetricValue: number;
  finalPrimaryMetricValue: number;
  finalModelId: string | null;
  stopReason: 'cap_reached' | 'plateau' | 'agent_declared_done' | 'no_action_proposed' | 'error';
  maxSteps: number;
  whitelistedFunctions: string[];
}
