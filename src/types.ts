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
  }[];
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
