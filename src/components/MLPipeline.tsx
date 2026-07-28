/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import {
  Play,
  Settings,
  Cpu,
  LineChart,
  BarChart2,
  CheckCircle,
  Terminal,
  Sparkles,
  Download,
  Table,
  ChevronRight,
  Info,
  Workflow,
  Brain,
  GitBranch,
  Database,
  BrainCircuit,
  ShieldCheck,
  ShieldAlert,
  Trophy,
  Loader2,
  Bot
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Legend,
  Line,
  LineChart as RechartsLineChart
} from 'recharts';
import { Dataset, MLResult } from '../types';
import {
  evaluateTargetSuitability,
  auditModelLeakage,
  runKMeans,
  runPCAReduction,
  computeClusterQuality,
  UnsupervisedResult
} from '../utils/mlEngine';
import ModelExplainer from './ModelExplainer';
import MLOpsDashboard from './MLOpsDashboard';
import AgentAnalysis from './AgentAnalysis';
import { usePipelineContext } from '../contexts/PipelineContext';

interface MLPipelineProps {
  dataset: Dataset;
  aiSuggestedTarget?: string;
  aiSuggestedType?: string;
  aiSuggestedFeatures?: string[];
  onTriggerPrediction: (
    target: string,
    features: string[],
    modelType: 'classification' | 'regression' | 'timeseries',
    hyperparameters: Record<string, any>
  ) => Promise<MLResult>;
  mlResult: MLResult | null;
  loadingML: boolean;
}

const REAL_ALGORITHM_OPTIONS = ['auto', 'linear', 'random_forest', 'xgboost', 'mlp'];

export default function MLPipeline({
  dataset,
  aiSuggestedTarget = '',
  aiSuggestedType = 'classification',
  aiSuggestedFeatures = [],
  onTriggerPrediction,
  mlResult,
  loadingML
}: MLPipelineProps) {
  const { expertMode } = usePipelineContext();
  const [target, setTarget] = useState(aiSuggestedTarget || dataset.columns[dataset.columns.length - 1]?.name || '');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [modelType, setModelType] = useState<'classification' | 'regression' | 'timeseries'>('classification');

  // Real model selector — trimmed to algorithms this app can genuinely train
  // (scikit-learn Linear/Logistic Regression, Random Forest, XGBoost, MLP).
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState<string>('auto');

  // Pipeline Tuning regulators — every one of these is sent to and actually
  // used by the real scikit-learn/XGBoost training call.
  const [estimators, setEstimators] = useState<number>(100);
  const [maxDepth, setMaxDepth] = useState<number>(8);
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const [splitRatio, setSplitRatio] = useState<number>(0.8);
  const [clusterK, setClusterK] = useState<number>(3);
  const [mlpMaxIterations, setMlpMaxIterations] = useState<number>(200);

  const [progressLog, setProgressLog] = useState<string[]>([]);
  const [unsupervisedRes, setUnsupervisedRes] = useState<UnsupervisedResult | null>(null);
  const [downloadingModel, setDownloadingModel] = useState<boolean>(false);

  // Tabs management
  const [activeRealm, setActiveRealm] = useState<'supervised' | 'unsupervised' | 'ensemble' | 'deep_learning' | 'comparison' | 'agent'>('supervised');
  const [selectedEstimatorIdx, setSelectedEstimatorIdx] = useState<number>(0);
  const [selectedRawPredPage, setSelectedRawPredPage] = useState<number>(1);

  // Memoize targets suitability list so we don't restart on minor movements
  const targetRecommendations = React.useMemo(() => evaluateTargetSuitability(dataset), [dataset]);

  // Real-time frontend-side model leakage monitor to protect the modeling stage
  const leakageAuditResult = React.useMemo(() => {
    if (!target) return { leakageRisk: 'None' as const, passed: true, issues: [] };
    return auditModelLeakage(dataset, target, selectedFeatures);
  }, [dataset, target, selectedFeatures]);

  // Sync suggestion weights when analysis loads
  const aiSuggestedFeaturesStr = JSON.stringify(aiSuggestedFeatures || []);
  const datasetColumnsStr = dataset.columns.map(c => c.name).join(',');

  useEffect(() => {
    if (aiSuggestedTarget) {
      setTarget(aiSuggestedTarget);
    }
    if (aiSuggestedType) {
      setModelType(aiSuggestedType as any);
    }
    if (aiSuggestedFeatures && aiSuggestedFeatures.length > 0) {
      setSelectedFeatures(aiSuggestedFeatures.filter(f => dataset.columns.some(col => col.name === f)));
    } else {
      // Default to picking numeric columns excluding target
      const t = aiSuggestedTarget || dataset.columns[dataset.columns.length - 1]?.name;
      const initial = dataset.columns.map(c => c.name).filter(n => n !== t).slice(0, 5);
      setSelectedFeatures(initial);
    }
  }, [aiSuggestedTarget, aiSuggestedType, aiSuggestedFeaturesStr, datasetColumnsStr]);

  // Handle changing target - ensure target is removed from feature lists
  const handleTargetChange = (val: string) => {
    setTarget(val);
    const colMeta = dataset.columns.find(c => c.name === val);
    if (colMeta) {
      const isClass = colMeta.type === 'categorical' || colMeta.type === 'boolean' || colMeta.distinctCount < 10;
      setModelType(isClass ? 'classification' : 'regression');
    }
    setSelectedFeatures(current => current.filter(f => f !== val));
  };

  const handleSetTargetFromAdvisor = (targetName: string, suggestedType: 'classification' | 'regression' | 'timeseries', suggestedFeats: string[]) => {
    setTarget(targetName);
    setModelType(suggestedType);
    setSelectedFeatures(suggestedFeats.filter(f => f !== targetName && dataset.columns.some(col => col.name === f)));
    setProgressLog(prev => [...prev, `⚡ Automatically configured target context to "${targetName}" (${suggestedType.toUpperCase()}) with advised input dimensions.`]);
  };

  const toggleFeature = (name: string) => {
    if (selectedFeatures.includes(name)) {
      setSelectedFeatures(selectedFeatures.filter(f => f !== name));
    } else {
      setSelectedFeatures([...selectedFeatures, name]);
    }
  };

  const handleLaunchPipeline = async () => {
    setProgressLog([`Sending ${dataset.rowCount.toLocaleString()} rows to the Python ML compute service for real training...`]);

    // Real unsupervised analysis (K-Means + PCA + genuine cluster-quality metrics)
    // runs instantly client-side on the actual active dataset.
    const clusterCols = selectedFeatures.filter(f => {
      const meta = dataset.columns.find(c => c.name === f);
      return meta && meta.type === 'numeric';
    });
    const effectiveClusterCols = clusterCols.length > 1 ? clusterCols : selectedFeatures.slice(0, 3);
    const kMeansRes = runKMeans(dataset.rows, effectiveClusterCols, clusterK);
    const pcaRes = runPCAReduction(dataset.rows, effectiveClusterCols, kMeansRes.clusterAssignments);
    const quality = computeClusterQuality(dataset.rows, effectiveClusterCols, kMeansRes.clusterAssignments);
    setUnsupervisedRes({
      clusterAssignments: kMeansRes.clusterAssignments,
      centroids: kMeansRes.centroids,
      pcaComponents: pcaRes.pcaComponents,
      explainedVarianceRatios: pcaRes.explainedVarianceRatios,
      silhouetteScore: quality.silhouetteScore,
      daviesBouldinIndex: quality.daviesBouldinIndex
    });

    const params = {
      n_estimators: estimators,
      max_depth: maxDepth,
      learning_rate: learningRate,
      train_ratio: splitRatio,
      k: clusterK,
      epochs: mlpMaxIterations,
      selectedAlgorithmId
    };

    setProgressLog(prev => [...prev, `Training real scikit-learn/XGBoost model(s) on a genuine ${Math.round(splitRatio * 100)}/${Math.round((1 - splitRatio) * 100)} train/test split — this can take several seconds depending on dataset size...`]);

    try {
      await onTriggerPrediction(target, selectedFeatures, modelType, params);
      setProgressLog(prev => [...prev, `✓ Training complete. All metrics below were measured on held-out test data — nothing here is estimated.`]);
      setSelectedRawPredPage(1);
      setSelectedEstimatorIdx(0);
      setActiveRealm('supervised');
    } catch (err: any) {
      setProgressLog(prev => [...prev, `❌ ${err.message || 'Training failed — the ML compute service may be offline.'}`]);
    }
  };

  // Downloads the real, genuinely-fitted joblib artifact from the Python service.
  const downloadModelFile = async () => {
    if (!mlResult?.modelId) return;
    setDownloadingModel(true);
    try {
      const res = await fetch(`/api/ml/download/${encodeURIComponent(mlResult.modelId)}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || 'Download failed.');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${target}_${(mlResult.modelAlgorithm || 'model').replace(/\s+/g, '')}.joblib`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setProgressLog(prev => [...prev, `❌ Model download failed: ${err.message}`]);
    } finally {
      setDownloadingModel(false);
    }
  };

  // Pre-training: a column-structure heuristic (clearly labeled as such).
  // Post-training: the REAL champion model and its REAL measured metric.
  const getEDAModelFit = () => {
    if (mlResult) {
      const m = mlResult.metrics || {};
      const isClass = mlResult.modelType === 'classification';
      const primaryValue = isClass ? m.accuracy : m.r2Score;
      const comparedCount = mlResult.comparison?.length || 1;
      return {
        modelName: mlResult.modelAlgorithm,
        // The real, explainable AutoML selection rule from the server — CV
        // mean score with a stated tie-break — shown verbatim, never hidden.
        reasoning: mlResult.selectionReason || (comparedCount > 1
          ? `Selected as the top performer out of ${comparedCount} real models trained and evaluated on your held-out test split.`
          : `Trained and evaluated on your held-out test split using scikit-learn/XGBoost.`),
        primaryLabel: isClass ? 'Accuracy' : 'R² Score',
        primaryValue: primaryValue !== undefined ? primaryValue : null,
        isReal: true
      };
    }

    const totalCols = dataset.columns.length;
    let modelName = 'Random Forest Classifier';
    let reasoning = 'A high concentration of categorical variables recommends Random Forest to avoid scaling issues.';
    const isClass = modelType === 'classification';
    if (!isClass) {
      modelName = 'XGBoost Regressor';
      reasoning = 'Continuous distribution curves benefit from sequence-based gradient boosting.';
    } else if (totalCols > 15) {
      modelName = 'Multi-Layer Perceptron (Neural Network)';
      reasoning = 'High-dimensional feature spaces benefit from a learned layered representation.';
    }
    return { modelName, reasoning, primaryLabel: null as string | null, primaryValue: null as number | null, isReal: false };
  };

  const edaFit = getEDAModelFit();

  const isBinaryConfusion = (cm: any): cm is { tp: number; tn: number; fp: number; fn: number; positiveLabel?: string } =>
    !!cm && typeof cm.tp === 'number';

  // Real SHAP values are the more rigorous explainability signal when
  // available; fall back to the model's own built-in importance otherwise.
  // Both are real — this only decides which one is more trustworthy to lead with.
  const hasShap = !!(mlResult?.shapImportance && mlResult.shapImportance.length > 0);
  const importanceData = hasShap ? mlResult!.shapImportance! : (mlResult?.featureImportance || []);
  const importanceLabel = hasShap ? 'Real SHAP Feature Importance' : 'Real Feature Importance (model built-in)';

  return (
    <div className="space-y-8 animate-fade-in" id="ml_module">

      {/* 🚀 Consistent ML Pipeline stage header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-slate-900/40 backdrop-blur-md rounded-2xl border border-slate-800/80 p-5 shadow-2xl relative overflow-hidden">
        <div>
          <span className="text-[10px] font-bold text-indigo-400 tracking-widest font-mono uppercase">MODEL CONFIGURATION</span>
          <h2 className="text-xl font-extrabold text-white tracking-tight mt-1">4. Predictive Machine Learning Modeling</h2>
          <p className="text-xs text-slate-400 mt-1 max-w-xl">
            Train real scikit-learn / XGBoost models on your actual data and review genuinely measured performance.
          </p>
        </div>
        <span className="bg-[#131B2E]/90 text-indigo-400 text-[10px] font-mono font-bold px-3 py-1 rounded-full border border-indigo-500/30 uppercase tracking-wide flex items-center gap-1.5 shadow-md shrink-0">
          <BrainCircuit className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
          ML Pipeline Stage: ACTIVE
        </span>
      </div>

      {/* 1. MODEL RECOMMENDATION / REAL CHAMPION BANNER */}
      <div className="bg-gradient-to-r from-indigo-950/80 via-purple-950/40 to-slate-900 border border-indigo-500/25 p-5 sm:p-6 rounded-2xl relative overflow-hidden shadow-2xl">
        <div className="absolute top-0 right-0 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-transparent via-indigo-400 to-transparent" />

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-400 animate-pulse shrink-0" />
              <span className="text-[10px] font-mono font-bold text-indigo-400 uppercase tracking-widest font-mono">
                {edaFit.isReal ? 'Real Trained Champion Model' : 'Pre-Training Recommendation (Heuristic)'}
              </span>
            </div>
            <h3 className="text-lg font-black text-white leading-tight flex items-center gap-2">
              {edaFit.isReal ? 'Best Real Model:' : 'Suggested Starting Point:'} <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 to-purple-300">{edaFit.modelName}</span>
            </h3>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              {edaFit.reasoning}
            </p>
          </div>

          {edaFit.isReal && edaFit.primaryValue !== null && edaFit.primaryValue !== undefined ? (
            <div className="bg-indigo-500/10 border border-indigo-400/20 px-4.5 py-3 rounded-xl flex items-center justify-between gap-4 shrink-0 font-mono">
              <div>
                <p className="text-[9px] text-[#A2B4F6] uppercase tracking-wider font-bold">Real {edaFit.primaryLabel} (test set)</p>
                <p className="text-2xl font-black text-indigo-300 mt-0.5">
                  {edaFit.primaryLabel === 'Accuracy' ? `${(edaFit.primaryValue * 100).toFixed(1)}%` : edaFit.primaryValue.toFixed(3)}
                </p>
              </div>
              <div className="w-10 h-10 rounded-full border-2 border-indigo-400/30 flex items-center justify-center bg-indigo-950">
                <Cpu className="w-5 h-5 text-indigo-400" />
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/50 border border-slate-700/40 px-4 py-3 rounded-xl flex items-center gap-2 shrink-0 font-mono text-[10px] text-slate-400 max-w-[220px]">
              <Info className="w-3.5 h-3.5 shrink-0" /> Train a model below to see real, measured performance.
            </div>
          )}
        </div>
      </div>

      {/* 2. TARGET VARIABLES SUITABILITY SCOUT CARD */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4 mb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[9.5px] font-mono font-bold text-emerald-450 text-emerald-400 uppercase tracking-widest bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                Data Field Audit
              </span>
            </div>
            <h3 className="text-base font-black text-white mt-1.5 flex items-center gap-2 font-sans tracking-tight">
              <Database className="w-4.5 h-4.5 text-emerald-400" /> Target Variable Suitability Analysis
            </h3>
            <p className="text-xs text-slate-400 mt-0.5 leading-relaxed font-sans">
              Analyzes category layouts, distinct values, and missing percentages to locate optimal predictive goals.
            </p>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            {dataset.columns.length} columns analyzed
          </span>
        </div>

        {/* Scrollable list or compact responsive cards layout of candidate variables */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4.5">
          {targetRecommendations.slice(0, 6).map((item, idx) => {
            const isSelected = target === item.name;
            const isExcellent = item.grade === 'Excellent';
            const isGood = item.grade === 'Good';
            const isFair = item.grade === 'Fair';

            return (
              <div
                key={item.name}
                className={`p-4.5 rounded-xl border transition-all duration-300 relative flex flex-col justify-between ${
                  isSelected
                    ? 'bg-gradient-to-b from-indigo-950/40 to-indigo-900/10 border-indigo-500 shadow-md shadow-indigo-500/5'
                    : 'bg-slate-950/60 border-slate-850 hover:bg-[#0d1220]/80 hover:border-slate-700'
                }`}
              >
                {/* Score and Grade header */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2 border-b border-slate-850 pb-2 mb-2">
                    <span className="font-mono text-slate-200 font-extrabold text-xs truncate max-w-[130px]" title={item.name}>
                      {item.name}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[8.5px] font-bold font-mono tracking-wider ${
                      isExcellent ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' :
                      isGood ? 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/20' :
                      isFair ? 'bg-amber-500/15 text-amber-400 border border-amber-500/20' :
                      'bg-slate-900 text-slate-405 text-slate-400 border border-slate-800'
                    }`}>
                      {item.grade} ({item.score} pts)
                    </span>
                  </div>

                  <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 font-bold">
                    <span>Task Type:</span>
                    <span className="text-slate-300 uppercase">{item.type}</span>
                  </div>

                  <p className="text-[10px] text-slate-405 text-slate-400 leading-snug mt-1 min-h-[46px] font-sans">
                    {item.reason}
                  </p>
                </div>

                {/* Set as Target CTA button */}
                <div className="mt-4 pt-3 border-t border-slate-850 flex items-center justify-between gap-1.5 font-sans">
                  <span className="text-[8.5px] font-mono text-slate-500">
                    {isSelected ? '🎯 Active Target' : 'Candidate'}
                  </span>

                  <button
                    type="button"
                    onClick={() => handleSetTargetFromAdvisor(item.name, item.type, item.suggestedFeatures)}
                    className={`py-1.5 px-3 rounded-lg text-[10px] font-black font-mono transition-all duration-200 shrink-0 cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-600 border border-indigo-500 text-white shadow shadow-indigo-600/30'
                        : 'bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800'
                    }`}
                  >
                    {isSelected ? 'ISOLATED' : 'SET AS TARGET'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. CONFIGURATION HUB CARD & TRAINING LOGS */}
      <div className="bg-slate-900/60 backdrop-blur-md p-6 sm:p-8 rounded-2xl border border-slate-800 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

        <span className="text-[10px] font-mono font-bold text-indigo-400 uppercase tracking-widest font-sans">MODEL TRAINING</span>
        <h2 className="text-xl font-extrabold text-white tracking-tight mt-1 flex items-center gap-2 font-sans">
          <Cpu className="w-5.5 h-5.5 text-indigo-400" /> Real Model Training
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-xl font-sans">
          Configure hyperparameters and launch a genuine scikit-learn / XGBoost training run against your live dataset.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8 relative z-10">
          {/* Section A: Target Selector and Feature Checklist */}
          <div className="space-y-4">
            <div>
              <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest font-mono mb-2 px-0.5">
                A. Select Target Variable (Continuous / Discrete)
              </label>
              <select
                value={target}
                onChange={(e) => handleTargetChange(e.target.value)}
                className="w-full bg-[#111625] border border-slate-800 rounded-xl p-3 text-xs text-white font-mono focus:ring-1 focus:ring-indigo-500 font-bold cursor-pointer"
              >
                {dataset.columns.map(c => (
                  <option key={c.name} value={c.name} className="bg-slate-950 font-mono">
                    {c.name} ({(c.type || '').toUpperCase()})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest font-mono mb-2 px-0.5">
                B. Feature Architect — Input Selection
              </label>
              <div className="max-h-[220px] overflow-y-auto space-y-1.5 border border-slate-850 p-3.5 rounded-xl bg-slate-950/45">
                {dataset.columns
                  .filter(c => c.name !== target)
                  .map(col => {
                    const isSuggested = aiSuggestedFeatures.includes(col.name);
                    return (
                      <label
                        key={col.name}
                        className="flex items-center gap-3 text-xs p-2 rounded-lg hover:bg-slate-900/60 cursor-pointer transition-colors"
                      >
                        <input
                          type="checkbox"
                          checked={selectedFeatures.includes(col.name)}
                          onChange={() => toggleFeature(col.name)}
                          className="rounded text-indigo-500 focus:ring-indigo-600 cursor-pointer w-4 h-4 bg-slate-900 border-slate-800"
                        />
                        <span className="font-mono text-slate-300 flex-1 truncate">{col.name}</span>
                        {isSuggested && (
                          <span className="text-[8px] font-mono tracking-wider uppercase bg-indigo-505 bg-indigo-500/15 text-indigo-400 font-bold px-2 py-0.5 rounded border border-indigo-500/20 shrink-0">
                            Advised
                          </span>
                        )}
                      </label>
                    );
                  })}
              </div>
            </div>

            {/* Model Leakage Guard Auditing */}
            {expertMode && (
            <div className={`p-3.5 rounded-xl border transition-all ${
              leakageAuditResult.leakageRisk === 'High'
                ? 'bg-rose-950/25 border-rose-500/30 text-rose-300'
                : leakageAuditResult.leakageRisk === 'Medium'
                ? 'bg-amber-950/20 border-amber-500/30 text-amber-300'
                : 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300'
            }`}>
              <div className="flex items-center gap-2 mb-1.5">
                {leakageAuditResult.leakageRisk === 'High' ? (
                  <ShieldAlert className="w-4 h-4 text-rose-500 shrink-0" style={{ color: '#f43f5e' }} />
                ) : (
                  <ShieldCheck className="w-4 h-4 text-emerald-405 shrink-0" />
                )}
                <span className="text-[10.5px] font-extrabold uppercase tracking-wider font-sans">
                  Leakage Audit Shield
                </span>
                <span className={`ml-auto font-mono text-[9px] font-black px-2 py-0.5 rounded border uppercase ${
                  leakageAuditResult.leakageRisk === 'High'
                    ? 'bg-rose-500/10 border-rose-500/20 text-rose-400'
                    : leakageAuditResult.leakageRisk === 'Medium'
                    ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                    : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                }`}>
                  {leakageAuditResult.leakageRisk} RISK
                </span>
              </div>
              <p className="text-[10px] leading-relaxed text-slate-300">
                {leakageAuditResult.passed
                  ? 'Audit Passed. Remove high-risk columns before training to prevent leakage or overfit memorization.'
                  : 'Leakage Threat Found! High-risk columns are selected. Remove them to prevent overfit memorization.'}
              </p>
              {leakageAuditResult.issues.length > 0 && (
                <div className="mt-2 space-y-1 max-h-[100px] overflow-y-auto pr-1">
                  {leakageAuditResult.issues.map((issue, idx) => (
                    <div key={idx} className="text-[9px] bg-slate-950/60 p-1.5 rounded border border-slate-850 font-mono text-slate-350 leading-normal">
                      <span className="text-rose-450 font-bold uppercase mr-1">[{issue.risk}]</span>
                      <strong className="text-white">{issue.feature}</strong>: {issue.message}
                    </div>
                  ))}
                </div>
              )}
            </div>
            )}
          </div>

          {/* Section B: Unified Model Parameters Regulator */}
          {expertMode && (
          <div className="space-y-4">
            <div>
              <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest font-mono mb-2 px-0.5">
                C. Pipeline Type Selector
              </label>
              <div className="grid grid-cols-3 gap-1.5 p-1 bg-slate-950/50 rounded-xl border border-slate-850 text-[10px] font-bold font-mono">
                {(['classification', 'regression', 'timeseries'] as const).map((type) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => {
                      setModelType(type);
                    }}
                    className={`py-2 px-1 rounded-lg cursor-pointer text-center tracking-tight transition-all duration-200 ${
                      modelType === type ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-250'
                    }`}
                  >
                    {type === 'classification' ? 'CLASSIFY' : type === 'regression' ? 'FORECAST' : 'TIMESERIES'}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest font-mono mb-2 px-0.5">
                D. Choose Learning Model / Algorithm To Train
              </label>
              <select
                value={selectedAlgorithmId}
                onChange={(e) => setSelectedAlgorithmId(e.target.value)}
                className="w-full bg-[#111625] border border-slate-800 rounded-xl p-3 text-xs text-white font-mono focus:ring-1 focus:ring-indigo-500 font-bold cursor-pointer"
              >
                <option value="auto" className="font-mono border-b border-slate-700 bg-emerald-900/20 text-emerald-300 font-black">
                  AUTO: Train &amp; Compare Linear, Random Forest &amp; XGBoost
                </option>

                <optgroup label="Ensemble & Tree Builders" className="text-indigo-300 font-black mt-2">
                  <option value="random_forest" className="font-mono text-slate-300 font-medium">Random Forest</option>
                  <option value="xgboost" className="font-mono text-slate-300 font-medium">XGBoost (eXtreme Gradient Boosting)</option>
                </optgroup>

                <optgroup label="Linear Methods" className="text-indigo-300 font-black">
                  <option value="linear" className="font-mono text-slate-300 font-medium">Linear / Logistic Regression</option>
                </optgroup>

                <optgroup label="Deep Learning" className="text-indigo-300 font-black">
                  <option value="mlp" className="font-mono text-slate-300 font-medium">Multi-Layer Perceptron (Neural Network)</option>
                </optgroup>
              </select>
              <p className="text-[10px] text-slate-450 leading-tight mt-1.5 px-0.5 font-sans">
                {selectedAlgorithmId === 'auto'
                  ? `Trains 3 real models on the same split and picks the highest-scoring one on held-out test data.`
                  : `Trains a single real "${selectedAlgorithmId.replace('_', ' ').toUpperCase()}" model.`}
              </p>
            </div>

            <div className="border border-slate-800 rounded-xl p-4.5 bg-slate-950/20 space-y-4">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-300 uppercase tracking-wider font-mono">
                <Settings className="w-4.5 h-4.5 text-indigo-400" />
                <span>Hyperparameter Regulators</span>
              </div>

              <div className="grid grid-cols-2 gap-3.5 text-xs">
                <div>
                  <label className="block text-[10px] text-slate-400 font-bold font-mono mb-1">Max Estimators</label>
                  <select
                    value={estimators}
                    onChange={(e) => setEstimators(Number(e.target.value))}
                    className="w-full bg-[#111625] border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono cursor-pointer"
                  >
                    <option value={50}>50 Trees</option>
                    <option value={100}>100 Trees</option>
                    <option value={150}>150 Trees</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[10px] text-slate-400 font-bold font-mono mb-1">Max Depth</label>
                  <select
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(Number(e.target.value))}
                    className="w-full bg-[#111625] border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono cursor-pointer"
                  >
                    <option value={5}>Depth (5)</option>
                    <option value={8}>Depth (8)</option>
                    <option value={12}>Depth (12)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[10px] text-slate-400 font-bold font-mono mb-1">K Cluster Density</label>
                  <select
                    value={clusterK}
                    onChange={(e) => setClusterK(Number(e.target.value))}
                    className="w-full bg-[#111625] border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono cursor-pointer animate-pulse-once"
                  >
                    <option value={2}>2 Clusters</option>
                    <option value={3}>3 Clusters</option>
                    <option value={4}>4 Clusters</option>
                    <option value={5}>5 Clusters</option>
                  </select>
                </div>

                <div>
                  <label className="block text-[10px] text-slate-400 font-bold font-mono mb-1">MLP Max Iterations</label>
                  <select
                    value={mlpMaxIterations}
                    onChange={(e) => setMlpMaxIterations(Number(e.target.value))}
                    className="w-full bg-[#111625] border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono cursor-pointer"
                  >
                    <option value={100}>100 iterations</option>
                    <option value={200}>200 iterations</option>
                    <option value={300}>300 iterations</option>
                  </select>
                </div>

                <div className="col-span-2">
                  <label className="block text-[10px] text-slate-400 font-bold font-mono mb-1.5 flex justify-between">
                    <span>Validation Split hold ratio</span>
                    <span className="text-indigo-400 font-bold">{Math.round(splitRatio * 100)}% Train / {Math.round((1 - splitRatio) * 100)}% Test</span>
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="0.9"
                    step="0.1"
                    value={splitRatio}
                    onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500 cursor-ew-resize h-1 bg-slate-950 rounded-lg"
                  />
                </div>
              </div>
            </div>
          </div>
          )}

          {/* Section C: Training Progress Logger and Execution Launch */}
          <div className="flex flex-col justify-between bg-slate-950/40 backdrop-blur-md border border-slate-800/80 p-5 rounded-2xl text-white shadow-inner">
            <div className="space-y-3">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-indigo-400 flex items-center gap-2 font-mono">
                <Terminal className="w-4 h-4 text-emerald-400 animate-pulse" /> Training Log
              </h3>
              <div className="bg-slate-900 border border-slate-800 font-mono text-[10px] p-3.5 rounded-xl space-y-2 h-[160px] overflow-y-auto text-emerald-400 shadow-inner">
                {progressLog.length === 0 ? (
                  <span className="text-slate-500 italic flex items-center gap-1.5 mt-2 font-mono">
                    System awaiting pipeline compilation. Select features and trigger.
                  </span>
                ) : (
                  progressLog.map((log, i) => (
                    <div key={i} className="leading-relaxed flex items-start gap-1">
                      <span className="text-indigo-400">›</span>
                      <span>{log}</span>
                    </div>
                  ))
                )}
                {loadingML && (
                  <div className="flex items-center gap-1.5 text-indigo-300">
                    <Loader2 className="w-3 h-3 animate-spin shrink-0" />
                    <span>Waiting on the real training call...</span>
                  </div>
                )}
              </div>
            </div>

            <button
              onClick={handleLaunchPipeline}
              disabled={loadingML || selectedFeatures.length === 0}
              className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-3.5 px-4 rounded-xl flex items-center justify-center gap-2 mt-4 shadow-lg shadow-indigo-600/30 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer border-0 font-mono"
            >
              {loadingML ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Training Real Model...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current text-white shrink-0 animate-bounce-once" /> TRAIN REAL MODEL
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* RESULTS PORTFOLIO — everything below is sourced from mlResult (real Python training response) */}
      {mlResult && (
        <div className="space-y-8 animate-fade-in">

          {/* SYSTEM-WIDE MODEL COMPARISON & SERIALIZATION ROW */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* COMPARISON BENCHMARK DATAFRAME */}
            <div className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800 shadow-xl col-span-2 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h3 className="font-extrabold text-white text-sm">Real Model Comparison Benchmarks</h3>
                  <p className="text-[10px] text-slate-400 font-mono mt-0.5">Every candidate model is shown — ranked by cross-validation, not just the winner</p>
                </div>
                <Table className="w-4.5 h-4.5 text-indigo-400 shrink-0" />
              </div>

              <div className="overflow-x-auto text-[11px]">
                <table className="w-full text-left border-collapse font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-405 text-[10px] font-bold text-slate-400 tracking-wider">
                      <th className="pb-2.5 font-bold">Trained Model</th>
                      <th className="pb-2.5 font-bold text-center">Test Score</th>
                      <th className="pb-2.5 font-bold text-center">CV Mean (±std)</th>
                      <th className="pb-2.5 text-center font-bold">Train Time</th>
                      <th className="pb-2.5 text-right font-bold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40">
                    {(mlResult.comparison || []).map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                        <td className="py-3 font-extrabold text-white flex items-center gap-1.5">
                          {idx === 0 ? (
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping inline-block" />
                          ) : (
                            <span className="w-1.5 h-1.5 rounded-full bg-slate-600 inline-block" />
                          )}
                          {row.modelName}
                        </td>
                        <td className="font-bold text-white text-center">
                          {row.primaryMetric.toLowerCase().includes('accuracy')
                            ? `${(row.metricValue * 100).toFixed(1)}%`
                            : row.metricValue.toFixed(3)}
                          <span className="block text-[8.5px] text-slate-500 font-normal normal-case">{row.primaryMetric}</span>
                        </td>
                        <td className="text-center text-indigo-300 font-bold">
                          {row.cv && !row.cv.error
                            ? (row.cv.metric === 'accuracy' ? `${(row.cv.mean * 100).toFixed(1)}%` : row.cv.mean.toFixed(3)) + ` (±${row.cv.std.toFixed(3)})`
                            : '—'}
                        </td>
                        <td className="text-slate-450 text-center">{row.executionTimeMs.toLocaleString()} ms</td>
                        <td className="text-right py-3 pr-1">
                          <span className={`px-2 py-0.5 rounded-md font-mono text-[9px] font-bold tracking-wider ${
                            idx === 0
                              ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20 shadow-md shadow-emerald-550/5'
                              : 'bg-slate-950 text-slate-400 border border-slate-850'
                          }`}>
                            {idx === 0 ? '🏆 REAL CHAMPION' : `RANK #${idx + 1}`}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {(!mlResult.comparison || mlResult.comparison.length === 0) && (
                      <tr>
                        <td colSpan={5} className="py-4 text-center text-slate-500 text-[10px]">
                          Only one model was trained this run — pick "AUTO" to compare multiple algorithms.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* REAL MODEL ARTIFACT DOWNLOAD */}
            <div className="bg-gradient-to-br from-[#0B0F19] to-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl flex flex-col justify-between">
              <div className="space-y-3">
                <span className="text-[9px] font-mono font-bold tracking-wider uppercase text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                  REAL EXPORTABLE MODEL ARTIFACT
                </span>
                <h3 className="font-extrabold text-white text-sm mt-2 flex items-center gap-2">
                  <Database className="w-4.5 h-4.5 text-emerald-400" /> Trained Model Package
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed font-sans mt-1">
                  Downloads the genuine fitted scikit-learn Pipeline (preprocessing + model) as a real <code>.joblib</code> file, ready to load in Python.
                </p>

                <div className="p-3.5 bg-slate-950/65 rounded-xl border border-slate-850 space-y-1.5 text-[10.5px] font-mono text-slate-400 shadow-inner">
                  <div className="flex justify-between">
                    <span>Algorithm:</span>
                    <strong className="text-indigo-400 font-bold">{mlResult.modelAlgorithm}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Train / Test Rows:</span>
                    <strong className="text-slate-300">{mlResult.trainRows ?? '-'} / {mlResult.testRows ?? '-'}</strong>
                  </div>
                  <div className="flex justify-between border-t border-slate-850 pt-1.5 mt-1 text-[10px]">
                    <span>Model ID:</span>
                    <span className="text-slate-500 truncate max-w-[140px]" title={mlResult.modelId}>{mlResult.modelId?.slice(0, 12) || 'n/a'}…</span>
                  </div>
                </div>
              </div>

              <button
                onClick={downloadModelFile}
                disabled={!mlResult.modelId || downloadingModel}
                className="w-full bg-slate-805 hover:bg-slate-800 bg-[#121927] border border-slate-800 hover:border-slate-700 text-slate-100 font-bold text-xs py-3 rounded-xl flex items-center justify-center gap-2 cursor-pointer mt-5 transition-all shadow-inner disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {downloadingModel ? (
                  <Loader2 className="w-4 h-4 text-emerald-400 shrink-0 animate-spin" />
                ) : (
                  <Download className="w-4 h-4 text-emerald-400 shrink-0" />
                )}
                {downloadingModel ? 'DOWNLOADING…' : 'DOWNLOAD REAL MODEL (.joblib)'}
              </button>
            </div>
          </div>

          {/* TRAINING DISPLAY TABS SELECTOR */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 border-b border-slate-800">
              <span className="text-[11px] font-bold text-slate-400 font-mono tracking-wider uppercase pb-2 px-1">REALM DASHBOARDS:</span>
              <div className="flex gap-1 overflow-x-auto pb-2 shrink-0">
                <button
                  type="button"
                  onClick={() => setActiveRealm('supervised')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono transition-colors tracking-tight ${
                    activeRealm === 'supervised' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-950/40 hover:text-white'
                  }`}
                >
                  <Workflow className="w-3.5 h-3.5 inline mr-1" /> Supervised Outputs
                </button>
                <button
                  type="button"
                  onClick={() => setActiveRealm('unsupervised')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono transition-colors tracking-tight ${
                    activeRealm === 'unsupervised' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-950/40 hover:text-white'
                  }`}
                >
                  <BarChart2 className="w-3.5 h-3.5 inline mr-1" /> Unsupervised Outputs
                </button>
                <button
                  type="button"
                  onClick={() => setActiveRealm('ensemble')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono transition-colors tracking-tight ${
                    activeRealm === 'ensemble' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-950/40 hover:text-white'
                  }`}
                >
                  <GitBranch className="w-3.5 h-3.5 inline mr-1" /> Ensemble Methods
                </button>
                <button
                  type="button"
                  onClick={() => setActiveRealm('deep_learning')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono transition-colors tracking-tight ${
                    activeRealm === 'deep_learning' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-950/40 hover:text-white'
                  }`}
                >
                  <Brain className="w-3.5 h-3.5 inline mr-1" /> Deep Learning
                </button>
                <button
                  type="button"
                  onClick={() => setActiveRealm('comparison')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono transition-colors tracking-tight ${
                    activeRealm === 'comparison' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-950/40 hover:text-white'
                  }`}
                >
                  <Trophy className="w-3.5 h-3.5 inline mr-1" /> Model Comparison
                </button>
                <button
                  type="button"
                  onClick={() => setActiveRealm('agent')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono transition-colors tracking-tight ${
                    activeRealm === 'agent' ? 'bg-indigo-600 text-white' : 'text-slate-400 bg-slate-950/40 hover:text-white'
                  }`}
                >
                  <Bot className="w-3.5 h-3.5 inline mr-1" /> Agentic Analysis
                </button>
              </div>
            </div>

            {/* REALM A: SUPERVISED LEARNING PORTFOLIO */}
            {activeRealm === 'supervised' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in">

                {/* 1. Supervised Evaluation Metrics Card */}
                <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl flex flex-col justify-between hover:border-slate-700 transition">
                  <div className="space-y-4">
                    <h4 className="font-extrabold text-white text-xs tracking-wider uppercase font-mono text-indigo-400">Real Test-Set Metrics</h4>
                    <div className="grid grid-cols-2 gap-3.5">
                      {mlResult.modelType === 'classification' ? (
                        <>
                          <div className="bg-indigo-500/5 p-4 rounded-xl border border-indigo-500/10 text-center">
                            <p className="text-[9px] text-slate-400 font-mono font-bold uppercase tracking-wider">Test Accuracy</p>
                            <p className="text-2xl font-black text-indigo-300 font-mono mt-0.5">
                              {((mlResult.metrics.accuracy ?? 0) * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="bg-emerald-500/5 p-4 rounded-xl border border-emerald-500/10 text-center">
                            <p className="text-[9px] text-slate-400 font-mono font-bold uppercase tracking-wider">F1-Score</p>
                            <p className="text-2xl font-black text-emerald-400 font-mono mt-0.5">
                              {((mlResult.metrics.f1Score ?? 0) * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="col-span-2 bg-slate-950/60 p-3.5 rounded-xl border border-slate-850 font-mono text-[10.5px] text-slate-350 space-y-1 my-1 text-center">
                            <div>Precision: <strong className="text-white">{(mlResult.metrics.precision ?? 0).toFixed(3)}</strong> · Recall: <strong className="text-white">{(mlResult.metrics.recall ?? 0).toFixed(3)}</strong></div>
                            {mlResult.metrics.rocAuc !== undefined && (
                              <div>ROC-AUC: <strong className="text-white">{mlResult.metrics.rocAuc.toFixed(3)}</strong></div>
                            )}
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="bg-indigo-500/5 p-4 rounded-xl border border-indigo-500/10 text-center col-span-2">
                            <p className="text-[9px] text-slate-450 font-mono font-bold uppercase tracking-wider">R² Explained Variance</p>
                            <p className="text-2xl font-black text-indigo-300 font-mono mt-0.5">
                              {(mlResult.metrics.r2Score ?? 0).toFixed(3)}
                            </p>
                            <span className="text-[9px] text-slate-500 mt-1 block">Accountability of total feature info</span>
                          </div>
                          <div className="col-span-2 bg-slate-950 p-3.5 rounded-xl border border-slate-850 font-mono text-[10.5px] text-slate-300 flex justify-between px-5 font-bold">
                            <span>MAE: <strong className="text-white">{(mlResult.metrics.mae ?? 0).toFixed(2)}</strong></span>
                            <span>RMSE: <strong className="text-white">{(mlResult.metrics.rmse ?? 0).toFixed(2)}</strong></span>
                          </div>
                        </>
                      )}
                    </div>

                    {mlResult.cv && !mlResult.cv.error && (
                      <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-850 font-mono text-[10px] text-slate-400 flex justify-between items-center">
                        <span>{mlResult.cv.folds}-Fold Cross-Validation ({mlResult.cv.metric}):</span>
                        <strong className="text-indigo-300">
                          {mlResult.cv.metric === 'accuracy' ? `${(mlResult.cv.mean * 100).toFixed(1)}%` : mlResult.cv.mean.toFixed(3)} (±{mlResult.cv.std.toFixed(3)})
                        </strong>
                      </div>
                    )}
                  </div>

                  {mlResult.confusionMatrix && isBinaryConfusion(mlResult.confusionMatrix) && (
                    <div className="mt-5 border-t border-slate-850 pt-4">
                      <p className="text-[10px] text-slate-450 font-mono font-bold uppercase tracking-wider mb-2.5">
                        Confusion Matrix {mlResult.confusionMatrix.positiveLabel ? `(positive = "${mlResult.confusionMatrix.positiveLabel}")` : ''}
                      </p>
                      <div className="grid grid-cols-2 gap-1.5 font-mono text-center text-xs">
                        <div className="bg-emerald-950/40 p-2.5 rounded-lg border border-emerald-900/30">
                          <p className="text-[9px] text-emerald-450 font-bold uppercase tracking-wider">True Positives</p>
                          <p className="text-lg font-black text-emerald-400 mt-0.5">{mlResult.confusionMatrix.tp}</p>
                        </div>
                        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-850">
                          <p className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">False Positives</p>
                          <p className="text-lg font-black text-slate-300 mt-0.5">{mlResult.confusionMatrix.fp}</p>
                        </div>
                        <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-850">
                          <p className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">False Negatives</p>
                          <p className="text-lg font-black text-slate-300 mt-0.5">{mlResult.confusionMatrix.fn}</p>
                        </div>
                        <div className="bg-emerald-950/40 p-2.5 rounded-lg border border-emerald-900/30">
                          <p className="text-[9px] text-emerald-450 font-bold uppercase tracking-wider">True Negatives</p>
                          <p className="text-lg font-black text-emerald-400 mt-0.5">{mlResult.confusionMatrix.tn}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {mlResult.confusionMatrix && !isBinaryConfusion(mlResult.confusionMatrix) && (mlResult.confusionMatrix as any).matrix && (
                    <div className="mt-5 border-t border-slate-850 pt-4">
                      <p className="text-[10px] text-slate-450 font-mono font-bold uppercase tracking-wider mb-2.5">Multiclass Confusion Matrix</p>
                      <div className="overflow-x-auto">
                        <table className="text-[9px] font-mono border-collapse">
                          <thead>
                            <tr>
                              <th className="p-1 text-slate-500"></th>
                              {(mlResult.confusionMatrix as any).labels.map((l: string) => (
                                <th key={l} className="p-1 text-slate-400 font-bold">{l}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {(mlResult.confusionMatrix as any).matrix.map((rowArr: number[], rIdx: number) => (
                              <tr key={rIdx}>
                                <td className="p-1 text-slate-400 font-bold">{(mlResult.confusionMatrix as any).labels[rIdx]}</td>
                                {rowArr.map((val, cIdx) => (
                                  <td key={cIdx} className={`p-1.5 text-center rounded ${rIdx === cIdx ? 'bg-emerald-950/50 text-emerald-300 font-bold' : 'bg-slate-950 text-slate-400'}`}>{val}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>

                {/* 2. REALTIME PREDICTION ARRAY TABLE GRID */}
                <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl col-span-2 space-y-4 shadow-xl">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                    <div>
                      <h4 className="font-extrabold text-white text-sm">Real Test-Set Predictions</h4>
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5">Actual vs. predicted values on held-out rows the model never trained on</p>
                    </div>
                    <span className="text-[10px] font-mono bg-indigo-505 bg-indigo-500/10 border border-indigo-500/25 px-2.5 py-0.5 rounded text-indigo-400 font-bold">
                      {mlResult.predictions.length} rows evaluated
                    </span>
                  </div>

                  <div className="overflow-x-auto text-[11px]">
                    <table className="w-full text-left border-collapse font-mono">
                      <thead>
                        <tr className="border-b border-slate-850 text-slate-450 text-[10px] font-bold">
                          <th className="pb-2">Row ID</th>
                          <th className="pb-2">Actual</th>
                          <th className="pb-2">Predicted</th>
                          <th className="pb-2 text-right">Confidence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850/40">
                        {mlResult.predictions
                          .slice((selectedRawPredPage - 1) * 6, selectedRawPredPage * 6)
                          .map((pred, i) => {
                            const isCorrect = String(pred.actual).toLowerCase() === String(pred.predicted).toLowerCase();
                            return (
                              <tr key={i} className="hover:bg-slate-900/20">
                                <td className="py-2.5 text-slate-500 font-semibold">#{pred.id}</td>
                                <td className="py-2.5 font-bold text-slate-300">{pred.actual}</td>
                                <td className="py-2.5 font-black text-white">
                                  {mlResult.modelType === 'classification' ? (
                                    <span className={isCorrect ? 'text-emerald-400' : 'text-rose-400'}>{pred.predicted}</span>
                                  ) : (
                                    pred.predicted
                                  )}
                                </td>
                                <td className="py-2.5 text-right font-bold text-indigo-300">
                                  {pred.confidence !== undefined && pred.confidence !== null ? `${pred.confidence}%` : '—'}
                                </td>
                              </tr>
                            );
                          })}
                      </tbody>
                    </table>
                  </div>

                  <div className="flex justify-between items-center pt-2 font-mono text-[10px]">
                    <button
                      disabled={selectedRawPredPage === 1}
                      onClick={() => setSelectedRawPredPage(p => Math.max(1, p - 1))}
                      className="px-3 py-1 bg-slate-900 border border-slate-800 rounded hover:text-white cursor-pointer hover:border-slate-700 disabled:opacity-40"
                    >
                      Previous Index
                    </button>
                    <span className="text-slate-400 font-bold">Page {selectedRawPredPage} of {Math.max(1, Math.ceil(mlResult.predictions.length / 6))}</span>
                    <button
                      disabled={selectedRawPredPage >= Math.ceil(mlResult.predictions.length / 6)}
                      onClick={() => setSelectedRawPredPage(p => p + 1)}
                      className="px-3 py-1 bg-slate-900 border border-slate-800 rounded hover:text-white cursor-pointer hover:border-slate-700 disabled:opacity-40"
                    >
                      Next Index
                    </button>
                  </div>
                </div>

              </div>
            )}

            {/* REALM B: UNSUPERVISED LEARNING & DIMENSIONALITY REDUCTION (real K-Means + PCA on the active dataset) */}
            {activeRealm === 'unsupervised' && unsupervisedRes && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in text-[11px]">

                {/* 1. Clustering and PCA plots */}
                <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl col-span-2 space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div>
                      <h4 className="font-extrabold text-white text-sm">Principal Component Analysis (PCA) Coordinates Plot</h4>
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5">Real features projected into PC1 & PC2 eigenvectors, colored by real K-Means cluster</p>
                    </div>
                    <LineChart className="w-4.5 h-4.5 text-indigo-400 shrink-0" />
                  </div>

                  <div className="h-[220px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 8, right: 10, left: -20, bottom: 5 }}>
                        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="pc1" name="PC1 Component" stroke="#64748b" fontSize={9} />
                        <YAxis type="number" dataKey="pc2" name="PC2 Component" stroke="#64748b" fontSize={9} />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          contentStyle={{ fontSize: '11px', background: 'rgba(15,23,42,0.92)', backdropFilter: 'blur(8px)', borderColor: '#334155', color: '#fff', borderRadius: '12px' }}
                          formatter={(value, name) => [value, name === 'clusterId' ? 'Cluster ID' : String(name).toUpperCase()]}
                        />
                        <Scatter name="PCA Project" data={unsupervisedRes.pcaComponents} fill="#818cf8">
                          {unsupervisedRes.pcaComponents.map((pt, index) => {
                            const colors = ['#38bdf8', '#fb7185', '#34d399', '#fbbf24', '#c084fc'];
                            const dotColor = colors[pt.clusterId % colors.length];
                            return <cell key={`cell-${index}`} fill={dotColor} />;
                          })}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Variance explanations */}
                  <div className="bg-slate-950/80 rounded-xl p-4.5 border border-slate-850 space-y-3">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider font-mono">Principal Components Explained Variance Ratios</p>
                    <div className="grid grid-cols-2 gap-4">
                      {unsupervisedRes.explainedVarianceRatios.map((item, idx) => (
                        <div key={idx} className="space-y-1 font-mono text-[10.5px]">
                          <div className="flex justify-between font-bold text-slate-350">
                            <span>{item.component} Info Retained:</span>
                            <span className="text-white">{(item.ratio * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
                            <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${item.ratio * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 2. Density metrics + Centroids map */}
                <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl flex flex-col justify-between space-y-4">
                  <div className="space-y-4">
                    <h4 className="font-extrabold text-white text-xs tracking-wider uppercase font-mono text-indigo-400">Cluster Density Diagnostics</h4>
                    <div className="grid grid-cols-2 gap-3 font-mono text-center">
                      <div className="bg-indigo-500/5 border border-indigo-500/10 p-3.5 rounded-xl">
                        <p className="text-[8.5px] text-slate-400 uppercase tracking-wider font-bold">Silhouette Score</p>
                        <p className="text-xl font-black text-indigo-300 mt-1">{unsupervisedRes.silhouetteScore.toFixed(3)}</p>
                      </div>
                      <div className="bg-emerald-500/5 border border-emerald-500/10 p-3.5 rounded-xl">
                        <p className="text-[8.5px] text-slate-400 uppercase tracking-wider font-bold">Davies-Bouldin Index</p>
                        <p className="text-xl font-black text-emerald-400 mt-1">{unsupervisedRes.daviesBouldinIndex.toFixed(3)}</p>
                      </div>
                    </div>

                    <div className="space-y-2 mt-4">
                      <p className="text-[10px] text-slate-450 font-mono font-bold uppercase tracking-wider">Cluster Mathematical Centroids</p>
                      <div className="space-y-2 max-h-[170px] overflow-y-auto pr-1">
                        {unsupervisedRes.centroids.map((cent, idx) => (
                          <div key={idx} className="bg-slate-950 p-2.5 rounded-xl border border-slate-850 font-mono text-[10px] space-y-1 hover:border-slate-700 transition">
                            <div className="flex justify-between border-b border-slate-850 pb-1 mb-1">
                              <span className="font-extrabold text-white">Cluster #{cent.clusterId} (n={cent.size})</span>
                              <span className="text-indigo-400 font-bold">Center point</span>
                            </div>
                            {Object.entries(cent.coordinates).slice(0, 3).map(([key, val]) => (
                              <div key={key} className="flex justify-between text-slate-400">
                                <span className="truncate max-w-[120px]">{key}:</span>
                                <span className="font-bold text-slate-200">{val}</span>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <p className="text-[9.5px] text-slate-450 italic font-mono leading-tight pt-2 border-t border-slate-850">
                    *K-Means partitioned the records by optimizing Euclidean distance parameters globally across all coordinates. Silhouette/Davies-Bouldin computed directly from these real assignments.
                  </p>
                </div>

              </div>
            )}

            {/* REALM C: ENSEMBLE METHODS (real Random Forest tree splits, when available) */}
            {activeRealm === 'ensemble' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in text-[11px]">

                {/* 1. Feature Importance Rankings (real, from the trained model — any algorithm) */}
                <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <h4 className="font-extrabold text-white text-xs tracking-wider uppercase font-mono text-indigo-400">{importanceLabel}</h4>
                    <BarChart2 className="w-4 h-4 text-indigo-400 shrink-0" />
                  </div>

                  <div className="h-[210px] mt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={importanceData}
                        layout="vertical"
                        margin={{ top: 5, right: 10, left: -20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#1e293b" />
                        <XAxis type="number" stroke="#64748b" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} fontSize={9} />
                        <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={9} tickLine={false} />
                        <Tooltip
                          formatter={(val: any) => [`${(val * 100).toFixed(1)}%`, hasShap ? 'Mean |SHAP value| (normalized)' : 'Real Importance Weight']}
                          contentStyle={{ fontSize: '11px', background: 'rgba(15,23,42,0.9)', borderColor: '#334155', color: '#fff', borderRadius: '12px' }}
                        />
                        <Bar dataKey="score" fill="#6366f1" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  {hasShap && (
                    <p className="text-[9.5px] text-slate-500 leading-tight font-mono">
                      Computed via SHAP on a real held-out test sample — the model-agnostic, game-theoretic gold standard for feature attribution.
                    </p>
                  )}

                  {mlResult.oobScore !== undefined && mlResult.oobScore !== null && (
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-850 flex justify-between items-center font-mono">
                      <span className="text-slate-400 font-bold">Real Out-Of-Bag (OOB) Score:</span>
                      <span className="px-2.5 py-1 bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-black rounded text-[10px]">
                        {(mlResult.oobScore * 100).toFixed(2)}%
                      </span>
                    </div>
                  )}
                </div>

                {/* 2. Estimators Individual Tree Explorer — real Random Forest trees only */}
                <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl col-span-2 space-y-4 shadow-xl">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                    <div>
                      <h4 className="font-extrabold text-white text-sm">Individual Decision Tree Explorer</h4>
                      <p className="text-[10px] text-slate-405 text-slate-400 font-mono mt-0.5">Real root-split data read directly from fitted Random Forest trees</p>
                    </div>
                    <GitBranch className="w-4.5 h-4.5 text-indigo-400 shrink-0" />
                  </div>

                  {mlResult.estimators && mlResult.estimators.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4.5">
                      {/* List element column */}
                      <div className="space-y-1.5 md:col-span-1 max-h-[220px] overflow-y-auto pr-1">
                        {mlResult.estimators.map((estim, idx) => (
                          <button
                            key={estim.id}
                            onClick={() => setSelectedEstimatorIdx(idx)}
                            className={`w-full text-left font-mono text-[10.5px] p-2.5 rounded-lg border cursor-pointer transition flex items-center justify-between ${
                              selectedEstimatorIdx === idx
                                ? 'bg-indigo-600 border-indigo-500 text-white font-extrabold shadow'
                                : 'bg-slate-950 border-slate-850 text-slate-400 hover:text-white hover:border-slate-700'
                            }`}
                          >
                            <span>{estim.name}</span>
                            <ChevronRight className="w-3.5 h-3.5 shrink-0" />
                          </button>
                        ))}
                      </div>

                      {/* Estimator details block */}
                      {mlResult.estimators[selectedEstimatorIdx] && (
                        <div className="md:col-span-2 bg-slate-950 rounded-xl p-4.5 border border-slate-850 space-y-4 font-mono">
                          <div className="flex justify-between items-center border-b border-slate-850 pb-2">
                            <span className="font-extrabold text-indigo-400 text-xs">
                              {mlResult.estimators[selectedEstimatorIdx].name} Root Split
                            </span>
                            <span className="text-[10px] text-slate-500">
                              Impurity: {mlResult.estimators[selectedEstimatorIdx].impurity}
                            </span>
                          </div>

                          <div className="p-2.5 bg-indigo-505 bg-indigo-500/10 border border-indigo-500/25 rounded-lg flex flex-col items-center justify-center text-center">
                            <span className="text-[9px] text-slate-450 font-bold uppercase tracking-wider text-indigo-350">Real Root Decision Split</span>
                            <span className="font-extrabold text-white mt-1">If [{mlResult.estimators[selectedEstimatorIdx].splitFeature}] &le; {mlResult.estimators[selectedEstimatorIdx].splitValue}</span>
                          </div>

                          <div className="grid grid-cols-2 gap-4 text-center">
                            <div className="p-2.5 bg-slate-900 border border-slate-800 rounded-lg">
                              <span className="text-[8.5px] text-slate-450 font-bold tracking-wider uppercase block">Tree Depth</span>
                              <span className="font-extrabold text-white text-sm mt-0.5 block">{mlResult.estimators[selectedEstimatorIdx].treeDepth}</span>
                            </div>
                            <div className="p-2.5 bg-slate-900 border border-slate-800 rounded-lg">
                              <span className="text-[8.5px] text-slate-450 font-bold tracking-wider uppercase block">Leaf Count</span>
                              <span className="font-extrabold text-white text-sm mt-0.5 block">{mlResult.estimators[selectedEstimatorIdx].leafCount}</span>
                            </div>
                          </div>

                          <div className="pt-2 border-t border-slate-850 text-[10px] text-slate-500 flex justify-between">
                            <span>Root Node Samples: {mlResult.estimators[selectedEstimatorIdx].sampleCount} rows</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="p-6 text-center bg-slate-950/50 border border-slate-850 rounded-xl text-[11px] text-slate-400 flex flex-col items-center gap-2">
                      <Info className="w-5 h-5 text-slate-500" />
                      Individual tree exploration is available for Random Forest models. Current champion: <strong className="text-slate-200">{mlResult.modelAlgorithm}</strong>. Select "Random Forest" from the algorithm dropdown and retrain to explore real trees.
                    </div>
                  )}
                </div>

              </div>
            )}

            {/* REALM D: DEEP LEARNING — real MLP loss curve, only when an MLP was trained */}
            {activeRealm === 'deep_learning' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in text-[11px]">
                {mlResult.deepLearning ? (
                  <>
                    {/* 1. Training Epoch Loss curve (real loss_curve_ + real validation_scores_) */}
                    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl col-span-1 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <h4 className="font-extrabold text-white text-xs tracking-wider uppercase font-mono text-indigo-400">Real Training Loss Curve</h4>
                        <LineChart className="w-4 h-4 text-indigo-400 shrink-0" />
                      </div>

                      <div className="h-[170px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <RechartsLineChart data={mlResult.deepLearning.trainingLogs}>
                            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                            <XAxis dataKey="epoch" stroke="#64748b" fontSize={9} />
                            <YAxis stroke="#64748b" fontSize={9} />
                            <Tooltip contentStyle={{ fontSize: '11px', background: 'rgba(15,23,42,0.92)', borderColor: '#334155' }} />
                            <Legend wrapperStyle={{ fontSize: '9px' }} />
                            <Line type="monotone" dataKey="trainingLoss" stroke="#6366f1" activeDot={{ r: 6 }} strokeWidth={2} name="Real Train Loss" />
                          </RechartsLineChart>
                        </ResponsiveContainer>
                      </div>

                      <div className="bg-slate-950 p-3 rounded-xl border border-slate-850 space-y-1 font-mono text-[10.5px]">
                        <div className="flex justify-between text-slate-500">
                          <span>Iterations Run:</span>
                          <span className="text-white font-bold">{mlResult.deepLearning.nIterations}</span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>Initial Training Loss:</span>
                          <span>{mlResult.deepLearning.trainingLogs[0]?.trainingLoss.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>Final Training Loss:</span>
                          <span className="text-white font-bold">{mlResult.deepLearning.trainingLogs[mlResult.deepLearning.trainingLogs.length - 1]?.trainingLoss.toFixed(4)}</span>
                        </div>
                        {mlResult.deepLearning.finalValidationScore !== undefined && mlResult.deepLearning.finalValidationScore !== null && (
                          <div className="flex justify-between text-emerald-400 pt-1 border-t border-slate-850">
                            <span>Final Validation {mlResult.deepLearning.validationMetric === 'accuracy' ? 'Accuracy' : 'R²'}:</span>
                            <span className="font-bold">
                              {mlResult.deepLearning.validationMetric === 'accuracy'
                                ? `${(mlResult.deepLearning.finalValidationScore * 100).toFixed(1)}%`
                                : mlResult.deepLearning.finalValidationScore.toFixed(3)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* 2. Real network architecture summary */}
                    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl col-span-2 space-y-4 shadow-xl">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <div>
                          <h4 className="font-extrabold text-white text-sm">Real Network Architecture</h4>
                          <p className="text-[10px] text-slate-405 text-slate-400 font-mono mt-0.5 font-bold">Actual scikit-learn MLP hidden layer sizes and trainable parameter count</p>
                        </div>
                        <Workflow className="w-4.5 h-4.5 text-indigo-400 shrink-0" />
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-[10.5px]">
                        <div className="bg-slate-950 p-4 rounded-xl border border-slate-850 space-y-1.5 text-center">
                          <span className="text-[9px] text-slate-450 font-bold uppercase tracking-wider block">Input → Hidden Layers</span>
                          <span className="font-extrabold text-indigo-400 text-sm">{mlResult.deepLearning.hiddenLayerSizes.join(' → ')}</span>
                        </div>
                        <div className="bg-slate-950 p-4 rounded-xl border border-slate-850 space-y-1.5 text-center">
                          <span className="text-[9px] text-slate-450 font-bold uppercase tracking-wider block">Total Layers</span>
                          <span className="font-extrabold text-white text-sm">{mlResult.deepLearning.nLayers}</span>
                        </div>
                        <div className="bg-slate-950 p-4 rounded-xl border border-slate-850 space-y-1.5 text-center">
                          <span className="text-[9px] text-slate-450 font-bold uppercase tracking-wider block">Trainable Parameters</span>
                          <span className="font-extrabold text-white text-sm">{mlResult.deepLearning.totalTrainableParams.toLocaleString()}</span>
                        </div>
                      </div>

                      <p className="text-[10px] text-slate-500 leading-relaxed font-sans pt-2 border-t border-slate-850">
                        Trained with early stopping enabled — a real 10% internal validation split is used to halt training once performance plateaus, rather than always running the full iteration budget.
                      </p>
                    </div>
                  </>
                ) : (
                  <div className="col-span-3 p-6 text-center bg-slate-950/50 border border-slate-850 rounded-xl text-[11px] text-slate-400 flex flex-col items-center gap-2">
                    <Info className="w-5 h-5 text-slate-500" />
                    Real training-loss curves are shown when a Multi-Layer Perceptron is trained. Current champion: <strong className="text-slate-200">{mlResult.modelAlgorithm}</strong>. Select "Multi-Layer Perceptron" from the algorithm dropdown and retrain.
                  </div>
                )}
              </div>
            )}

            {/* REALM E: COMPARISON DASHBOARD */}
            {activeRealm === 'comparison' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-fade-in text-xs">
                {/* Winner Card */}
                <div className="lg:col-span-1 space-y-6">
                  <div className="bg-emerald-900/10 backdrop-blur-md border border-emerald-500/30 p-6 rounded-2xl relative overflow-hidden flex flex-col hover:border-emerald-500/50 transition-colors shadow-[0_0_30px_rgba(16,185,129,0.05)]">
                    <div className="absolute top-0 right-0 p-4 opacity-10 filter blur-[2px]">
                      <Trophy className="w-32 h-32 text-emerald-400" />
                    </div>
                    <div className="relative z-10 space-y-4 flex-1">
                      <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-500/20 text-emerald-300 font-bold font-mono text-[10px] rounded-full border border-emerald-500/30 shadow-sm">
                        <CheckCircle className="w-3.5 h-3.5" /> REAL CHAMPION MODEL
                      </div>
                      <div className="space-y-1 mt-6">
                        <h3 className="text-xl font-bold text-white font-display tracking-wide">{mlResult.modelAlgorithm}</h3>
                        <p className="text-emerald-400/80 text-xs font-mono tracking-tight">Evaluated on held-out test data</p>
                      </div>

                      {mlResult.comparison && mlResult.comparison[0] && (
                        <div className="pt-6 mt-8 border-t border-emerald-500/20">
                          <p className="text-[10px] text-emerald-500/70 font-bold uppercase tracking-wider mb-1">Winning {mlResult.comparison[0].primaryMetric}</p>
                          <p className="text-3xl font-black text-emerald-400 font-mono tracking-tighter">
                            {mlResult.comparison[0].primaryMetric.toLowerCase().includes('accuracy')
                              ? `${(mlResult.comparison[0].metricValue * 100).toFixed(2)}%`
                              : mlResult.comparison[0].metricValue.toFixed(3)}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Score comparison grid */}
                <div className="lg:col-span-2 bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl flex flex-col shadow-lg overflow-hidden">
                  <h4 className="text-[11px] font-bold text-slate-300 font-mono tracking-wider uppercase mb-6 flex justify-between items-center bg-slate-950/40 p-3 rounded-xl border border-slate-800">
                    <span className="flex items-center gap-2"><Sparkles className="w-4 h-4 text-indigo-400" /> Real Side-by-Side Algorithm Comparison</span>
                    <span className="text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-md border border-indigo-500/20 text-[10px] tracking-widest">
                      EVAL MATRIX
                    </span>
                  </h4>

                  <div className="flex-1 overflow-x-auto pb-4 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
                    <div className="flex gap-4 h-full min-w-max">
                      {(mlResult.comparison || []).map((c, idx) => {
                        const isWinner = idx === 0;
                        return (
                          <div key={c.modelKey} className={`w-[260px] rounded-xl border p-5 flex flex-col justify-between shrink-0 relative overflow-hidden transition-all duration-300 ${isWinner ? 'bg-emerald-950/20 border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.05)]' : 'bg-slate-950/40 border-slate-800 hover:border-indigo-500/30'}`}>

                            {isWinner && (
                              <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/10 blur-2xl rounded-full translate-x-1/2 -translate-y-1/2 pointer-events-none" />
                            )}

                            <div>
                              <div className="flex items-center justify-between mb-4">
                                <span className={`text-[10px] font-bold font-mono tracking-wider px-2.5 py-1 rounded-full border ${isWinner ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-slate-800/50 text-slate-400 border-slate-700'}`}>
                                  RANK #{idx + 1}
                                </span>
                                {isWinner && <CheckCircle className="w-4 h-4 text-emerald-400 drop-shadow-[0_0_5px_rgba(52,211,153,0.5)] animate-pulse" />}
                              </div>

                              <h3 className={`text-base font-bold font-display tracking-wide leading-tight mb-1 ${isWinner ? 'text-white' : 'text-slate-200'}`}>
                                {c.modelName}
                              </h3>
                              <p className="text-[11px] text-slate-500 font-mono tracking-tight mb-6">
                                Real held-out evaluation
                              </p>

                              <div className="space-y-4">
                                <div>
                                  <p className="text-[9px] uppercase tracking-widest text-slate-500 font-bold mb-1">{c.primaryMetric}</p>
                                  <p className={`text-2xl font-black font-mono tracking-tighter ${isWinner ? 'text-emerald-400' : 'text-indigo-300'}`}>
                                    {c.primaryMetric.toLowerCase().includes('accuracy') ? `${(c.metricValue * 100).toFixed(2)}%` : c.metricValue.toFixed(3)}
                                  </p>
                                </div>

                                <div className="h-px w-full bg-slate-800/60" />

                                <div className="flex items-center justify-between">
                                  <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Execution Time</span>
                                  <span className="text-[11.5px] font-mono text-slate-300">{c.executionTimeMs.toLocaleString()} ms</span>
                                </div>
                              </div>
                            </div>

                            <div className={`mt-6 pt-4 border-t ${isWinner ? 'border-emerald-500/20' : 'border-slate-800/60'}`}>
                              <p className={`text-[10px] leading-relaxed font-medium ${isWinner ? 'text-emerald-500/80' : 'text-slate-500'}`}>
                                {isWinner ? 'Highest real test-set score among the models trained this run.' : 'Also trained and evaluated on the identical held-out split.'}
                              </p>
                            </div>
                          </div>
                        );
                      })}
                      {(!mlResult.comparison || mlResult.comparison.length <= 1) && (
                        <div className="text-slate-500 text-[11px] p-6">Only one model was trained. Pick "AUTO" from the algorithm dropdown to compare multiple real models side by side.</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* REALM F: AGENTIC ANALYSIS — reason -> act -> observe loop against real ml-service functions */}
            {activeRealm === 'agent' && (
              mlResult.comparison && mlResult.comparison[0] ? (
                <AgentAnalysis
                  dataset={dataset}
                  target={target}
                  features={selectedFeatures}
                  championModelKey={mlResult.comparison[0].modelKey}
                  primaryMetric={mlResult.comparison[0].primaryMetric}
                  primaryMetricValue={mlResult.comparison[0].metricValue}
                />
              ) : (
                <div className="p-6 text-center bg-slate-950/50 border border-slate-850 rounded-xl text-[11px] text-slate-400">
                  Train a model first to unlock agentic analysis.
                </div>
              )
            )}

            {/* MODEL EXPLAINER MODULE — narrates the real feature importances above */}
            <ModelExplainer result={mlResult} />

            {/* MLOps Dashboard */}
            <MLOpsDashboard dataset={dataset} target={target} features={selectedFeatures} />

          </div>

        </div>
      )}

    </div>
  );
}
