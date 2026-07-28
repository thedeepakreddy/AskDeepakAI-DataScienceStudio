/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  AreaChart,
  Eye,
  Compass,
  TrendingUp,
  AlertTriangle,
  Lightbulb,
  BrainCircuit,
  Sparkles,
  Table,
  Binary,
  FileQuestion,
  Layers,
  Percent,
  Dot,
  CheckCircle2,
  Info,
  Calendar,
  ToggleLeft,
  Database
} from 'lucide-react';
import { Dataset, DatasetColumn, EdaReport as EdaReportType } from '../types';
import {
  calculateFullNumericSummaries,
  mapAllMissingValues,
  detectDuplicateRecords,
  findExtremeOutliers,
  computeCorrelationMatrix,
  computeCategoricalFrequencies,
  NumericSummary,
  MissingValueDetails,
  OutlierDetail,
  CategoricalValue
} from '../utils/edaEngine';
import HypothesisLab from './HypothesisLab';
import ABTestInterpreter from './ABTestInterpreter';
import SQLAssistant from './SQLAssistant';
import { usePipelineContext } from '../contexts/PipelineContext';

interface EDAReportProps {
  dataset: Dataset;
  aiAnalysis: {
    overviewSummary?: string;
    recommendedTarget?: string;
    modelType?: string;
    suggestedFeatures?: string[];
    scientistFocus?: string;
    scientistRationale?: string;
    strategicSlicer?: string;
    insights?: string[];
  } | null;
  loadingAI: boolean;
  onTriggerAI: () => void;
}

export default function EDAReport({ dataset, aiAnalysis, loadingAI, onTriggerAI }: EDAReportProps) {
  const { expertMode } = usePipelineContext();
  // State for Sub-Workspaces
  const [activeSubTab, setActiveSubTab] = useState<'overview' | 'summaries' | 'integrity' | 'outliers' | 'correlation' | 'distributions' | 'categorical' | 'hypothesis' | 'abtest' | 'sql' | 'deepeda'>('overview');

  // Real server-side EDA (Python/pandas) — computed on-demand since it hits the network.
  const [deepEdaResult, setDeepEdaResult] = useState<EdaReportType | null>(null);
  const [loadingDeepEda, setLoadingDeepEda] = useState<boolean>(false);
  const [deepEdaError, setDeepEdaError] = useState<string | null>(null);

  const runDeepEda = async () => {
    setLoadingDeepEda(true);
    setDeepEdaError(null);
    try {
      const res = await fetch('/api/eda', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetRows: dataset.rows })
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || body.hint || 'EDA request failed.');
      setDeepEdaResult(body);
    } catch (err: any) {
      setDeepEdaError(err.message || 'Real EDA computation failed — is the ML compute service running?');
    } finally {
      setLoadingDeepEda(false);
    }
  };

  
  // Interactive distribution selections
  const numericColsObj = dataset.columns.filter(c => c.type === 'numeric');
  const [selectedNumCol, setSelectedNumCol] = useState<string>(numericColsObj[0]?.name || '');
  
  const categoricalColsObj = dataset.columns.filter(c => c.type === 'categorical' || c.type === 'boolean');
  const [selectedCatCol, setSelectedCatCol] = useState<string>(categoricalColsObj[0]?.name || '');

  // Detailed coordinate list viewing toggles
  const [expandedMissingCol, setExpandedMissingCol] = useState<string | null>(null);
  const [expandedOutlierCol, setExpandedOutlierCol] = useState<string | null>(null);

  // Run analytic calculations
  const numSummaries = React.useMemo(() => calculateFullNumericSummaries(dataset.rows, dataset.columns), [dataset]);
  const missingDetails = React.useMemo(() => mapAllMissingValues(dataset.rows, dataset.columns), [dataset]);
  const duplicates = React.useMemo(() => detectDuplicateRecords(dataset.rows), [dataset]);
  const outlierDetails = React.useMemo(() => findExtremeOutliers(dataset.rows, dataset.columns), [dataset]);
  const corrMatrix = React.useMemo(() => computeCorrelationMatrix(dataset.rows, dataset.columns), [dataset]);

  // Compute total missing count across entire workspace
  const totalMissingCells = missingDetails.reduce((sum, col) => sum + col.count, 0);
  const totalCellsCount = dataset.rowCount * dataset.columns.length;
  const overallMissingnessPercent = totalCellsCount > 0 ? parseFloat(((totalMissingCells / totalCellsCount) * 100).toFixed(2)) : 0;

  // Compute total outliers count
  const totalOutliersCount = outlierDetails.reduce((sum, col) => sum + col.outliers.length, 0);

  // Helper for computing active numerical values of selected column
  const activeColValues = selectedNumCol 
    ? dataset.rows.map(r => r[selectedNumCol]).filter((v): v is number => typeof v === 'number' && !isNaN(v) && v !== null)
    : [];

  // Helper to calculate Histogram bins
  const histogramBinsCount = 10;
  const histogramData = selectedNumCol && activeColValues.length > 0
    ? (() => {
        const min = activeColValues.reduce((a, b) => (a < b ? a : b), activeColValues[0]);
        const max = activeColValues.reduce((a, b) => (a > b ? a : b), activeColValues[0]);
        const range = max - min;
        
        if (range === 0) {
          return [{
            binStart: min,
            binEnd: max,
            label: `${min.toFixed(2)}`,
            count: activeColValues.length,
            percentage: 100,
            relativeHeight: 1
          }];
        }

        const binSize = range / histogramBinsCount;
        const bins = Array.from({ length: histogramBinsCount }, (_, i) => {
          const start = min + i * binSize;
          const end = start + binSize;
          return {
            binStart: start,
            binEnd: end,
            label: `${start.toFixed(2)} - ${end.toFixed(2)}`,
            count: 0
          };
        });

        activeColValues.forEach(v => {
          let index = Math.floor((v - min) / binSize);
          if (index >= histogramBinsCount) index = histogramBinsCount - 1;
          if (index < 0) index = 0;
          bins[index].count++;
        });

        const maxCount = Math.max(...bins.map(b => b.count), 1);
        return bins.map(b => ({
          ...b,
          percentage: parseFloat(((b.count / activeColValues.length) * 100).toFixed(2)),
          relativeHeight: b.count / maxCount
        }));
      })()
    : [];

  // Categorical Data Fetching
  const activeCatDistribution = selectedCatCol
    ? computeCategoricalFrequencies(dataset.rows, selectedCatCol)
    : [];

  return (
    <div className="space-y-6" id="eda_interactive_workspace">
      
      {/* 🚀 Consistent ML Pipeline stage header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-slate-900/40 backdrop-blur-md rounded-2xl border border-slate-800/80 p-5 shadow-2xl relative overflow-hidden">
        <div>
          <span className="text-[10px] font-bold text-indigo-400 tracking-widest font-mono uppercase">EXPLORATORY ANALYSIS</span>
          <h2 className="text-xl font-extrabold text-white tracking-tight mt-1">3. Automated Descriptive Statistics & Insights</h2>
          <p className="text-xs text-slate-405 text-slate-400 mt-1 max-w-xl">
            Audit standard deviations, trace anomalies, correlations, and invoke AskDeepakAI for AI-recommender target selections.
          </p>
        </div>
        <span className="bg-[#131B2E]/90 text-indigo-400 text-[10px] font-mono font-bold px-3 py-1 rounded-full border border-indigo-500/30 uppercase tracking-wide flex items-center gap-1.5 shadow-md shrink-0">
          <BrainCircuit className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
          ML Pipeline Stage: ACTIVE
        </span>
      </div>
      
      {/* 🧠 COGNITIVE COPILOT ASSISTANT PROMPT */}
      {!aiAnalysis && (
        <div className="relative bg-[#0E1325]/85 backdrop-blur-md text-white p-6 sm:p-8 rounded-2xl border border-indigo-500/20 shadow-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-6 overflow-hidden">
          {/* Ambient Decorative Orbit Glow */}
          <div className="absolute right-0 top-0 translate-x-12 -translate-y-12 w-64 h-64 bg-indigo-500/15 rounded-full blur-3xl pointer-events-none animate-pulse" />
          <div className="absolute left-0 bottom-0 -translate-x-12 translate-y-12 w-48 h-48 bg-teal-500/5 rounded-full blur-2xl pointer-events-none" />

          <div className="relative z-10 space-y-2.5">
            <span className="text-[9px] font-mono font-extrabold text-[#7e9dff] uppercase tracking-widest bg-indigo-500/15 px-3 py-1 rounded-full border border-indigo-500/25 font-mono">
              AI ASSISTANT ADVISOR
            </span>
            <h3 className="text-base sm:text-lg font-bold tracking-tight flex items-center gap-2 mt-1 font-sans">
              <Sparkles className="w-5 h-5 text-indigo-400 animate-pulse" /> AskDeepakAI Strategic Insights
            </h3>
            <p className="text-xs text-slate-300 max-w-xl leading-relaxed font-sans">
              Activate automated AI diagnostics. Analyzes column patterns, profiles key business metrics, and offers high-accuracy target predictor recommendations using AskDeepakAI.
            </p>
          </div>
          <button
            onClick={onTriggerAI}
            disabled={loadingAI}
            className="relative z-10 bg-indigo-600 hover:bg-indigo-500 text-white font-extrabold text-xs py-3 px-6 rounded-xl flex items-center justify-center gap-2.5 transition-all duration-300 shadow-lg shadow-indigo-600/40 border-0 cursor-pointer disabled:opacity-75 disabled:cursor-not-allowed shrink-0"
          >
            {loadingAI ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Synthesizing coordinates...
              </>
            ) : (
              <>
                <BrainCircuit className="w-4.5 h-4.5 text-slate-100" />
                Run AskDeepakAI Advisory
              </>
            )}
          </button>
        </div>
      )}

      {/* 🧭 COGNITIVE SCAN SUMMARY PANELS - IF AI ANALYSIS HAS LOADED */}
      {aiAnalysis && (
        <div className="bg-gradient-to-r from-slate-900 to-indigo-950/40 rounded-2xl border border-indigo-550/15 p-5 sm:p-6 shadow-xl animate-fade-in duration-300 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 space-y-3.5">
            <div className="flex items-center gap-2">
              <span className="text-[9px] font-mono font-bold text-indigo-400 uppercase tracking-wider bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20 font-mono">
                AskDeepakAI Scan Result
              </span>
              <p className="text-[11px] text-slate-400 font-medium font-sans">Model recommends target column '{aiAnalysis.recommendedTarget}'</p>
            </div>
            <h4 className="text-white text-sm font-bold flex items-center gap-2 font-sans">
              <Compass className="w-4 h-4 text-indigo-400" /> Dataset Executive Summary
            </h4>
            <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-4 rounded-xl border border-slate-850 italic font-sans">
              "{aiAnalysis.overviewSummary}"
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
              <div className="bg-slate-950/30 p-3 rounded-lg border border-slate-850 space-y-1">
                <span className="text-[10px] uppercase font-mono font-bold text-amber-400">Primary Target Recommendation:</span>
                <p className="text-xs text-slate-200 mt-1 font-semibold font-sans">{aiAnalysis.scientistFocus}</p>
                <p className="text-[10px] text-slate-400 leading-relaxed mt-1 font-sans">{aiAnalysis.scientistRationale}</p>
              </div>
              <div className="bg-slate-950/30 p-3 rounded-lg border border-slate-850 space-y-1">
                <span className="text-[10px] uppercase font-mono font-bold text-emerald-400 font-mono">Recommended Category Slicing Field:</span>
                <p className="text-xs text-[#10b981] mt-1 font-semibold font-mono">{aiAnalysis.strategicSlicer}</p>
                <p className="text-[10px] text-slate-400 leading-relaxed mt-1 font-sans">Dividing your data cohorts along this column yields clear separation and key performance insights.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-950/50 border border-slate-850 p-4 rounded-xl flex flex-col justify-between">
            <div className="space-y-4">
              <span className="text-[9px] font-mono font-extrabold text-indigo-300 bg-indigo-500/15 px-2.5 py-1 rounded-md tracking-wider">
                COGNITIVE PATHWAY
              </span>
              <div className="space-y-2 mt-2">
                <p className="text-xs text-slate-300 font-medium">Advised Target Variable:</p>
                <p className="text-lg font-bold text-white font-mono uppercase bg-indigo-650/10 border border-indigo-500/25 p-2 rounded-lg text-center tracking-tight">
                  {aiAnalysis.recommendedTarget || 'Unspecified'}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                  <span className="text-[9px] uppercase font-mono text-slate-400 block">Model Style</span>
                  <span className="text-[11px] font-bold text-white font-mono uppercase">{aiAnalysis.modelType || 'N/A'}</span>
                </div>
                <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                  <span className="text-[9px] uppercase font-mono text-slate-400 block">Input Attributes</span>
                  <span className="text-[11px] font-bold text-white font-mono">{(aiAnalysis.suggestedFeatures || []).length} cols</span>
                </div>
              </div>
            </div>
            
            <div className="pt-4 border-t border-slate-850 text-[11px] text-slate-400 flex flex-wrap gap-1.5 items-center">
              <span className="font-semibold text-slate-305">Suggested:</span>
              <span className="font-mono text-indigo-400">{(aiAnalysis.suggestedFeatures || []).slice(0,3).join(', ')}...</span>
            </div>
          </div>
        </div>
      )}

      {/* 🎛️ SUB WORKSTATION NAV RAIL */}
      <div className="flex flex-wrap gap-2 border-b border-slate-800/80 pb-1" id="eda_tabs_nav">
        {[
          { id: 'overview', label: 'Structure Overview', icon: Table },
          { id: 'summaries', label: 'Summary Statistics', icon: Binary },
          { id: 'integrity', label: 'Data Integrity Mapping', icon: AlertTriangle },
          { id: 'deepeda', label: 'Deep EDA (Real, Server-Side)', icon: Database },
          ...(expertMode ? [
            { id: 'sql', label: 'SQL Assistant', icon: Database },
            { id: 'outliers', label: 'IQR Outlier Scan', icon: Percent },
            { id: 'correlation', label: 'Feature Correlation Matrix', icon: AreaChart },
            { id: 'distributions', label: 'Numeric Distributions', icon: TrendingUp },
            { id: 'categorical', label: 'Categorical Distributions', icon: Layers },
            { id: 'hypothesis', label: 'Hypothesis Lab', icon: Lightbulb },
            { id: 'abtest', label: 'A/B Test Interpreter', icon: Sparkles }
          ] : [])
        ].map(tab => {
          const TabIcon = tab.icon;
          const isSelected = activeSubTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold select-none cursor-pointer border transition-all duration-300 ${
                isSelected
                  ? 'bg-indigo-600/15 border-indigo-500/40 text-indigo-300'
                  : 'bg-transparent border-transparent text-slate-400 hover:text-slate-205 hover:bg-slate-900/30'
              }`}
            >
              <TabIcon className="w-4 h-4 shrink-0" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* 🚀 WORKSPACE CONTENT DECIDER */}
      <div className="min-h-[400px]">

        {/* 1. DATA STRUCTURE OVERVIEW SUBTAB */}
        {activeSubTab === 'overview' && (
          <div className="space-y-6 animate-fade-in">
            {/* Meta Statistics Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-900/45 border border-slate-850 p-4 rounded-xl space-y-1 flex flex-col justify-between">
                <span className="text-[10px] tracking-wider uppercase font-mono font-bold text-slate-400">Row Count</span>
                <p className="text-2xl font-black text-white font-mono mt-1">{dataset.rowCount.toLocaleString()}</p>
                <span className="text-[10px] text-slate-500">Total observed instances in worksheet</span>
              </div>
              <div className="bg-slate-900/45 border border-slate-850 p-4 rounded-xl space-y-1 flex flex-col justify-between">
                <span className="text-[10px] tracking-wider uppercase font-mono font-bold text-slate-400">Column Count</span>
                <p className="text-2xl font-black text-white font-mono mt-1">{dataset.columns.length}</p>
                <span className="text-[10px] text-slate-500">Calculated features & inputs</span>
              </div>
              <div className="bg-slate-900/45 border border-slate-850 p-4 rounded-xl space-y-1 flex flex-col justify-between">
                <span className="text-[10px] tracking-wider uppercase font-mono font-bold text-slate-400">Missingness Density</span>
                <p className="text-2xl font-black font-mono mt-1 text-amber-400">{overallMissingnessPercent}%</p>
                <span className="text-[10px] text-slate-500">({totalMissingCells.toLocaleString()} empty values map)</span>
              </div>
              <div className="bg-slate-900/45 border border-slate-850 p-4 rounded-xl space-y-1 flex flex-col justify-between">
                <span className="text-[10px] tracking-wider uppercase font-mono font-bold text-slate-400">Duplicate Record Clusters</span>
                <p className={`text-2xl font-black font-mono mt-1 ${duplicates.length > 0 ? 'text-rose-455' : 'text-emerald-400'}`}>
                  {duplicates.length} {duplicates.length === 1 ? 'group' : 'groups'}
                </p>
                <span className="text-[10px] text-slate-500">Identical overlapping data rows</span>
              </div>
            </div>

            {/* Column Schema Details Table */}
            <div className="bg-slate-900/40 rounded-xl border border-slate-800 p-5 overflow-hidden shadow-lg">
              <h4 className="text-white text-sm font-bold flex items-center gap-2 mb-4">
                <Table className="w-4 h-4 text-indigo-400" /> Data Columns & Properties
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-mono text-[10px]">
                      <th className="py-2 px-3">Column Index</th>
                      <th className="py-2 px-3">Variable Name</th>
                      <th className="py-2 px-3">Data Type</th>
                      <th className="py-2 px-3">Distinct Values</th>
                      <th className="py-2 px-3">Missing Rows</th>
                      <th className="py-2 px-3 text-right">Completeness Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-850">
                    {dataset.columns.map((col, idx) => {
                      const missingPercent = parseFloat(((col.missingCount / dataset.rowCount) * 100).toFixed(1));
                      const completenessPercent = 100 - missingPercent;
                      return (
                        <tr key={idx} className="hover:bg-slate-800/10 text-slate-300">
                          <td className="py-3 px-3 font-mono text-slate-500">#{idx + 1}</td>
                          <td className="py-3 px-3 font-mono font-bold text-white max-w-[200px] truncate">{col.name}</td>
                          <td className="py-3 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase font-semibold ${
                              col.type === 'numeric' ? 'bg-blue-500/10 text-blue-450 border border-blue-500/20' :
                              col.type === 'categorical' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                              col.type === 'boolean' ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20' :
                              'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                            }`}>
                              {col.type}
                            </span>
                          </td>
                          <td className="py-3 px-3 font-mono">{col.distinctCount.toLocaleString()}</td>
                          <td className="py-3 px-3 font-mono text-slate-400">{col.missingCount.toLocaleString()} <span className="text-[10px] text-slate-500">({missingPercent}%)</span></td>
                          <td className="py-3 px-3 text-right">
                            <div className="flex items-center justify-end gap-2.5 font-mono">
                              <span className="text-slate-200">{completenessPercent.toFixed(1)}%</span>
                              <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden inline-block border border-slate-700/30">
                                <div 
                                  className={`h-full rounded-full ${completenessPercent === 100 ? 'bg-emerald-500' : completenessPercent > 80 ? 'bg-indigo-550' : 'bg-amber-500'}`}
                                  style={{ width: `${completenessPercent}%` }}
                                />
                              </div>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 2. COMPREHENSIVE SUMMARY STATISTICS SUBTAB */}
        {activeSubTab === 'summaries' && (
          <div className="space-y-6 animate-fade-in">
            <div className="bg-slate-900/40 rounded-xl border border-slate-800 p-5 shadow-lg space-y-4">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                <div>
                  <h4 className="text-white text-sm font-bold flex items-center gap-2">
                    <Binary className="w-4.5 h-4.5 text-indigo-400" /> Numerical Summary Stats
                  </h4>
                  <p className="text-xs text-slate-450 mt-1">
                    Complete statistical distributions detailing central tendencies, variation coefficients, standard deviation, and quartile limits.
                  </p>
                </div>
                <div className="bg-indigo-500/10 border border-indigo-500/20 px-3 py-1.5 rounded-lg text-[10px] font-mono text-indigo-300">
                  Calculated values are double-precision floating points
                </div>
              </div>

              {numSummaries.length === 0 ? (
                <div className="text-center p-8 text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
                  No numerical columns found in the active dataset. Change type or upload numeric inputs.
                </div>
              ) : (
                <div className="overflow-x-auto select-text">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-mono text-[10px]">
                        <th className="py-3 px-3">Variable</th>
                        <th className="py-3 px-3 text-right">Observation Count</th>
                        <th className="py-3 px-3 text-right">Minimum Value</th>
                        <th className="py-3 px-3 text-right">Quartile 1 (25%)</th>
                        <th className="py-3 px-3 text-right">Median (50%)</th>
                        <th className="py-3 px-3 text-right">Quartile 3 (75%)</th>
                        <th className="py-3 px-3 text-right">Maximum Value</th>
                        <th className="py-3 px-3 text-right">Mean Average (μ)</th>
                        <th className="py-3 px-3 text-right">Standard Deviation (σ)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850">
                      {numSummaries.map((s, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/10 text-slate-300">
                          <td className="py-3.5 px-3 font-mono font-bold text-white truncate max-w-[160px]">{s.columnName}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-slate-400">{s.count.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-emerald-400">{s.min.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-indigo-300">{s.q1.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-[#3bc8c8] font-semibold">{s.median.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-indigo-300">{s.q3.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-rose-455">{s.max.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-white font-medium">{s.mean.toLocaleString()}</td>
                          <td className="py-3.5 px-3 text-right font-mono text-slate-400">{s.stdDev.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            
            <div className="bg-slate-905 p-4 rounded-xl border border-slate-850 flex gap-3 text-xs leading-relaxed text-slate-400">
              <Info className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-slate-300">Data Science Definition Guide:</strong>
                <ul className="list-disc list-inside mt-1.5 space-y-1">
                  <li><strong className="text-slate-205">Standard Deviation (σ)</strong>: Measures the amount of variation or dispersion from the mean value. Lower indicates compact clustering.</li>
                  <li><strong className="text-slate-205">Quartiles (25%, 50%, 75%)</strong>: Splits the sorted values list into four equal clusters. Visualizes skewness and potential outlier limits.</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* 3. MISSING & DUPLICATES DATA INTEGRITY SCAN SUBTAB */}
        {activeSubTab === 'integrity' && (
          <div className="space-y-6 animate-fade-in text-xs">
            
            {/* Missing Values Detail Block */}
            <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-xl shadow-lg space-y-4">
              <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                <h4 className="text-white text-sm font-bold flex items-center gap-2">
                  <Percent className="w-4 h-4 text-indigo-400" /> Missing Values & Locations
                </h4>
                <span className="bg-indigo-500/10 border border-indigo-500/25 text-indigo-300 px-2 py-0.5 rounded font-mono font-bold text-[10px]">
                  {totalMissingCells.toLocaleString()} TOTAL NULL COORD
                </span>
              </div>

              <p className="text-slate-400">
                This table explicitly lists exact indices where nulls are observed. Locate missing records precisely before pipeline cleansing transformations.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {missingDetails.map((col, idx) => {
                  const isExpanded = expandedMissingCol === col.columnName;
                  return (
                    <div key={idx} className="bg-slate-950/40 border border-slate-850 p-4 rounded-xl space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="font-mono font-bold text-white text-[12px]">{col.columnName}</span>
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-mono ${col.count > 0 ? 'bg-amber-500/15 text-amber-400 border border-amber-500/20' : 'bg-emerald-500/10 text-emerald-400'}`}>
                            {col.count} {col.count === 1 ? 'Null Row' : 'Null Rows'} ({col.percent}%)
                          </span>
                        </div>
                      </div>

                      {col.count === 0 ? (
                        <div className="text-emerald-450 font-semibold font-mono text-[10px] flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" /> Integrity perfect. 100% complete
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono">
                            <span>Coordinate Index Array:</span>
                            <button
                              onClick={() => setExpandedMissingCol(isExpanded ? null : col.columnName)}
                              className="text-indigo-400 hover:text-indigo-300 underline font-semibold bg-transparent border-0 cursor-pointer p-0"
                            >
                              {isExpanded ? 'Collapse' : `Show all ${col.count} coordinates`}
                            </button>
                          </div>
                          
                          <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto bg-slate-900/60 p-2 rounded-lg border border-slate-850">
                            {col.locations.slice(0, isExpanded ? undefined : 15).map((loc, lIdx) => (
                              <span key={lIdx} className="px-1.5 py-0.5 rounded bg-slate-950 text-slate-350 font-mono text-[10px] border border-slate-800">
                                Row {loc}
                              </span>
                            ))}
                            {!isExpanded && col.locations.length > 15 && (
                              <span className="px-2 py-0.5 text-[10px] text-slate-500 font-mono select-none">
                                + {col.locations.length - 15} more locations
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Duplicate Records Detail Block */}
            <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-xl shadow-lg space-y-4">
              <h4 className="text-white text-sm font-bold flex items-center gap-2 pb-2 border-b border-slate-800">
                <Layers className="w-4.5 h-4.5 text-rose-500" /> Row Uniqueness & Duplicates Checking
              </h4>
              
              <p className="text-slate-450">
                Data redundancy compromises evaluation accuracy. Below is a list of all identical clusters having overlapping inputs across every single column coordinate.
              </p>

              {duplicates.length === 0 ? (
                <div className="bg-[#10B981]/5 border border-[#10B981]/25 text-emerald-300 p-4 rounded-xl flex items-center gap-3 font-medium">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                  <span>Success: 0 duplicate records identified. Every row in this spreadsheet possesses distinct feature properties.</span>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="bg-rose-500/10 border border-rose-500/20 text-rose-300 p-4 rounded-xl font-medium">
                    Attention: We found <strong>{duplicates.length} duplicate groups</strong> in the active worksheet. Eliminating these redundant records is advised for robust ML training.
                  </div>

                  <div className="divide-y divide-slate-850 border border-slate-850 bg-slate-950/40 rounded-xl overflow-hidden">
                    {duplicates.map((group, idx) => (
                      <div key={idx} className="p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 hover:bg-slate-900/10">
                        <div className="space-y-1 flex-1">
                          <p className="font-mono text-white text-xs font-semibold">
                            Redundancy Group #{idx + 1}
                          </p>
                          <p className="text-[10px] font-mono text-slate-500 truncate max-w-xl">
                            Payload Signature: {group.rowString}
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-1 md:justify-end max-w-md">
                          <span className="text-[10px] uppercase font-mono tracking-wider font-bold text-rose-455 mr-2 self-center">
                            {group.indices.length} overlapping rows:
                          </span>
                          {group.indices.map((rNum, gIdx) => (
                            <span key={gIdx} className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 font-mono text-[10px] border border-slate-800 shadow">
                              Row {rNum}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

          </div>
        )}

        {/* 4. IQR OUTLIER SCAN SUBTAB */}
        {activeSubTab === 'outliers' && (
          <div className="space-y-6 animate-fade-in text-xs">
            <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-xl shadow-lg space-y-4">
              <div>
                <h4 className="text-white text-sm font-bold flex items-center gap-2">
                  <Percent className="w-4 h-4 text-indigo-400" /> Outlier Points Analysis
                </h4>
                <p className="text-slate-450 mt-1">
                  Using structural Tukey limits. Numeric limits mapped dynamically as: <code className="bg-slate-950/80 px-1.5 py-0.5 rounded text-indigo-300 font-mono text-[11px]">[ Q1 - 1.5 × IQR , Q3 + 1.5 × IQR ]</code>. Values falling outside are flagged.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-3">
                
                {/* Total Stats card */}
                <div className="bg-slate-950/60 p-5 rounded-xl border border-slate-850 flex flex-col justify-between space-y-4">
                  <div className="space-y-2">
                    <span className="text-[9px] uppercase font-mono font-bold tracking-widest text-[#9BB1FF]">AGGREGATE STATS</span>
                    <h5 className="font-bold text-white text-sm mt-1">Workspace Anomaly Summary</h5>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      We observed {totalOutliersCount} anomalous parameters out of {activeColValues.length * numericColsObj.length} numeric observations in total.
                    </p>
                  </div>
                  <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-800 text-center">
                    <p className="text-[10px] font-mono text-slate-500 uppercase">Extreme Outlier Points</p>
                    <p className={`text-4xl font-extrabold font-mono mt-1 ${totalOutliersCount > 0 ? 'text-amber-500' : 'text-emerald-400'}`}>
                      {totalOutliersCount}
                    </p>
                    <span className="text-[10px] text-slate-500">records require data checking</span>
                  </div>
                </div>

                {/* Columns Anomalytrend list */}
                <div className="lg:col-span-2 space-y-4">
                  <div className="overflow-x-auto border border-slate-850 rounded-xl overflow-hidden">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-850 bg-slate-950/40 text-slate-400 font-mono text-[9px] uppercase tracking-wider">
                          <th className="p-3">Variable Metric</th>
                          <th className="p-3 text-center">IQR Range</th>
                          <th className="p-3 text-center">Boundary limits</th>
                          <th className="p-3 text-center">Anomaly Count</th>
                          <th className="p-3 text-right">View Coordinates</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850 bg-slate-900/10">
                        {outlierDetails.map((det, idx) => {
                          const isExpanded = expandedOutlierCol === det.columnName;
                          const ratePercent = dataset.rowCount > 0 ? parseFloat(((det.outliers.length / dataset.rowCount) * 100).toFixed(1)) : 0;
                          return (
                            <React.Fragment key={idx}>
                              <tr className="hover:bg-slate-800/10 text-slate-300">
                                <td className="p-3 font-mono font-bold text-white truncate max-w-[140px]">{det.columnName}</td>
                                <td className="p-3 text-center font-mono text-slate-400">{det.iqr.toLocaleString()}</td>
                                <td className="p-3 text-center font-mono text-[10px] text-indigo-300">
                                  [{det.lowerBound.toLocaleString()} , {det.upperBound.toLocaleString()}]
                                </td>
                                <td className="p-3 text-center">
                                  <span className={`px-2 py-0.5 rounded-full font-mono text-[10px] font-bold ${
                                    det.outliers.length > 0 ? 'bg-amber-500/15 text-amber-500 border border-amber-500/20' : 'bg-emerald-500/10 text-emerald-400'
                                  }`}>
                                    {det.outliers.length} ({ratePercent}%)
                                  </span>
                                </td>
                                <td className="p-3 text-right">
                                  {det.outliers.length > 0 ? (
                                    <button
                                      onClick={() => setExpandedOutlierCol(isExpanded ? null : det.columnName)}
                                      className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 underline bg-transparent border-0 cursor-pointer"
                                    >
                                      {isExpanded ? 'Hide' : 'Expand'}
                                    </button>
                                  ) : (
                                    <span className="text-[10px] font-mono text-emerald-450 bg-emerald-500/10 px-2 py-0.5 rounded">
                                      Healthy Q
                                    </span>
                                  )}
                                </td>
                              </tr>
                              
                              {/* Inline Outliers Coordinates expanded drawer */}
                              {isExpanded && det.outliers.length > 0 && (
                                <tr>
                                  <td colSpan={5} className="p-3 bg-slate-950/80 border-t border-slate-850">
                                    <div className="space-y-2">
                                      <p className="text-[10px] font-mono text-[#9BB1FF] uppercase font-semibold">
                                        Anomalous Values Located (Row location & actual observation):
                                      </p>
                                      <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto p-1">
                                        {det.outliers.map((out, outIdx) => (
                                          <span key={outIdx} className="px-2.5 py-1 rounded bg-slate-900 border border-red-500/15 text-slate-300 font-mono text-[11px] flex items-center shadow">
                                            <span className="font-bold text-rose-455 mr-1.5">Row {out.rowIndex}:</span>
                                            <span className="text-white font-semibold">{out.value.toLocaleString()}</span>
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </React.Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}

        {/* 5. COVARIANCE & PEARSON CORRELATION MATRIX SUBTAB */}
        {activeSubTab === 'correlation' && (
          <div className="space-y-6 animate-fade-in text-xs">
            <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-xl shadow-lg space-y-4">
              <div>
                <h4 className="text-white text-sm font-bold flex items-center gap-2">
                  <AreaChart className="w-4.5 h-4.5 text-indigo-400" /> Metrics Correlations Strength
                </h4>
                <p className="text-slate-455 mt-1">
                  Covariance normalization mapped as numerical coefficients <code className="bg-slate-950/80 px-1 rounded text-teal-300 font-mono">r ∈ [-1, +1]</code>. Quantifies bidirectional linear feature associations.
                </p>
              </div>

              {corrMatrix.features.length < 2 ? (
                <div className="text-center p-8 text-xs text-slate-500 border border-dashed border-slate-850 rounded-xl">
                  Correlation matrix requires at least 2 numeric variables. Add other numeric properties in raw files or change category assignments under the cleaning tab.
                </div>
              ) : (
                <div className="space-y-6">
                  
                  {/* Graphical Heatmap */}
                  <div className="overflow-x-auto border border-slate-850 rounded-xl p-4 bg-slate-950/30">
                    <div className="min-w-[600px] flex flex-col items-center">
                      <p className="text-[10px] uppercase font-mono tracking-wider font-bold text-indigo-400 mb-6 bg-indigo-500/5 px-3 py-1 rounded-full border border-indigo-500/10">
                        Bidirectional Interactive Correlation Heatmap
                      </p>
                      
                      {/* Grid representation */}
                      <div className="flex flex-col gap-1 w-full max-w-3xl">
                        
                        {/* Header labels */}
                        <div className="flex text-[11px] font-mono font-bold text-slate-400 text-center pb-2">
                          <div className="w-1/4 text-left self-center text-[10px] text-slate-500 uppercase">Variable Y / X</div>
                          {corrMatrix.features.map(feat => (
                            <div key={feat} className="flex-1 truncate px-1 text-[10px] font-semibold" title={feat}>
                              {feat}
                            </div>
                          ))}
                        </div>

                        {/* Heated Rows */}
                        {corrMatrix.features.map(rowFeat => (
                          <div key={rowFeat} className="flex gap-1 text-center font-mono">
                            {/* Row selector label */}
                            <div className="w-1/4 text-[11px] font-bold text-white text-left self-center truncate pr-2" title={rowFeat}>
                              {rowFeat}
                            </div>

                            {/* Correlation cells */}
                            {corrMatrix.features.map(colFeat => {
                              const r = corrMatrix.matrix[rowFeat][colFeat];
                              const isSelf = rowFeat === colFeat;
                              
                              // Shading logic depending on positive or negative correlation strength
                              let cellStyle = 'bg-slate-900 border border-slate-850 text-slate-400 hover:scale-105 transition-all text-xs';
                              if (isSelf) {
                                cellStyle = 'bg-indigo-600 font-bold border border-indigo-550 text-white font-mono shadow';
                              } else if (r > 0.7) {
                                cellStyle = 'bg-[#10b981]/50 font-bold text-white border border-[#10b981]/60 hover:scale-[1.03] transition-all';
                              } else if (r > 0.4) {
                                cellStyle = 'bg-[#10b981]/25 text-emerald-200 border border-[#10b981]/30 hover:scale-[1.03] transition-all';
                              } else if (r > 0.1) {
                                cellStyle = 'bg-[#10b981]/10 text-emerald-350 border border-[#10b981]/15 hover:scale-[1.03] transition-all';
                              } else if (r < -0.7) {
                                cellStyle = 'bg-rose-500/50 font-bold text-white border border-rose-500/60 hover:scale-[1.03] transition-all';
                              } else if (r < -0.4) {
                                cellStyle = 'bg-rose-500/25 text-rose-200 border border-rose-500/30 hover:scale-[1.03] transition-all';
                              } else if (r < -0.1) {
                                cellStyle = 'bg-rose-500/10 text-rose-350 border border-rose-500/15 hover:scale-[1.03] transition-all';
                              }

                              return (
                                <div
                                  key={colFeat}
                                  className={`flex-1 py-3 text-[11px] font-mono rounded-md flex items-center justify-center select-none cursor-help ${cellStyle}`}
                                  title={`${rowFeat} ↔ ${colFeat}: r = ${r}`}
                                >
                                  {r > 0 && !isSelf ? `+${r.toFixed(3)}` : r.toFixed(3)}
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>

                      {/* Legends */}
                      <div className="flex gap-6 mt-6 justify-center text-[10px] font-mono text-slate-500 bg-[#0E121E] px-4 py-2 rounded-lg border border-slate-850">
                        <div className="flex items-center gap-1.5">
                          <span className="w-3.5 h-3.5 rounded bg-rose-500/50 block" />
                          <span>Strong Negative (r ≤ -0.7)</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="w-3.5 h-3.5 rounded bg-slate-900 border border-slate-850 block" />
                          <span>Near Neutral (|r| &lt; 0.1)</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="w-3.5 h-3.5 rounded bg-[#10b981]/50 block" />
                          <span>Strong Positive (r ≥ 0.7)</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Scatter plot Feature Relationship visualizer */}
                  <FeatureRelationshipScatter dataset={dataset} numericCols={corrMatrix.features} />

                </div>
              )}
            </div>
          </div>
        )}

        {/* 6. NUMERIC DISTRIBUTIONS DYNAMIC VISUALIZER (HISTOGRAM + BOXPLOT) */}
        {activeSubTab === 'distributions' && (
          <div className="space-y-6 animate-fade-in text-xs">
            <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-xl shadow-lg space-y-4">
              
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pb-2 border-b border-slate-800">
                <div>
                  <h4 className="text-white text-sm font-bold flex items-center gap-2">
                    <TrendingUp className="w-4.5 h-4.5 text-indigo-400" /> Range Distributions for Numerical Columns
                  </h4>
                  <p className="text-slate-455 mt-1">
                    Select a continuous variable from the drop-down tool to compile numerical frequencies, sample bins, and quartile boundaries.
                  </p>
                </div>
                
                {/* Column variable selector drops */}
                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-slate-500 font-bold uppercase tracking-wider">Select Variable:</span>
                  <select
                    value={selectedNumCol}
                    onChange={(e) => setSelectedNumCol(e.target.value)}
                    className="bg-[#0B0E17] border border-slate-750 text-slate-200 px-3 py-1.5 rounded-xl text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors cursor-pointer"
                  >
                    {numericColsObj.map(c => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              {!selectedNumCol ? (
                <div className="text-center p-8 text-xs text-slate-500 border border-dashed border-slate-850 rounded-xl">
                  No numeric columns available to compile histograms. Convert column parameters.
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-2">
                  
                  {/* Left Column stats report */}
                  <div className="bg-slate-950/50 p-5 rounded-xl border border-slate-850 space-y-4">
                    <span className="text-[9px] uppercase font-mono font-bold text-[#3bc8c8] bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 rounded">
                      Variable Stats
                    </span>
                    <h5 className="font-extrabold text-white text-base font-mono truncate">{selectedNumCol}</h5>
                    
                    {(() => {
                      const sumStats = numSummaries.find(s => s.columnName === selectedNumCol);
                      if (!sumStats) return null;
                      return (
                        <div className="space-y-3 pt-2 font-mono">
                          <div className="flex justify-between border-b border-slate-850 pb-2.5">
                            <span className="text-slate-400 text-[11px]">Mean Average</span>
                            <span className="text-white font-bold">{sumStats.mean.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between border-b border-slate-850 pb-2.5">
                            <span className="text-slate-400 text-[11px]">Std Deviation (σ)</span>
                            <span className="text-slate-350">{sumStats.stdDev.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between border-b border-slate-850 pb-2.5">
                            <span className="text-slate-400 text-[11px]">Median Value</span>
                            <span className="text-white font-bold">{sumStats.median.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between border-b border-slate-850 pb-2.5">
                            <span className="text-slate-400 text-[11px]">25th Percentile (Q1)</span>
                            <span className="text-indigo-400 font-semibold">{sumStats.q1.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between border-b border-slate-850 pb-2.5">
                            <span className="text-slate-400 text-[11px]">75th Percentile (Q3)</span>
                            <span className="text-indigo-400 font-semibold">{sumStats.q3.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between text-[11px] text-slate-500 pt-1.5">
                            <span>Min: {sumStats.min.toLocaleString()}</span>
                            <span>Max: {sumStats.max.toLocaleString()}</span>
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                  {/* Histogram and Box plot column view */}
                  <div className="lg:col-span-2 space-y-6">
                    
                    {/* SVG HISTOGRAM PROBABILITY BINS */}
                    <div className="bg-slate-950/30 p-4 rounded-xl border border-slate-850 shadow-inner">
                      <p className="text-[10px] uppercase font-mono tracking-wider font-bold text-indigo-400 mb-4 text-center">
                        Frequency Distribution Histogram (10 Normalized Bins)
                      </p>
                      
                      <div className="h-60 w-full flex items-end gap-1.5 pt-4 px-2 select-none relative">
                        
                        {/* Normalized baseline axis */}
                        <div className="absolute left-0 bottom-0 w-full h-[1px] bg-slate-700" />
                        
                        {histogramData.map((bin, bIdx) => (
                          <div
                            key={bIdx}
                            className="flex-1 flex flex-col justify-end items-center group relative cursor-help"
                            style={{ height: '100%' }}
                          >
                            {/* Hover Coordinate Info Card Tooltips */}
                            <div className="absolute bottom-full mb-3 hidden group-hover:block bg-slate-900 border border-slate-700 p-2.5 rounded-lg shadow-xl z-10 w-44 pointer-events-none text-left">
                              <p className="font-mono text-indigo-300 text-[10px] uppercase font-bold">Bin Bounds:</p>
                              <p className="text-[11px] text-white font-mono mt-0.5">{bin.label}</p>
                              <p className="font-mono text-emerald-450 text-[10px] uppercase font-bold mt-2">Row Count / %:</p>
                              <p className="text-[11px] text-white font-mono mt-0.5">{bin.count.toLocaleString()} rows ({bin.percentage}%)</p>
                            </div>

                            {/* Actual Histogram bar */}
                            <div
                              className="w-full bg-gradient-to-t from-indigo-700/80 to-[#3bc8c8]/95 hover:brightness-125 transition-all duration-300 rounded-t border-t border-teal-400/20"
                              style={{ height: `${Math.max(4, bin.relativeHeight * 85)}%` }}
                            />

                            {/* Small Count marker above bar if hover */}
                            <span className="text-[9px] font-mono text-slate-500 mt-1 opacity-0 group-hover:opacity-100 transition-opacity absolute bottom-[5%] bg-slate-950 px-1 py-0.5 rounded border border-slate-800">
                              {bin.count}
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* X and Y Axis helpers */}
                      <div className="flex justify-between items-center text-[9px] font-mono text-slate-500 mt-2 px-2.5 border-t border-slate-850 pt-1.5">
                        <span>MIN Bounds: {activeColValues.length > 0 ? activeColValues.reduce((a, b) => (a < b ? a : b)).toLocaleString() : '0'}</span>
                        <span>Normalized Continuous Coordinates</span>
                        <span>MAX Bounds: {activeColValues.length > 0 ? activeColValues.reduce((a, b) => (a > b ? a : b)).toLocaleString() : '0'}</span>
                      </div>
                    </div>

                    {/* SVG QUARTILE BOX PLOT DRAW */}
                    <div className="bg-slate-950/30 p-4 rounded-xl border border-slate-850 shadow-inner space-y-4">
                      <p className="text-[10px] uppercase font-mono tracking-wider font-bold text-indigo-400 text-center">
                        Tukey Quartile Box & Whisker Anomaly Plot
                      </p>

                      {(() => {
                        const sumStats = numSummaries.find(s => s.columnName === selectedNumCol);
                        if (!sumStats || sumStats.min === sumStats.max) return null;
                        
                        const min = sumStats.min;
                        const max = sumStats.max;
                        const range = max - min;
                        
                        // Percentage map function
                        const getPct = (val: number) => ((val - min) / range) * 100;
                        
                        const q1Pct = getPct(sumStats.q1);
                        const medianPct = getPct(sumStats.median);
                        const q3Pct = getPct(sumStats.q3);
                        
                        return (
                          <div className="pt-6 pb-4 px-4 select-none relative">
                            {/* Whisker Line */}
                            <div className="absolute left-[10%] w-[80%] h-[1px] bg-indigo-500/40 top-[50%] -translate-y-1/2" style={{
                              left: '0%',
                              width: '100%'
                            }} />

                            {/* Min cap */}
                            <div className="absolute w-[1px] h-4 bg-indigo-400 top-[50%] -translate-y-1/2" style={{ left: '0%' }} />
                            
                            {/* Max cap */}
                            <div className="absolute w-[1px] h-4 bg-indigo-400 top-[50%] -translate-y-1/2" style={{ left: '100%' }} />

                            {/* Interquartile Box */}
                            <div className="absolute h-8 bg-gradient-to-r from-indigo-900/60 to-indigo-850/60 rounded border border-indigo-400/50 top-[50%] -translate-y-1/2 shadow-lg" style={{
                              left: `${q1Pct}%`,
                              width: `${q3Pct - q1Pct}%`
                            }} />

                            {/* Median Line */}
                            <div className="absolute h-8 w-1 bg-[#3bc8c8] hover:bg-white transition-colors top-[50%] -translate-y-1/2 shadow" style={{
                              left: `${medianPct}%`
                            }} />

                            {/* Spacing alignment spacers */}
                            <div className="h-10 w-full" />

                            {/* Dynamic coordinates text */}
                            <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 pt-4 border-t border-slate-850/40">
                              <span className="text-slate-400">Min: <strong className="text-white">{min.toLocaleString()}</strong></span>
                              <span className="text-indigo-400 font-bold">Q1 (25%): <strong className="text-slate-200">{sumStats.q1.toLocaleString()}</strong></span>
                              <span className="text-[#3bc8c8] font-bold">Median: <strong className="text-white">{sumStats.median.toLocaleString()}</strong></span>
                              <span className="text-indigo-400 font-bold">Q3 (75%): <strong className="text-slate-200">{sumStats.q3.toLocaleString()}</strong></span>
                              <span className="text-slate-400">Max: <strong className="text-white">{max.toLocaleString()}</strong></span>
                            </div>
                          </div>
                        );
                      })()}
                    </div>

                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 7. CATEGORICAL DISTRIBUTIONS WORKSPACE */}
        {activeSubTab === 'categorical' && (
          <div className="space-y-6 animate-fade-in text-xs">
            <div className="bg-slate-900/40 border border-slate-800 p-5 rounded-xl shadow-lg space-y-4">
              
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pb-2 border-b border-slate-800">
                <div>
                  <h4 className="text-white text-sm font-bold flex items-center gap-2">
                    <Layers className="w-4.5 h-4.5 text-indigo-400" /> Categorical Fields Group & Frequency Counts
                  </h4>
                  <p className="text-slate-455 mt-1">
                    Select a text or category coordinate to compile unique label frequencies, observation occurrences, and probability weights.
                  </p>
                </div>

                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-slate-550 font-bold uppercase tracking-wider">Category variable:</span>
                  <select
                    value={selectedCatCol}
                    onChange={(e) => setSelectedCatCol(e.target.value)}
                    className="bg-[#0B0E17] border border-slate-750 text-slate-200 px-3 py-1.5 rounded-xl text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors cursor-pointer"
                  >
                    {categoricalColsObj.map(c => (
                      <option key={c.name} value={c.name}>{c.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              {!selectedCatCol ? (
                <div className="text-center p-8 text-xs text-slate-500 border border-dashed border-slate-850 rounded-xl">
                  No categorical columns identified in the active workbook parameters. Add columns containing text labels.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-slate-950/40 p-4 border border-slate-850 rounded-xl space-y-1 mb-2 font-mono flex justify-between text-[11px] text-slate-400">
                    <span>Selected Category: <strong className="text-white">{selectedCatCol}</strong></span>
                    <span>Analyzed Unique States: <strong className="text-indigo-400">{activeCatDistribution.length} distinct levels</strong></span>
                  </div>

                  <div className="space-y-4 bg-slate-950/20 p-5 rounded-xl border border-slate-850">
                    {activeCatDistribution.slice(0, 15).map((item, idx) => (
                      <div key={idx} className="space-y-2">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className="text-white font-bold truncate max-w-[240px]">
                            {item.value === 'null' || item.value === '' ? (
                              <em className="text-slate-550 font-sans">[Empty Field Value]</em>
                            ) : item.value}
                          </span>
                          <span className="text-slate-400 text-[11px]">
                            {item.count.toLocaleString()} occurrences <span className="text-slate-550 mr-1">|</span> <strong className="text-indigo-300 font-semibold">{item.percent}%</strong>
                          </span>
                        </div>
                        <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-850 relative">
                          <div
                            className="bg-gradient-to-r from-indigo-650 to-indigo-505 h-full rounded-full transition-all duration-500"
                            style={{ width: `${item.percent}%` }}
                          />
                        </div>
                      </div>
                    ))}
                    {activeCatDistribution.length > 15 && (
                      <p className="text-[10px] text-slate-500 text-center font-mono pt-3 border-t border-slate-850">
                        ... list truncated. Showing top 15 highest-frequency discrete unique levels.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 8. HYPOTHESIS LAB */}
        {activeSubTab === 'hypothesis' && (
          <HypothesisLab dataset={dataset} />
        )}

        {/* 9. A/B TEST INTERPRETER */}
        {activeSubTab === 'abtest' && (
          <ABTestInterpreter />
        )}

        {/* 10. SQL ASSISTANT */}
        {activeSubTab === 'sql' && (
          <SQLAssistant dataset={dataset} />
        )}

        {/* 11. DEEP EDA — REAL SERVER-SIDE PANDAS/NUMPY COMPUTATION */}
        {activeSubTab === 'deepeda' && (
          <div className="space-y-6 animate-fade-in">
            {!deepEdaResult && (
              <div className="relative bg-[#0E1325]/85 backdrop-blur-md text-white p-6 sm:p-8 rounded-2xl border border-indigo-500/20 shadow-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-6 overflow-hidden">
                <div className="absolute right-0 top-0 translate-x-12 -translate-y-12 w-64 h-64 bg-indigo-500/15 rounded-full blur-3xl pointer-events-none animate-pulse" />
                <div className="relative z-10 space-y-2.5">
                  <span className="text-[9px] font-mono font-extrabold text-[#7e9dff] uppercase tracking-widest bg-indigo-500/15 px-3 py-1 rounded-full border border-indigo-500/25 font-mono">
                    REAL SERVER-SIDE ANALYSIS
                  </span>
                  <h3 className="text-base sm:text-lg font-bold tracking-tight flex items-center gap-2 mt-1 font-sans">
                    <Database className="w-5 h-5 text-indigo-400" /> Deep EDA Report (Python / pandas)
                  </h3>
                  <p className="text-xs text-slate-300 max-w-xl leading-relaxed font-sans">
                    Runs genuine pandas/numpy computation on your dataset — skewness, IQR outliers, Pearson correlation, and a missing-value report — then asks Gemini to narrate strictly those exact numbers, nothing invented.
                  </p>
                </div>
                <button
                  onClick={runDeepEda}
                  disabled={loadingDeepEda}
                  className="relative z-10 bg-indigo-600 hover:bg-indigo-500 text-white font-extrabold text-xs py-3 px-6 rounded-xl flex items-center justify-center gap-2.5 transition-all duration-300 shadow-lg shadow-indigo-600/40 border-0 cursor-pointer disabled:opacity-75 disabled:cursor-not-allowed shrink-0"
                >
                  {loadingDeepEda ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Computing Real Stats...
                    </>
                  ) : (
                    <>
                      <Database className="w-4.5 h-4.5 text-slate-100" />
                      Run Real Deep EDA
                    </>
                  )}
                </button>
              </div>
            )}

            {deepEdaError && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-xl text-xs flex items-start gap-3">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <p>{deepEdaError}</p>
              </div>
            )}

            {deepEdaResult && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-500">
                    {deepEdaResult.rowCount.toLocaleString()} rows × {deepEdaResult.columnCount} columns scanned
                  </span>
                  <button
                    onClick={runDeepEda}
                    disabled={loadingDeepEda}
                    className="text-[10px] font-mono text-indigo-400 hover:text-indigo-300 disabled:opacity-50 cursor-pointer"
                  >
                    {loadingDeepEda ? 'Re-running…' : 'Re-run scan'}
                  </button>
                </div>

                {/* Narrative panel — real numbers, AI (or template) narrated */}
                {deepEdaResult.narrative && (
                  <div className="bg-gradient-to-r from-slate-900 to-indigo-950/40 rounded-2xl border border-indigo-550/15 p-5 sm:p-6 shadow-xl">
                    <span className="text-[9px] font-mono font-bold text-indigo-400 uppercase tracking-wider bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20 font-mono">
                      Real Numbers, AI-Narrated
                    </span>
                    <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-4 rounded-xl border border-slate-850 font-sans mt-3">
                      {deepEdaResult.narrative.summary}
                    </p>

                    {deepEdaResult.narrative.dataQualityFlags.length > 0 && (
                      <div className="mt-4 space-y-1.5">
                        <span className="text-[10px] uppercase font-mono font-bold text-amber-400">Data Quality Flags</span>
                        {deepEdaResult.narrative.dataQualityFlags.map((f, i) => (
                          <div key={i} className="text-[11px] bg-slate-950/50 p-2 rounded border border-slate-850 flex items-center gap-2">
                            <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase shrink-0 ${f.severity === 'High' ? 'bg-rose-500/15 text-rose-400' : f.severity === 'Medium' ? 'bg-amber-500/15 text-amber-400' : 'bg-slate-800 text-slate-400'}`}>{f.severity}</span>
                            <strong className="text-slate-200">{f.column}:</strong>
                            <span className="text-slate-400">{f.issue}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {deepEdaResult.narrative.recommendedNextSteps.length > 0 && (
                      <div className="mt-4">
                        <span className="text-[10px] uppercase font-mono font-bold text-emerald-400">Recommended Next Steps</span>
                        <ul className="mt-1.5 space-y-1">
                          {deepEdaResult.narrative.recommendedNextSteps.map((s, i) => (
                            <li key={i} className="text-[11px] text-slate-300 flex items-start gap-1.5">
                              <Dot className="w-3.5 h-3.5 text-indigo-400 shrink-0 -mt-0.5" />{s}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* Real per-column distribution stats including skew */}
                <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono mb-3">Real Per-Column Distribution Stats</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-[11px] font-mono border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 text-[10px]">
                          <th className="py-2 pr-3">Column</th>
                          <th className="py-2 pr-3">Mean</th>
                          <th className="py-2 pr-3">Median</th>
                          <th className="py-2 pr-3">Std</th>
                          <th className="py-2 pr-3">Skew</th>
                          <th className="py-2 pr-3">Min</th>
                          <th className="py-2 pr-3">Max</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850/40">
                        {deepEdaResult.numericSummaries.map((s, i) => (
                          <tr key={i}>
                            <td className="py-2 pr-3 text-white font-bold">{s.column}</td>
                            <td className="py-2 pr-3 text-slate-300">{s.mean}</td>
                            <td className="py-2 pr-3 text-slate-300">{s.median}</td>
                            <td className="py-2 pr-3 text-slate-300">{s.std}</td>
                            <td className={`py-2 pr-3 font-bold ${Math.abs(s.skew) > 1 ? 'text-amber-400' : 'text-emerald-400'}`} title="Fisher-Pearson skewness: 0 = symmetric">{s.skew}</td>
                            <td className="py-2 pr-3 text-slate-500">{s.min}</td>
                            <td className="py-2 pr-3 text-slate-500">{s.max}</td>
                          </tr>
                        ))}
                        {deepEdaResult.numericSummaries.length === 0 && (
                          <tr><td colSpan={7} className="py-3 text-center text-slate-500">No numeric columns to summarize.</td></tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Real outliers + missing report side by side */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono mb-3">Real IQR Outlier Counts</h4>
                    <div className="space-y-1.5">
                      {deepEdaResult.outliers.map((o, i) => (
                        <div key={i} className="flex justify-between items-center text-[11px] bg-slate-950/50 p-2 rounded border border-slate-850">
                          <span className="text-slate-300 font-mono">{o.column}</span>
                          <span className={`font-mono font-bold ${o.outlierCount > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>{o.outlierCount} ({o.outlierPercent}%)</span>
                        </div>
                      ))}
                      {deepEdaResult.outliers.length === 0 && (
                        <p className="text-[11px] text-slate-500">No numeric columns to check for outliers.</p>
                      )}
                    </div>
                  </div>
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono mb-3">Real Missing-Value Report</h4>
                    <div className="space-y-1.5">
                      {deepEdaResult.missingReport.filter(m => m.missingCount > 0).map((m, i) => (
                        <div key={i} className="flex justify-between items-center text-[11px] bg-slate-950/50 p-2 rounded border border-slate-850">
                          <span className="text-slate-300 font-mono">{m.column}</span>
                          <span className="font-mono font-bold text-amber-400">{m.missingCount} ({m.missingPercent}%)</span>
                        </div>
                      ))}
                      {deepEdaResult.missingReport.every(m => m.missingCount === 0) && (
                        <p className="text-[11px] text-emerald-400">No missing values detected in any column.</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Real Pearson correlation matrix */}
                {deepEdaResult.correlation && deepEdaResult.correlation.columns.length >= 2 && (
                  <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6">
                    <h4 className="text-xs font-bold text-white uppercase tracking-wider font-mono mb-3">Real Pearson Correlation Matrix</h4>
                    <div className="overflow-x-auto">
                      <table className="text-[10px] font-mono border-collapse">
                        <thead>
                          <tr>
                            <th className="p-1.5 text-slate-500"></th>
                            {deepEdaResult.correlation.columns.map(c => (
                              <th key={c} className="p-1.5 text-slate-400 font-bold whitespace-nowrap">{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {deepEdaResult.correlation.matrix.map((row, rIdx) => (
                            <tr key={rIdx}>
                              <td className="p-1.5 text-slate-400 font-bold whitespace-nowrap">{deepEdaResult.correlation!.columns[rIdx]}</td>
                              {row.map((val, cIdx) => {
                                const v = val ?? 0;
                                const intensity = Math.min(1, Math.abs(v));
                                const bg = rIdx === cIdx
                                  ? 'rgba(99,102,241,0.25)'
                                  : v > 0 ? `rgba(16,185,129,${intensity * 0.5})` : `rgba(244,63,94,${intensity * 0.5})`;
                                return (
                                  <td key={cIdx} className="p-1.5 text-center text-slate-200" style={{ background: bg }}>
                                    {val === null ? '—' : val.toFixed(2)}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

/**
 * Highly interactive Scatter plot sub-component with SVG
 */
interface FeatureRelationshipScatterProps {
  dataset: Dataset;
  numericCols: string[];
}
function FeatureRelationshipScatter({ dataset, numericCols }: FeatureRelationshipScatterProps) {
  const [colX, setColX] = useState<string>(numericCols[0] || '');
  const [colY, setColY] = useState<string>(numericCols[1] || numericCols[0] || '');

  const allDataPoints = colX && colY
    ? dataset.rows.map((row, idx) => ({
        x: Number(row[colX]),
        y: Number(row[colY]),
        idx: idx + 1
      })).filter(pt => !isNaN(pt.x) && !isNaN(pt.y))
    : [];

  // Limit to 1500 points for browser rendering performance to prevent call stack issues / crashes
  const STEP = Math.max(1, Math.floor(allDataPoints.length / 1500));
  const dataPoints = allDataPoints.filter((_, i) => i % STEP === 0);

  const minX = dataPoints.length > 0 ? dataPoints.reduce((min, p) => (p.x < min ? p.x : min), dataPoints[0].x) : 0;
  const maxX = dataPoints.length > 0 ? dataPoints.reduce((max, p) => (p.x > max ? p.x : max), dataPoints[0].x) : 100;
  const minY = dataPoints.length > 0 ? dataPoints.reduce((min, p) => (p.y < min ? p.y : min), dataPoints[0].y) : 0;
  const maxY = dataPoints.length > 0 ? dataPoints.reduce((max, p) => (p.y > max ? p.y : max), dataPoints[0].y) : 100;

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  // Margin spacing inside SVG coordinate systems
  const margin = 40;
  const svgWidth = 600;
  const svgHeight = 350;

  // Grid line levels
  const xGridTicks = 5;
  const yGridTicks = 4;

  return (
    <div className="bg-slate-950/50 p-5 border border-slate-850 rounded-xl space-y-4">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <div>
          <span className="text-[9px] uppercase font-mono font-extrabold text-[#7e9dff] bg-indigo-500/15 border border-indigo-500/25 px-2.5 py-1 rounded">
            FEATURE CORATION MATRIX DIAGNOSTIC
          </span>
          <h5 className="font-extrabold text-white text-sm mt-1.5 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-indigo-400" /> Interactive Feature-Relationship Scatter Coordinator
          </h5>
        </div>

        {/* Dropdown axis control pairs */}
        <div className="flex gap-2 text-[11px] font-mono">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">X-Axis:</span>
            <select
              value={colX}
              onChange={(e) => setColX(e.target.value)}
              className="bg-[#0B0E17] border border-slate-750 text-slate-350 py-1 px-2 rounded-lg"
            >
              {numericCols.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">Y-Axis:</span>
            <select
              value={colY}
              onChange={(e) => setColY(e.target.value)}
              className="bg-[#0B0E17] border border-slate-750 text-slate-350 py-1 px-2 rounded-lg"
            >
              {numericCols.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto w-full select-none">
        <div className="min-w-[550px] flex justify-center bg-slate-950 rounded-lg p-3 relative h-[380px]">
          <svg
            width="100%"
            height="100%"
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="text-slate-500 font-mono text-[9px] max-w-2xl"
          >
            {/* Draw Y Axis Grid lines */}
            {Array.from({ length: yGridTicks }).map((_, i) => {
              const fraction = i / (yGridTicks - 1);
              const yVal = minY + fraction * rangeY;
              const yCoord = svgHeight - margin - fraction * (svgHeight - 2 * margin);
              return (
                <g key={i}>
                  <line
                    x1={margin}
                    y1={yCoord}
                    x2={svgWidth - margin}
                    y2={yCoord}
                    stroke="#1E293B"
                    strokeDasharray="3,3"
                  />
                  <text x={margin - 8} y={yCoord + 3} textAnchor="end" fill="#64748B">
                    {yVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Draw X Axis Grid lines */}
            {Array.from({ length: xGridTicks }).map((_, i) => {
              const fraction = i / (xGridTicks - 1);
              const xVal = minX + fraction * rangeX;
              const xCoord = margin + fraction * (svgWidth - 2 * margin);
              return (
                <g key={i}>
                  <line
                    x1={xCoord}
                    y1={margin}
                    x2={xCoord}
                    y2={svgHeight - margin}
                    stroke="#1E293B"
                    strokeDasharray="3,3"
                  />
                  <text x={xCoord} y={svgHeight - margin + 14} textAnchor="middle" fill="#64748B">
                    {xVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Draw Scatter Dots */}
            {dataPoints.map((pt, i) => {
              const cx = margin + ((pt.x - minX) / rangeX) * (svgWidth - 2 * margin);
              const cy = svgHeight - margin - ((pt.y - minY) / rangeY) * (svgHeight - 2 * margin);
              return (
                <g key={i} className="group cursor-help">
                  <circle
                    cx={cx}
                    cy={cy}
                    r={5.5}
                    className="fill-indigo-500 stroke-teal-400 hover:fill-teal-350 hover:r-8 transition-all hover:stroke-white duration-150"
                  />
                  {/* Tooltip on dot hover inside SVG */}
                  <title>
                    {`Observation Row ${pt.idx}:\n[X] ${colX} = ${pt.x}\n[Y] ${colY} = ${pt.y}`}
                  </title>
                </g>
              );
            })}

            {/* Axis labels */}
            <text x={svgWidth / 2} y={svgHeight - 4} textAnchor="middle" className="fill-slate-400 font-bold font-mono">
              X Variable: {colX}
            </text>
            <text
              x={margin}
              y={margin - 12}
              textAnchor="start"
              className="fill-slate-400 font-bold font-mono"
            >
              Y Variable: {colY}
            </text>
          </svg>
        </div>
      </div>
    </div>
  );
}
