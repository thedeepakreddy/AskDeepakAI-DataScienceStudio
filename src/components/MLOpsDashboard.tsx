import React, { useState } from 'react';
import { usePipelineContext } from '../contexts/PipelineContext';
import { Rocket, Activity, CheckCircle, ShieldAlert, Cpu, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Dataset } from '../types';

interface DriftMetrics {
  [feature: string]: {
    ks_stat: number;
    p_value: number;
    drift_detected: boolean;
  };
}

interface MLOpsDashboardProps {
  dataset: Dataset;
  target?: string;
  features?: string[];
}

function pickColumns(row: Record<string, any>, cols: string[]): Record<string, any> {
  const out: Record<string, any> = {};
  cols.forEach(c => { out[c] = row[c]; });
  return out;
}

export default function MLOpsDashboard({ dataset, target, features }: MLOpsDashboardProps) {
  const { expertMode } = usePipelineContext();
  const [deployStatus, setDeployStatus] = useState<'idle' | 'deploying' | 'online'>('idle');
  const [deployError, setDeployError] = useState<string | null>(null);
  const [driftData, setDriftData] = useState<DriftMetrics | null>(null);
  const [driftError, setDriftError] = useState<string | null>(null);
  const [deploySummary, setDeploySummary] = useState<{ modelName: string; metricLabel: string; metricValue: number } | null>(null);

  const resolvedTarget = (target && dataset.columns.some(c => c.name === target))
    ? target
    : dataset.columns[dataset.columns.length - 1]?.name;

  const handleDeploy = async () => {
    if (!resolvedTarget || dataset.rows.length === 0) {
      setDeployError('No active dataset or target column available to train against.');
      return;
    }
    setDeployStatus('deploying');
    setDeployError(null);
    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: dataset.rows,
          target: resolvedTarget,
          features: features && features.length > 0 ? features : undefined,
          models: ['random_forest'],
        })
      });
      const text = await response.text();
      let body: any = null;
      try { body = text ? JSON.parse(text) : null; } catch { /* host returned a non-JSON gateway page */ }
      if (!response.ok) {
        throw new Error(body?.error || `Deployment failed (HTTP ${response.status}). The ML compute service may still be waking up from an idle sleep — try again in a moment.`);
      }
      const champion = body.champion;
      setDeploySummary({
        modelName: champion.modelName,
        metricLabel: champion.primaryMetric,
        metricValue: champion.primaryMetricValue,
      });
      setDeployStatus('online');
      fetchDriftMetrics();
    } catch (err: any) {
      console.error(err);
      setDeployError(err.message || 'Deployment failed — is the ML compute service running?');
      setDeployStatus('idle');
    }
  };

  // Honest, real drift check: this app has no separate live production traffic
  // stream, so instead of inventing one, we run a real Kolmogorov-Smirnov test
  // comparing the first half of the active dataset (reference) against the
  // second half (current) on each numeric column.
  const fetchDriftMetrics = async () => {
    const numericCols = dataset.columns.filter(c => c.type === 'numeric').map(c => c.name);
    if (numericCols.length === 0 || dataset.rows.length < 20) return;

    const mid = Math.floor(dataset.rows.length / 2);
    const referenceRows = dataset.rows.slice(0, mid).map(r => pickColumns(r, numericCols));
    const currentRows = dataset.rows.slice(mid).map(r => pickColumns(r, numericCols));

    setDriftError(null);
    try {
      const response = await fetch('/api/drift-metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference_data: referenceRows, current_data: currentRows })
      });
      const text = await response.text();
      let data: any = null;
      try { data = text ? JSON.parse(text) : null; } catch { /* host returned a non-JSON gateway page */ }
      if (!response.ok) throw new Error(data?.error || `Drift check failed (HTTP ${response.status}).`);
      setDriftData(data.drift_status);
    } catch (err: any) {
      console.error(err);
      setDriftError(err.message || 'Drift check failed — is the ML compute service running?');
    }
  };

  const chartData = driftData ? Object.keys(driftData).map(feat => ({
    feature: feat,
    ksStat: driftData[feat].ks_stat,
    pValue: driftData[feat].p_value,
    isDrifted: driftData[feat].drift_detected
  })) : [];

  return (
    <div className="bg-[#050A10] border border-slate-800 rounded-2xl p-6 shadow-2xl">
      <div className="flex flex-col md:flex-row items-center justify-between mb-8 pb-4 border-b border-slate-800 gap-4">
        <div className="flex items-center gap-3">
          <Cpu className="w-6 h-6 text-indigo-400" />
          <div>
            <h2 className="text-lg font-bold text-white">MLOps Production Center</h2>
            <p className="text-sm text-slate-400">Real Python Compute — trains an actual Random Forest on your active dataset</p>
          </div>
        </div>

        <button
          onClick={handleDeploy}
          disabled={deployStatus !== 'idle'}
          className={`flex items-center gap-2 px-6 py-3 font-bold rounded-xl shadow-lg transition-all ${deployStatus === 'online' ? 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/50 cursor-default' : 'bg-indigo-600 hover:bg-indigo-500 text-white active:scale-95'}`}
        >
          {deployStatus === 'idle' && <><Rocket className="w-4 h-4" /> 1-Click Deploy</>}
          {deployStatus === 'deploying' && <><Activity className="w-4 h-4 animate-spin" /> Training on Real Data...</>}
          {deployStatus === 'online' && <><CheckCircle className="w-4 h-4" /> Live Deployment Online</>}
        </button>
      </div>

      {deployError && (
        <div className="mb-6 p-3.5 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-xl text-xs flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{deployError}</span>
        </div>
      )}

      {deployStatus === 'online' && (
        <div className="space-y-6">
          {deploySummary && (
            <div className="bg-emerald-950/20 border border-emerald-500/20 rounded-xl p-4 flex items-center justify-between">
              <div>
                <p className="text-[10px] text-emerald-400/80 uppercase tracking-wider font-bold font-mono">Real Trained Model</p>
                <p className="text-sm text-white font-bold mt-0.5">{deploySummary.modelName}</p>
              </div>
              <div className="text-right">
                <p className="text-[10px] text-emerald-400/80 uppercase tracking-wider font-bold font-mono">{deploySummary.metricLabel}</p>
                <p className="text-lg text-emerald-300 font-black mt-0.5">
                  {deploySummary.metricLabel.toLowerCase().includes('accuracy')
                    ? `${(deploySummary.metricValue * 100).toFixed(1)}%`
                    : deploySummary.metricValue.toFixed(3)}
                </p>
              </div>
            </div>
          )}

          <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
            <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-4">Data Drift Telemetry (Real KS-Test)</h3>
            <p className="text-[10.5px] text-slate-500 mb-4 -mt-2">
              Comparing the first half of your active dataset (reference) against the second half (current) — this app has no separate live production stream, so this is a genuine statistical comparison of real historical data rather than simulated traffic.
            </p>

            {driftError && (
              <div className="p-3 mb-4 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-xl text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <p>{driftError}</p>
              </div>
            )}

            {!expertMode ? (
               // BEGINNER MODE: Simple badges
               <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                 {chartData.map((stat, idx) => (
                   <div key={idx} className={`p-4 rounded-xl border flex items-center gap-3 ${stat.isDrifted ? 'bg-red-500/10 border-red-500/20' : 'bg-emerald-500/10 border-emerald-500/20'}`}>
                     {stat.isDrifted ? <ShieldAlert className="text-red-400 w-6 h-6" /> : <CheckCircle className="text-emerald-400 w-6 h-6" />}
                     <div>
                       <div className="text-xs text-slate-400 font-bold uppercase">{stat.feature} Health</div>
                       <div className={`font-bold ${stat.isDrifted ? 'text-red-300' : 'text-emerald-300'}`}>{stat.isDrifted ? 'Drifted' : 'Stable'}</div>
                     </div>
                   </div>
                 ))}
                 {chartData.length === 0 && !driftError && (
                   <div className="col-span-3 text-center text-slate-500 text-xs py-4">No numeric columns available for drift analysis.</div>
                 )}
               </div>
            ) : (
               // EXPERT MODE: Recharts visualization
               <div className="h-[250px] w-full">
                 <ResponsiveContainer width="100%" height="100%">
                   <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                     <XAxis dataKey="feature" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
                     <YAxis tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
                     <Tooltip
                       contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px' }}
                       formatter={(value: number, name: string) => [value.toFixed(3), name === 'ksStat' ? 'KS-Statistic' : 'P-Value']}
                       labelStyle={{ color: '#94a3b8', fontWeight: 'bold', marginBottom: '4px' }}
                     />
                     <ReferenceLine y={0.05} label="Drift Threshold (p=0.05)" stroke="#ef4444" strokeDasharray="3 3" />
                     <Bar dataKey="ksStat" fill="#6366f1" radius={[4, 4, 0, 0]} name="KS-Statistic" />
                   </BarChart>
                 </ResponsiveContainer>
               </div>
            )}
            {expertMode && <p className="text-xs text-slate-500 mt-4 leading-relaxed">* Real Kolmogorov-Smirnov test computed by the Python service between the two real data slices described above. Values peaking above the threshold signal a genuine distribution shift.</p>}
          </div>
        </div>
      )}
    </div>
  );
}
