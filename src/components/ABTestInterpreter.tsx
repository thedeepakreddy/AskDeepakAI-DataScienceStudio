import React, { useState } from 'react';
import { AreaChart, BrainCircuit, Loader2, Copy, CheckCircle, AlertCircle } from 'lucide-react';

export default function ABTestInterpreter() {
  const [controlSize, setControlSize] = useState(1000);
  const [controlRate, setControlRate] = useState(0.12);
  const [treatmentSize, setTreatmentSize] = useState(1000);
  const [treatmentRate, setTreatmentRate] = useState(0.14);
  const [confidence, setConfidence] = useState(0.95);
  const [testType, setTestType] = useState('two-tailed');
  
  const [loading, setLoading] = useState(false);
  const [memo, setMemo] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Manual calculation for A/B Test (Two Proportion Z-Test)
  const calculateTest = () => {
     const p1 = controlRate;
     const p2 = treatmentRate;
     const n1 = controlSize;
     const n2 = treatmentSize;

     const pPool = (p1*n1 + p2*n2) / (n1 + n2);
     const sePool = Math.sqrt(pPool * (1 - pPool) * ((1/n1) + (1/n2)));
     const zScore = (p2 - p1) / sePool;
     
     // Very rough normal CDF approximation for p-value
     const approxP = Math.exp(-0.717 * zScore - 0.416 * Math.pow(zScore, 2));
     const pValue = testType === 'two-tailed' ? approxP * 2 : approxP;
     const relativeLift = ((p2 - p1) / p1) * 100;
     const isSig = pValue < (1 - confidence);

     return {
       zScore: zScore.toFixed(3),
       pValue: pValue.toFixed(4),
       relativeLift: relativeLift.toFixed(2),
       isSignificant: isSig,
       mde: '2.5%' // standard mock calculation for simplicity
     };
  };

  const generateMemo = async () => {
    setLoading(true);
    setMemo(null);
    setError(null);
    try {
       const results = calculateTest();
       const response = await fetch('/api/ab-test-interpreter', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ results, params: { controlSize, controlRate, treatmentSize, treatmentRate, confidence, testType } })
       });
       if (!response.ok) {
         throw new Error(`Server responded with ${response.status}`);
       }
       const data = await response.json();
       if (data.memo) {
         setMemo(data.memo);
       } else {
         throw new Error('No memo was returned by the server.');
       }
    } catch(err: any) {
       console.error(err);
       setError(err.message || 'Network error generating the AI business memo.');
    } finally {
       setLoading(false);
    }
  };

  const copyMemo = () => {
    if (memo) {
       navigator.clipboard.writeText(memo);
       setCopied(true);
       setTimeout(() => setCopied(false), 2000);
    }
  };

  const results = calculateTest();

  return (
    <div className="space-y-6 animate-fade-in">
       <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Inputs */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
             <h3 className="text-lg font-bold text-white mb-6">A/B Test Configuration</h3>
             
             <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                   <div>
                     <label className="block text-xs font-bold text-slate-400 mb-1">Control Size (N)</label>
                     <input type="number" value={controlSize} onChange={e => setControlSize(Number(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white" />
                   </div>
                   <div>
                     <label className="block text-xs font-bold text-blue-400 mb-1">Control Rate</label>
                     <input type="number" step="0.01" value={controlRate} onChange={e => setControlRate(Number(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white" />
                   </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                   <div>
                     <label className="block text-xs font-bold text-slate-400 mb-1">Treatment Size (N)</label>
                     <input type="number" value={treatmentSize} onChange={e => setTreatmentSize(Number(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white" />
                   </div>
                   <div>
                     <label className="block text-xs font-bold text-emerald-400 mb-1">Treatment Rate</label>
                     <input type="number" step="0.01" value={treatmentRate} onChange={e => setTreatmentRate(Number(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white" />
                   </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                   <div>
                     <label className="block text-xs font-bold text-slate-400 mb-1">Confidence</label>
                     <select value={confidence} onChange={e => setConfidence(Number(e.target.value))} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white">
                        <option value={0.90}>90%</option>
                        <option value={0.95}>95%</option>
                        <option value={0.99}>99%</option>
                     </select>
                   </div>
                   <div>
                     <label className="block text-xs font-bold text-slate-400 mb-1">Test Type</label>
                     <select value={testType} onChange={e => setTestType(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-sm text-white">
                        <option value="one-tailed">One-Tailed</option>
                        <option value="two-tailed">Two-Tailed</option>
                     </select>
                   </div>
                </div>
             </div>
          </div>

          {/* Results locally generated */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
             <div>
                <h3 className="text-lg font-bold text-white mb-6">Test Results Calculator</h3>
                <div className="space-y-4">
                   <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                      <span className="text-sm text-slate-400">Relative Lift</span>
                      <span className={`text-lg font-bold font-mono ${Number(results.relativeLift) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{Number(results.relativeLift) > 0 ? '+' : ''}{results.relativeLift}%</span>
                   </div>
                   <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                      <span className="text-sm text-slate-400">P-Value</span>
                      <span className="text-lg font-bold text-white font-mono">{results.pValue}</span>
                   </div>
                   <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                      <span className="text-sm text-slate-400">Significance Level Reached?</span>
                      <span className={`text-lg font-bold ${results.isSignificant ? 'text-emerald-400' : 'text-rose-400'}`}>{results.isSignificant ? 'YES' : 'NO'}</span>
                   </div>
                </div>
             </div>

             <button
               onClick={generateMemo}
               disabled={loading}
               className="w-full mt-6 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-white font-bold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2"
             >
               {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <BrainCircuit className="w-4 h-4" />}
               Generate AI Business Memo
             </button>
          </div>
       </div>

       {error && (
         <div className="p-4 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-xl text-xs flex items-start gap-3">
           <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
           <p>{error}</p>
         </div>
       )}

       {memo && (
         <div className="bg-slate-900/60 border border-indigo-500/20 rounded-xl p-6 shadow-xl relative animate-fade-in">
           <h4 className="text-sm font-bold text-indigo-400 uppercase tracking-wider mb-4 border-b border-indigo-500/10 pb-2">Business Recommendation Memo</h4>
           <div className="prose prose-invert max-w-none text-sm text-slate-300 leading-relaxed whitespace-pre-wrap font-sans">
             {memo}
           </div>
           <button
             onClick={copyMemo}
             className="absolute top-6 right-6 flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs font-semibold text-slate-300 transition-colors"
           >
             {copied ? <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
             {copied ? 'Copied' : 'Copy Memo'}
           </button>
         </div>
       )}
    </div>
  );
}
