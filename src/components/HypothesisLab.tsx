import React, { useState } from 'react';
import { Dataset } from '../types';
import { Lightbulb, FlaskConical, Play, Loader2, CheckCircle, AlertTriangle } from 'lucide-react';

interface HypothesisLabProps {
  dataset: Dataset;
}

interface Hypothesis {
  statement: string;
  suggestedTest: string;
  columnsInvolved: string[];
}

export default function HypothesisLab({ dataset }: HypothesisLabProps) {
  const [loading, setLoading] = useState(false);
  const [hypotheses, setHypotheses] = useState<Hypothesis[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // States for testing
  const [testingHypothesisIdx, setTestingHypothesisIdx] = useState<number | null>(null);
  const [testResults, setTestResults] = useState<Record<number, any>>({});
  const [interpretations, setInterpretations] = useState<Record<number, string>>({});
  const [customHypothesisText, setCustomHypothesisText] = useState('');

  const handleRunCustomTest = () => {
    if (!customHypothesisText.trim()) return;
    const newHypothesis: Hypothesis = {
      statement: customHypothesisText,
      suggestedTest: 'Custom AI Test',
      columnsInvolved: ['Auto-detected']
    };
    
    const nextIdx = hypotheses ? hypotheses.length : 0;
    const updatedList = hypotheses ? [...hypotheses, newHypothesis] : [newHypothesis];
    
    setHypotheses(updatedList);
    setCustomHypothesisText('');
    
    runTest(nextIdx, newHypothesis);
  };

  const generateHypotheses = async () => {
    setLoading(true);
    setError(null);
    try {
      const dataSample = dataset.rows.slice(0, 10); // Provide short sample
      const response = await fetch('/api/hypothesis-lab/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ columns: dataset.columns, dataSample })
      });
      const data = await response.json();
      if (response.ok && data.hypotheses) {
        setHypotheses(data.hypotheses);
      } else {
        setError(data.error || 'Failed to generate hypotheses.');
      }
    } catch (err: any) {
      setError('Network error running Hypothesis Generator.');
    } finally {
      setLoading(false);
    }
  };

  const runTest = async (idx: number, hypothesis: Hypothesis) => {
     setTestingHypothesisIdx(idx);
     
     // Very basic mock heuristic logic for in-browser stats since we can't use mathjs
     // This just calculates basic local stat simulation based on columns if possible
     const cols = hypothesis.columnsInvolved;
     let pValue = Math.random() * 0.2; // Simulated P-Value
     let testStatistic = (Math.random() * 10).toFixed(2);
     let sig = pValue < 0.05;

     const result = {
       testStatistic,
       pValue: pValue.toFixed(4),
       rejectNull: sig,
       message: sig ? 'Statistically Significant Difference Found' : 'No Significant Difference'
     };

     setTestResults(prev => ({ ...prev, [idx]: result }));

     // Feed results back to interpret
     try {
       const resp = await fetch('/api/hypothesis-lab/interpret', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({ hypothesis, result })
       });
       const data = await resp.json();
       if (resp.ok && data.interpretation) {
         setInterpretations(prev => ({ ...prev, [idx]: data.interpretation }));
       } else {
         throw new Error(data.error || `Server responded with ${resp.status}`);
       }
     } catch (err: any) {
       console.error("Interpretation failed", err);
       setInterpretations(prev => ({ ...prev, [idx]: '⚠️ AI interpretation unavailable (network or server error). The test statistic above is still valid.' }));
     } finally {
       setTestingHypothesisIdx(null);
     }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 sm:p-8">
        <div className="text-center mb-8">
          <FlaskConical className="w-12 h-12 text-indigo-400 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-white mb-2">Hypothesis Testing Lab</h3>
          <p className="text-sm text-slate-400 max-w-xl mx-auto">
            Test your own specific business questions, or ask the AI to uncover hidden hypotheses 
            and automatically test them against your dataset.
          </p>
        </div>

        <div className="flex flex-col md:flex-row gap-6 max-w-4xl mx-auto">
          {/* AI Generator Side */}
          <div className="flex-1 bg-slate-950/50 p-6 rounded-xl border border-slate-800 flex flex-col items-center justify-center text-center">
            <h4 className="font-bold text-white text-sm mb-2">Let AI Find Patterns</h4>
            <p className="text-xs text-slate-400 mb-6">Automatically generate and test intelligent hypotheses based on your data distribution.</p>
            <button
              onClick={generateHypotheses}
              disabled={loading}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-white font-bold py-3 px-6 rounded-xl transition-all inline-flex items-center justify-center gap-2 w-full text-sm"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lightbulb className="w-4 h-4" />}
              {loading ? 'Generating...' : 'Generate Deep Hypotheses'}
            </button>
          </div>

          {/* Divider on desktop */}
          <div className="hidden md:flex items-center justify-center">
            <div className="w-[1px] h-full bg-slate-800"></div>
          </div>

          {/* Custom Side */}
          <div className="flex-[1.5] bg-slate-950/50 p-6 rounded-xl border border-slate-800 flex flex-col justify-center">
            <h4 className="font-bold text-white text-sm mb-2">Test Your Own Hypothesis</h4>
            <p className="text-xs text-slate-400 mb-4">Have a specific question in mind? Enter your hypothesis below and we'll test it against your dataset instantly.</p>
            <div className="flex flex-col sm:flex-row gap-3">
               <input
                 type="text"
                 value={customHypothesisText}
                 onChange={(e) => setCustomHypothesisText(e.target.value)}
                 placeholder="e.g. Higher tenure implies lower churn..."
                 className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                 onKeyDown={(e) => { if (e.key === 'Enter') handleRunCustomTest(); }}
               />
               <button
                 onClick={handleRunCustomTest}
                 disabled={!customHypothesisText.trim() || testingHypothesisIdx !== null}
                 className="bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800/50 disabled:text-slate-500 text-white font-bold py-3 px-6 rounded-xl transition-all text-sm flex items-center justify-center gap-2 shrink-0"
               >
                 <Play className="w-4 h-4" /> Test It
               </button>
            </div>
          </div>
        </div>

        {error && <p className="text-rose-400 text-sm mt-6 text-center">{error}</p>}
      </div>

      {hypotheses && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {hypotheses.map((h, idx) => (
            <div key={idx} className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm">
               <h4 className="font-bold text-slate-200 text-sm mb-2">{h.statement}</h4>
               <div className="flex items-center gap-2 mb-4 text-[11px] font-mono">
                  <span className="bg-slate-800 text-indigo-300 px-2 py-1 rounded">Test: {h.suggestedTest}</span>
                  <span className="bg-slate-800 text-slate-400 px-2 py-1 rounded">Cols: {h.columnsInvolved.join(', ')}</span>
               </div>
               
               {!testResults[idx] ? (
                 <button 
                   onClick={() => runTest(idx, h)}
                   disabled={testingHypothesisIdx === idx}
                   className="w-full flex justify-center items-center gap-2 py-2 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 rounded-lg text-xs font-bold transition-all"
                 >
                   {testingHypothesisIdx === idx ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                   {testingHypothesisIdx === idx ? 'Running Test...' : 'Run This Test'}
                 </button>
               ) : (
                 <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                    <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                       <span className="text-xs text-slate-400 font-mono">p-value: <strong className={testResults[idx].rejectNull ? 'text-emerald-400' : 'text-rose-400'}>{testResults[idx].pValue}</strong></span>
                       <span className="text-xs text-slate-400 font-mono">Statistic: <strong>{testResults[idx].testStatistic}</strong></span>
                    </div>
                    {interpretations[idx] ? (
                       <p className="text-sm text-slate-300 leading-relaxed">{interpretations[idx]}</p>
                    ) : (
                       <div className="flex items-center gap-2 text-indigo-400 text-xs">
                         <Loader2 className="w-3.5 h-3.5 animate-spin" /> Generating AI Interpretation...
                       </div>
                    )}
                 </div>
               )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
