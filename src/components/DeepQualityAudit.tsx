import React, { useState } from 'react';
import { Dataset } from '../types';
import { Microscope, Loader2, AlertTriangle, Download, Code, ShieldCheck } from 'lucide-react';
import { usePipelineContext } from '../contexts/PipelineContext';

interface DeepQualityAuditProps {
  dataset: Dataset;
}

interface Issue {
  issueName: string;
  severity: string;
  description: string;
  pythonFix: string;
}

export default function DeepQualityAudit({ dataset }: DeepQualityAuditProps) {
  const { setHasRunAudit } = usePipelineContext();
  const [loading, setLoading] = useState(false);
  const [issues, setIssues] = useState<Issue[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAudit = async () => {
    setLoading(true);
    setError(null);

    try {
      // 1. Calculate Audit Summary locally
      const duplicateCount = calculateDuplicates();
      const columnIssues = dataset.columns.map(col => {
        const uniqueValues = new Set();
        let stringCount = 0;
        let numberCount = 0;
        
        dataset.rows.forEach(row => {
           const val = row[col.name];
           if (val !== undefined && val !== null) {
              uniqueValues.add(val);
              if (typeof val === 'string') stringCount++;
              if (typeof val === 'number') numberCount++;
           }
        });

        const numUnique = uniqueValues.size;
        const cardinalityRatio = numUnique / dataset.rowCount;

        const report = {
           name: col.name,
           missingCount: col.missingCount,
           missingPercentage: ((col.missingCount / dataset.rowCount) * 100).toFixed(2) + '%',
           uniqueValues: numUnique,
           isConstant: numUnique === 1,
           isHighCardinality: col.type === 'categorical' && cardinalityRatio > 0.5 && numUnique > 50,
           typeMismatchWarning: (stringCount > 0 && numberCount > 0) ? true : false,
        };
        return report;
      });

      const auditSummary = {
        totalRows: dataset.rowCount,
        totalColumns: dataset.columns.length,
        duplicateRows: duplicateCount,
        columnIssues: columnIssues.filter(c => c.missingCount > 0 || c.isConstant || c.isHighCardinality || c.typeMismatchWarning)
      };

      // 2. Fetch AI Python fixes
      const response = await fetch('/api/deep-quality-audit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auditSummary })
      });

      const data = await response.json();
      if (response.ok && data.fixes) {
        setIssues(data.fixes);
        setHasRunAudit(true);
      } else {
        setError(data.error || 'Failed to process quality audit.');
      }
    } catch (err: any) {
      setError('Network error running deep audit.');
    } finally {
      setLoading(false);
    }
  };

  const calculateDuplicates = () => {
     let duplicates = 0;
     const seen = new Set();
     dataset.rows.forEach(row => {
        const str = JSON.stringify(row);
        if (seen.has(str)) duplicates++;
        else seen.add(str);
     });
     return duplicates;
  };

  const downloadReport = () => {
    if (!issues) return;
    let reportText = `Data Quality Audit Report\nDataset: ${dataset.filename}\nTotal Rows: ${dataset.rowCount}\n\n`;
    issues.forEach((issue, idx) => {
      reportText += `${idx + 1}. [${issue.severity.toUpperCase()}] ${issue.issueName}\n`;
      reportText += `Description: ${issue.description}\n`;
      reportText += `Python Fix:\n${issue.pythonFix}\n\n`;
    });

    const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `DeepQualityAudit_${dataset.filename}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800 p-6 sm:p-8 shadow-2xl relative overflow-hidden mt-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-500/10 rounded-lg flex items-center justify-center border border-indigo-500/20 shadow-inner">
            <Microscope className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white tracking-tight">Deep Quality Audit</h2>
            <p className="text-xs text-slate-400 mt-1">AI-powered scan for duplicates, mismatches, and formatting anomalies.</p>
          </div>
        </div>
        {!issues && !loading && (
          <button
            onClick={runAudit}
            className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-sm py-2 px-6 rounded-xl transition-all duration-300 shadow-lg shadow-indigo-500/20"
          >
            Run Comprehensive Scan
          </button>
        )}
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-12">
           <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mb-4" />
           <p className="text-sm font-bold text-indigo-300">Scanning thousands of cells across {dataset.columns.length} columns...</p>
           <p className="text-xs text-slate-500 mt-1">Evaluating type boundaries and detecting dataset anomalies.</p>
        </div>
      )}

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-xl text-xs flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <p>{error}</p>
        </div>
      )}

      {issues && !loading && (
        <div className="space-y-6 animate-fade-in">
          {issues.length === 0 ? (
            <div className="text-center py-8">
              <ShieldCheck className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
              <h3 className="text-lg font-bold text-white">Dataset is Pristine</h3>
              <p className="text-sm text-slate-400 mt-1">No critical anomalies, duplicates, or type mismatches discovered.</p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {issues.map((issue, idx) => {
                  const isHigh = issue.severity.toLowerCase() === 'high';
                  const isMed = issue.severity.toLowerCase() === 'medium';
                  return (
                    <div key={idx} className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 shadow-sm hover:border-indigo-500/30 transition-colors">
                      <div className="flex items-start justify-between mb-3">
                        <h4 className="text-sm font-bold text-white line-clamp-1 flex-1 pr-3">{issue.issueName}</h4>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase shrink-0 ${isHigh ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : isMed ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'}`}>
                          {issue.severity}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed mb-4">{issue.description}</p>
                      <div className="bg-slate-900 rounded-lg p-3 border border-slate-800/80">
                         <div className="flexItems-center gap-2 mb-2">
                           <Code className="w-3.5 h-3.5 text-slate-500" />
                           <span className="text-[10px] font-mono uppercase text-slate-500 font-bold tracking-wider">Suggested Fix</span>
                         </div>
                         <pre className="text-xs font-mono text-emerald-300 overflow-x-auto whitespace-pre-wrap">
                            {issue.pythonFix}
                         </pre>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-start mt-6">
                 <button
                   onClick={downloadReport}
                   className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white font-bold text-sm py-2.5 px-6 rounded-xl transition-all duration-300 border border-slate-700 border-b-2"
                 >
                   <Download className="w-4 h-4" /> Download Quality Report (TXT)
                 </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
