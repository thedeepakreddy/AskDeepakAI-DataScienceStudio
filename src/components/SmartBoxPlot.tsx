import React, { useMemo } from 'react';
import { useDashboardContext } from './DashboardContext';

interface SmartBoxPlotProps {
  column: string;
  groupColumn?: string;
}

export default function SmartBoxPlot({ column, groupColumn }: SmartBoxPlotProps) {
  const { filteredData } = useDashboardContext();

  const plotData = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return [];

    let groups: Record<string, number[]> = {};
    if (groupColumn) {
      filteredData.forEach(row => {
        let key = row[groupColumn];
        if (!key) key = 'Unknown';
        if (!groups[key]) groups[key] = [];
        const val = Number(row[column]);
        if (!isNaN(val)) groups[key].push(val);
      });
    } else {
      const vals = filteredData.map(r => Number(r[column])).filter(v => !isNaN(v));
      groups['All'] = vals;
    }

    const calcBox = (values: number[]) => {
      values.sort((a, b) => a - b);
      const q1 = values[Math.floor(values.length * 0.25)];
      const median = values[Math.floor(values.length * 0.5)];
      const q3 = values[Math.floor(values.length * 0.75)];
      const iqr = q3 - q1;
      const lowerFence = q1 - 1.5 * iqr;
      const upperFence = q3 + 1.5 * iqr;

      const regularVals = values.filter(v => v >= lowerFence && v <= upperFence);
      const min = regularVals.length ? regularVals[0] : lowerFence;
      const max = regularVals.length ? regularVals[regularVals.length - 1] : upperFence;
      const outliers = values.filter(v => v < lowerFence || v > upperFence);

      return { min, q1, median, q3, max, outliers };
    };

    return Object.entries(groups).map(([group, vals]) => {
      if (vals.length < 5) return null;
      return { group, ...calcBox(vals) };
    }).filter(Boolean) as any[];

  }, [filteredData, column, groupColumn]);

  if (plotData.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg min-h-[300px] flex flex-col items-center justify-center">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest mb-4">Box Plot: {column} {groupColumn ? `by ${groupColumn}` : ''}</h3>
        <p className="text-slate-500 text-xs text-center">No sufficient data for box plot.</p>
      </div>
    );
  }

  // Global scale mapping
  let globalMin = Infinity;
  let globalMax = -Infinity;
  plotData.forEach(d => {
    if (d.min < globalMin) globalMin = d.min;
    if (d.max > globalMax) globalMax = d.max;
    d.outliers.forEach((o: number) => {
      if (o < globalMin) globalMin = o;
      if (o > globalMax) globalMax = o;
    });
  });

  const range = globalMax - globalMin || 1;
  const mapY = (val: number) => 100 - (((val - globalMin) / range) * 100);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg min-h-[300px] flex flex-col">
       <div className="flex justify-between items-start mb-4">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest overflow-hidden text-ellipsis whitespace-nowrap pr-2">
          Box Plot: {column} {groupColumn ? `by ${groupColumn}` : ''}
        </h3>
      </div>
      
      <div className="w-full h-[280px] relative flex">
         {/* Y Axis Labels roughly */}
         <div className="w-10 h-full flex flex-col justify-between text-[9px] text-slate-500 font-mono items-end pr-2">
            <span>{globalMax.toFixed(1)}</span>
            <span>{globalMin.toFixed(1)}</span>
         </div>

         <div className="flex-1 h-full flex items-end justify-around border-b border-l border-slate-800 pb-2 relative pb-8">
            {plotData.map((d, i) => (
              <div key={i} className="flex flex-col items-center group w-12 cursor-pointer h-full relative" title={`Q1: ${d.q1.toFixed(1)}\nQ3: ${d.q3.toFixed(1)}\nMed: ${d.median.toFixed(1)}`}>
                 <div className="absolute w-full h-full bottom-0">
                    {/* Whiskers */}
                    <div className="absolute w-px bg-slate-500 left-1/2 -translate-x-1/2" style={{ top: `${mapY(d.max)}%`, bottom: `${100 - mapY(d.q3)}%` }}></div>
                    <div className="absolute w-px bg-slate-500 left-1/2 -translate-x-1/2" style={{ top: `${mapY(d.q1)}%`, bottom: `${100 - mapY(d.min)}%` }}></div>
                    {/* Whiskers Caps */}
                    <div className="absolute w-4 h-px bg-slate-400 left-1/2 -translate-x-1/2" style={{ top: `${mapY(d.max)}%` }}></div>
                    <div className="absolute w-4 h-px bg-slate-400 left-1/2 -translate-x-1/2" style={{ top: `${mapY(d.min)}%` }}></div>
                    
                    {/* IQR Box */}
                    <div className="absolute w-full bg-blue-500/20 border border-blue-400 left-0 transition-colors group-hover:bg-blue-500/40" style={{ top: `${mapY(d.q3)}%`, bottom: `${100 - mapY(d.q1)}%` }}>
                       {/* Median Line */}
                       <div className="absolute w-full h-px bg-emerald-400 left-0" style={{ top: `${((d.q3 - d.median) / (d.q3 - d.q1)) * 100}%` }}></div>
                    </div>

                    {/* Outliers */}
                    {d.outliers.map((o: number, oi: number) => (
                      <div key={`outlier-${oi}`} className="absolute w-1.5 h-1.5 rounded-full border border-rose-400 left-1/2 -translate-x-1/2 bg-slate-900" style={{ top: `calc(${mapY(o)}% - 3px)` }}></div>
                    ))}
                 </div>
                 <div className="absolute -bottom-6 text-[9px] font-bold text-slate-400 whitespace-nowrap overflow-hidden text-ellipsis w-16 text-center">{d.group}</div>
              </div>
            ))}
         </div>
      </div>
    </div>
  );
}
