/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Dataset, DatasetColumn } from '../types';

export function parseCSV(text: string, filename: string): Dataset {
  const lines: string[] = [];
  let currentWord = '';
  let inQuotes = false;
  let currentLine: string[] = [];

  // Parse lines considering quoted commas
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1] || '';

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        currentWord += '"';
        i++; // skip next quote
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      currentLine.push(currentWord.trim());
      currentWord = '';
    } else if ((char === '\r' || char === '\n') && !inQuotes) {
      if (char === '\r' && nextChar === '\n') {
        i++;
      }
      currentLine.push(currentWord.trim());
      if (currentLine.length > 0 && currentLine.some(cell => cell !== '')) {
        lines.push(JSON.stringify(currentLine));
      }
      currentLine = [];
      currentWord = '';
    } else {
      currentWord += char;
    }
  }
  // Add trailing line
  if (currentWord !== '' || currentLine.length > 0) {
    currentLine.push(currentWord.trim());
    lines.push(JSON.stringify(currentLine));
  }

  if (lines.length < 2) {
    throw new Error('Dataset must contain at least a header row and one data row.');
  }

  const parsedLines = lines.map(l => JSON.parse(l) as string[]);
  const headers = parsedLines[0].map(h => h.replace(/^"|"$/g, '').trim() || 'unnamed_column');
  
  // Deduplicate headers
  const headerCount: Record<string, number> = {};
  const cleanHeaders = headers.map(header => {
    let name = header;
    if (headerCount[name] !== undefined) {
      headerCount[name]++;
      name = `${name}_${headerCount[name]}`;
    } else {
      headerCount[name] = 0;
    }
    return name;
  });

  const rawRows = parsedLines.slice(1);
  const rows: Record<string, any>[] = [];

  rawRows.forEach(rowCells => {
    if (rowCells.length === 0 || rowCells.every(c => c === '')) return;
    const rowObj: Record<string, any> = {};
    cleanHeaders.forEach((header, index) => {
      rowObj[header] = rowCells[index] !== undefined ? rowCells[index] : '';
    });
    rows.push(rowObj);
  });

  // Extract columns metadata
  const columns: DatasetColumn[] = cleanHeaders.map(name => {
    // Collect non-empty values
    const values = rows.map(r => r[name]).filter(v => v !== undefined && v !== '');
    const missingCount = rows.length - values.length;

    // Detect Column Type
    // Try to parse as numbers
    let numberCount = 0;
    let booleanCount = 0;
    let dateCount = 0;

    const parsedValues = values.map(v => {
      const trimmed = String(v).trim().toLowerCase();
      // Boolean check
      if (trimmed === 'true' || trimmed === 'false' || trimmed === 'yes' || trimmed === 'no') {
        booleanCount++;
        return trimmed === 'true' || trimmed === 'yes';
      }
      // Number check
      const num = Number(v);
      if (!isNaN(num) && v !== '') {
        numberCount++;
        return num;
      }
      // Date check
      const d = Date.parse(v);
      if (!isNaN(d) && String(v).length > 5 && isNaN(Number(v))) {
        dateCount++;
        return new Date(d);
      }
      return v;
    });

    let detectedType: 'numeric' | 'categorical' | 'boolean' | 'datetime' = 'categorical';
    if (numberCount / values.length > 0.7) {
      detectedType = 'numeric';
    } else if (booleanCount / values.length > 0.7) {
      detectedType = 'boolean';
    } else if (dateCount / values.length > 0.7) {
      detectedType = 'datetime';
    }

    // Cast rows accordingly
    rows.forEach(r => {
      const val = r[name];
      if (val === undefined || val === '') {
        r[name] = null;
      } else if (detectedType === 'numeric') {
        const num = Number(val);
        r[name] = isNaN(num) ? null : num;
      } else if (detectedType === 'boolean') {
        const trm = String(val).trim().toLowerCase();
        r[name] = trm === 'true' || trm === 'yes' || trm === '1';
      } else if (detectedType === 'datetime') {
        const d = Date.parse(val);
        r[name] = isNaN(d) ? val : new Date(d).toISOString();
      } else {
        r[name] = String(val);
      }
    });

    // Compute Statistics
    const validCells = rows.map(r => r[name]).filter(v => v !== null && v !== undefined);
    const distinctSet = new Set(validCells);
    const distinctCount = distinctSet.size;

    const statistics: DatasetColumn['statistics'] = {};

    if (detectedType === 'numeric') {
      const numericCells = validCells as number[];
      if (numericCells.length > 0) {
        let min = numericCells[0];
        let max = numericCells[0];
        for (let idx = 1; idx < numericCells.length; idx++) {
          const v = numericCells[idx];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const sum = numericCells.reduce((a, b) => a + b, 0);
        const mean = sum / numericCells.length;
        
        // Median
        const sorted = [...numericCells].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

        // StdDev
        const sqDiffSum = numericCells.reduce((accum, val) => accum + Math.pow(val - mean, 2), 0);
        const stdDev = Math.sqrt(sqDiffSum / numericCells.length);

        statistics.min = parseFloat(min.toFixed(4));
        statistics.max = parseFloat(max.toFixed(4));
        statistics.mean = parseFloat(mean.toFixed(4));
        statistics.median = parseFloat(median.toFixed(4));
        statistics.stdDev = parseFloat(stdDev.toFixed(4));
      }
    }

    // Most Common of Categorical or Numeric
    const valueCounts: Record<string, number> = {};
    validCells.forEach(v => {
      const strVal = String(v);
      valueCounts[strVal] = (valueCounts[strVal] || 0) + 1;
    });

    const sortedFreq = Object.entries(valueCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([value, count]) => ({ value, count }));

    statistics.mostCommon = sortedFreq;

    return {
      name,
      type: detectedType,
      missingCount,
      distinctCount,
      statistics,
    };
  });

  return {
    filename,
    rows,
    columns,
    rowCount: rows.length,
    originalRowCount: rows.length,
  };
}

// Built-in Sample Datasets
export const SAMPlE_DATASETS: Record<string, string> = {
  // 1. Customer Churn Dataset
  churn: `CustomerID,Gender,Age,Tenure,MonthlyCharges,TotalCharges,ContractType,PaymentMethod,Target_Churn
C-10251,Male,42,12,65.85,790.2,Month-to-month,Credit Card,Yes
C-10252,Female,31,24,20.05,481.2,Two Year,,No
C-10253,Female,58,1,85.10,85.1,Month-to-month,Electronic Check,Yes
C-10254,Male,22,8,45.20,361.6,One Year,Bank Transfer,No
C-10255,Female,35,48,110.30,5294.4,Two Year,Credit Card,No
C-10256,Male,67,3,92.50,277.5,Month-to-month,Electronic Check,Yes
C-10257,Female,48,36,55.00,1980.0,One Year,Mailed Check,No
C-10258,Male,29,15,70.15,1052.25,Month-to-month,Bank Transfer,No
C-10259,Female,52,50,95.40,4770.0,Two Year,Electronic Check,No
C-10260,Male,73,6,89.90,539.4,Month-to-month,Electronic Check,Yes
C-10261,Female,19,4,30.50,122.0,Month-to-month,Mailed Check,No
C-10262,Male,34,18,60.45,1088.1,Month-to-month,Credit Card,No
C-10263,Female,41,10,75.00,750.0,One Year,Mailed Check,Yes
C-10264,Male,62,44,115.60,5086.4,Two Year,Bank Transfer,No
C-10265,Male,25,9,80.50,724.5,Month-to-month,Electronic Check,Yes
C-10266,Male,19,36,30.25,1115.21,One Year,Mailed Check,No
C-10267,Male,31,20,80.2,1613.82,Month-to-month,Mailed Check,Yes
C-10268,Female,69,36,62.26,2191.57,Two Year,Bank Transfer,No
C-10269,Female,24,6,54.4,318.71,Month-to-month,,Yes
C-10270,Male,42,9,108.55,963.43,Month-to-month,Credit Card,Yes
C-10271,Male,67,30,27.98,816.97,Month-to-month,Bank Transfer,Yes
C-10272,Female,28,16,113.67,1845.64,Month-to-month,Electronic Check,Yes
C-10273,Male,28,23,46.99,1126.75,Month-to-month,Electronic Check,Yes
C-10274,Male,32,40,26.62,1108.79,Two Year,Bank Transfer,No
C-10275,Female,43,22,84.29,1788.15,Two Year,Bank Transfer,No
C-10276,Female,75,4,33.83,135.23,One Year,Credit Card,No
C-10277,Male,68,72,79.64,5668.0,One Year,Mailed Check,No
C-10278,Male,61,28,46.68,1325.46,One Year,Mailed Check,No
C-10279,Male,64,20,46.34,926.85,Two Year,Credit Card,No
C-10280,Female,71,29,96.25,2802.24,One Year,Credit Card,Yes
C-10281,Female,19,9,56.3,531.29,Month-to-month,Electronic Check,Yes
C-10282,Male,23,51,73.27,3596.93,One Year,Electronic Check,No
C-10283,Female,31,1,95.53,97.34,Two Year,Mailed Check,Yes
C-10284,Female,46,25,26.4,628.39,Two Year,Electronic Check,No
C-10285,Male,32,16,23.14,363.96,Month-to-month,Bank Transfer,Yes
C-10286,Male,52,28,67.27,1937.17,Month-to-month,Electronic Check,No
C-10287,Female,40,7,66.7,477.6,Month-to-month,Credit Card,Yes
C-10288,Female,69,36,64.86,2316.72,Two Year,Electronic Check,No
C-10289,Male,46,45,105.6,4560.93,Two Year,Credit Card,No
C-10290,Male,28,1,110.24,106.54,Month-to-month,,Yes
C-10291,Female,36,16,115.73,1943.42,Month-to-month,Mailed Check,Yes
C-10292,Female,31,20,94.8,1812.04,Two Year,Electronic Check,No
C-10293,Male,72,6,79.5,485.36,Month-to-month,Credit Card,Yes
C-10294,Male,55,25,78.37,1963.71,One Year,Electronic Check,No
C-10295,Female,33,5,33.09,167.86,Month-to-month,Credit Card,Yes
C-10296,Male,22,28,54.9,1470.91,One Year,Bank Transfer,No
C-10297,Female,57,35,100.71,3535.05,Two Year,Bank Transfer,No
C-10298,Male,74,18,75.32,1324.9,Month-to-month,Bank Transfer,Yes
C-10299,Female,50,17,110.53,1944.04,Month-to-month,Mailed Check,Yes
C-10300,Male,18,36,36.16,1294.19,Month-to-month,Credit Card,No
C-10301,Male,52,38,56.92,2174.32,Month-to-month,Credit Card,No
C-10302,Male,75,6,55.37,334.21,Month-to-month,Electronic Check,Yes
C-10303,Male,73,1,99.99,96.76,Month-to-month,Electronic Check,Yes
C-10304,Female,68,45,101.07,4442.09,Two Year,Credit Card,No
C-10305,Male,72,1,39.96,41.63,Month-to-month,Electronic Check,Yes
C-10306,Male,43,34,47.91,1651.99,Month-to-month,Bank Transfer,Yes
C-10307,Male,74,8,37.86,316.89,Two Year,Credit Card,No
C-10308,Female,64,1,31.56,32.82,One Year,Credit Card,Yes
C-10309,Male,51,51,73.85,3849.0,Two Year,Electronic Check,No
C-10310,Male,60,21,32.46,708.93,Two Year,Mailed Check,No`,

  // 2. SaaS Performance Analytics
  saas: `Date,Customer_Segment,Subscription_Tier,Monthly_Recurring_Revenue,Users_Active_Daily,Support_Tickets_Opened,Customer_Success_Rating,Target_ChurnProbability
2026-01-01,Enterprise,Premium,5200,95,2,4.8,0.05
2026-01-02,Mid-Market,Standard,1200,45,4,3.2,0.35
2026-01-03,SMB,Starter,250,12,6,2.1,0.78
2026-01-04,Enterprise,Premium,4800,88,1,4.9,0.04
2026-01-05,Mid-Market,Standard,1400,50,3,3.9,0.18
2026-01-06,Enterprise,Standard,3000,65,5,3.1,0.42
2026-01-07,SMB,Starter,280,15,5,2.5,0.65
2026-01-08,Enterprise,Premium,7500,140,2,4.7,0.02
2026-01-09,Mid-Market,Premium,2450,72,0,4.5,0.08
2026-01-10,SMB,Starter,210,8,8,1.8,0.91
2026-01-11,Mid-Market,Standard,1350,48,4,3.5,0.22
2026-01-12,Enterprise,Standard,3200,70,3,4.1,0.15
2026-01-13,SMB,Starter,290,18,2,3.8,0.45
2026-01-14,Enterprise,Premium,8100,165,1,4.9,0.01
2026-01-15,Mid-Market,Premium,2600,80,2,4.2,0.12
2026-01-16,SMB,Standard,172,18,1,2.5,0.6
2026-01-17,SMB,Starter,255,14,0,2.2,0.74
2026-01-18,SMB,Starter,205,19,6,1.9,0.97
2026-01-19,Mid-Market,Standard,3112,50,2,3.7,0.49
2026-01-20,Enterprise,Standard,7507,156,2,4.1,0.45
2026-01-21,Enterprise,Premium,4469,130,0,4.4,0.18
2026-01-22,SMB,Standard,268,20,1,2.2,0.62
2026-01-23,SMB,Premium,278,8,2,2.6,0.51
2026-01-24,Mid-Market,Premium,1548,47,0,3.6,0.29
2026-01-25,SMB,Starter,245,15,9,1.8,0.97
2026-01-26,SMB,Starter,304,19,2,2.7,0.8
2026-01-27,Mid-Market,Standard,2654,80,3,3.4,0.53
2026-01-28,Mid-Market,Premium,1629,57,6,3.0,0.63
2026-02-01,Enterprise,Premium,4388,158,2,4.2,0.23
2026-02-02,Mid-Market,Standard,1455,66,5,3.2,0.7
2026-02-03,SMB,Premium,300,17,0,2.2,0.46
2026-02-04,Mid-Market,Standard,2616,78,4,3.9,0.52
2026-02-05,Mid-Market,Standard,1064,61,1,3.9,0.44
2026-02-06,Enterprise,Starter,6205,120,0,4.6,0.43
2026-02-07,Enterprise,Premium,3814,129,2,4.1,0.26
2026-02-08,Mid-Market,Standard,1834,66,5,3.0,0.66
2026-02-09,Enterprise,Standard,3313,114,4,4.7,0.36
2026-02-10,Enterprise,Standard,4196,72,4,4.3,0.44
2026-02-11,Mid-Market,Standard,2241,53,2,3.4,0.53
2026-02-12,SMB,Starter,282,10,4,2.6,0.83
2026-02-13,Enterprise,Standard,6455,118,0,4.0,0.37
2026-02-14,SMB,Standard,269,14,6,2.3,0.86
2026-02-15,Enterprise,Standard,6396,75,2,4.2,0.4
2026-02-16,Mid-Market,Premium,1924,71,4,2.9,0.57
2026-02-17,SMB,Starter,228,9,0,3.0,0.63
2026-02-18,SMB,Starter,229,13,5,1.8,0.97
2026-02-19,Enterprise,Premium,5705,166,0,4.4,0.09
2026-02-20,Mid-Market,Premium,1467,56,2,3.3,0.42
2026-02-21,SMB,Starter,244,11,7,1.8,0.97
2026-02-22,SMB,Starter,271,19,6,2.3,0.95
2026-02-23,Enterprise,Premium,4460,154,0,4.5,0.12
2026-02-24,SMB,Starter,210,19,2,2.0,0.86
2026-02-25,Enterprise,Standard,4893,165,0,4.1,0.31
2026-02-26,Mid-Market,Premium,2297,73,1,3.7,0.33
2026-02-27,Enterprise,Premium,5165,108,5,3.8,0.48
2026-02-28,SMB,Standard,310,20,4,1.6,0.82
2026-03-01,Mid-Market,Standard,1969,51,5,3.7,0.56
2026-03-02,Enterprise,Premium,3444,150,1,4.6,0.12
2026-03-03,SMB,Starter,164,12,7,2.0,0.97
2026-03-04,Enterprise,Standard,5282,126,6,3.8,0.63`,

  // 3. Equipment Hardware Telemetry & Machinery Maintenance
  hardware: `Vibration_Level,Temperature_Celsius,Pressure_PSI,Usage_Hours_Continuous,Operator_Experience,Maintenance_Status,Target_HardwareFailure
4.2,78.5,120,4.5,Intermediate,Good,0
8.5,95.2,165,12.0,Novice,Overdue,1
3.1,65.0,110,2.1,Expert,Good,0
5.8,82.4,142,6.8,Intermediate,Good,0
7.9,91.8,158,10.5,Novice,Pending,1
2.5,60.2,95,1.5,Expert,Good,0
6.1,84.0,135,8.0,Intermediate,Pending,0
9.2,98.6,180,14.2,Novice,Overdue,1
4.0,72.1,118,5.0,Expert,Good,0
5.0,79.5,130,5.8,Intermediate,Good,0
7.1,89.0,150,9.1,Novice,Pending,1
3.4,68.2,105,3.2,Expert,Good,0
6.5,86.4,140,8.5,Intermediate,Pending,0
8.9,96.8,172,13.0,Novice,Overdue,1
4.5,75.0,122,6.0,Expert,Good,0
9.2,91.2,154,10.8,Intermediate,Pending,1
6.6,83.4,133,11.9,Novice,Good,1
4.7,82.9,116,10.3,Intermediate,Good,0
3.8,74.7,112,4.1,Novice,Pending,0
6.9,89.4,138,8.5,Novice,Good,1
2.0,70.6,103,8.1,Expert,Good,0
7.3,89.2,131,3.9,Intermediate,Pending,0
4.7,82.2,128,7.1,Intermediate,Overdue,0
1.6,68.3,113,7.1,Expert,Good,0
1.7,69.0,99,8.2,Expert,Overdue,0
4.0,79.9,112,4.5,Intermediate,Good,0
7.0,90.0,129,0.6,Novice,Good,0
2.5,73.0,109,6.9,Intermediate,Good,0
5.6,82.8,121,14.5,Novice,Good,0
3.4,75.1,111,8.0,Expert,Pending,0
5.9,87.3,127,5.9,Novice,Overdue,1
3.0,71.8,108,8.2,Intermediate,Good,0
6.3,88.0,130,10.1,Expert,Good,0
6.0,84.8,134,0.9,Novice,Good,0
3.5,75.2,123,12.7,Intermediate,Good,0
1.0,67.5,98,12.1,Expert,Good,0
6.7,87.7,130,7.0,Expert,Good,0
6.1,87.3,125,8.9,Novice,Overdue,1
9.8,93.0,158,0.8,Novice,Good,0
6.4,86.9,134,9.0,Intermediate,Good,0
2.8,74.9,108,7.8,Intermediate,Overdue,0
10.7,101.3,162,12.7,Novice,Good,1
8.1,87.4,141,8.1,Novice,Good,1
6.0,81.6,133,5.5,Novice,Good,0
3.9,78.6,118,6.2,Intermediate,Good,0
3.0,71.2,120,11.8,Expert,Good,0
5.8,82.0,138,3.9,Novice,Overdue,1
5.5,84.2,120,9.8,Novice,Good,0
7.1,91.5,144,2.9,Intermediate,Good,0
3.7,79.3,113,5.0,Expert,Overdue,0
6.8,83.9,129,2.2,Novice,Pending,1
8.2,88.3,151,8.2,Novice,Pending,1
5.8,82.5,130,0.5,Intermediate,Good,0
10.9,98.2,166,4.4,Novice,Overdue,1
5.0,82.4,128,5.0,Intermediate,Good,0
5.0,79.1,119,9.4,Expert,Pending,0
1.6,69.1,107,10.9,Expert,Good,0
3.8,78.3,120,11.7,Intermediate,Overdue,0
3.9,73.8,126,8.2,Expert,Pending,0
5.2,85.2,129,9.7,Novice,Pending,1`
};

export function loadSampleDataset(key: 'churn' | 'saas' | 'hardware'): Dataset {
  const text = SAMPlE_DATASETS[key];
  const filenames = {
    churn: 'customer_churn_and_retention.csv',
    saas: 'saas_company_performance.csv',
    hardware: 'predictive_hardware_maintenance.csv'
  };
  return parseCSV(text, filenames[key]);
}
