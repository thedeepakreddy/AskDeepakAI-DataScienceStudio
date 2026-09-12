/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence, useMotionValue, animate } from 'motion/react';
import { Sparkles, MessageSquare, Send, X, Trash2, Cpu, BarChart2, Check, RefreshCw, Layers, Compass, Code, Play } from 'lucide-react';
import { Dataset } from '../types';

interface AssistiveTouchBotProps {
  activeDataset: Dataset | null;
  onUpdateDataset: (updated: Dataset) => void;
  onResetOriginal: () => void;
  activeTab: string;
  setActiveTab: (tab: any) => void;
  triggerAIScan: () => Promise<void>;
  triggerPrediction: (
    targetCol: string,
    features: string[],
    modelClass: 'classification' | 'regression' | 'timeseries',
    hyperparameters: Record<string, any>
  ) => Promise<any>;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  executedCommands?: string[];
}

export default function AssistiveTouchBot({
  activeDataset,
  onUpdateDataset,
  onResetOriginal,
  activeTab,
  setActiveTab,
  triggerAIScan,
  triggerPrediction
}: AssistiveTouchBotProps) {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-msg',
      role: 'assistant',
      content: `Hello! I am **AskAI**, acting as your interactive **AskDeepakAI** co-pilot. 👋

I have real-time access to your active dataset. I can **read** statistics, **write adjustments** (like dropping columns or filling missing cells), and **edit workflow stages** instantly by your command.

💡 **Tips for interacting with me:**
You can give a **SQL script** or **Python script** or **English** or other language common English prompt or any language prompt (so AI can understand what update user wants on current dataset). Choose your preferred mode below!

Try asking me items like:
- *"Analyze this dataset with an EDA scan"*
- *"Drop the PaymentMethod column"*
- *"SELECT * FROM dataset WHERE Age > 30"*
- *"dataset.rows.map(r => r.Age = Number(r.Age) + 10)"*`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [inputMode, setInputMode] = useState<'english' | 'sql' | 'python'>('english');
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastExecutedTaskMsg, setLastExecutedTaskMsg] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const dragContainerRef = useRef<HTMLDivElement | null>(null);
  const isDragging = useRef(false);

  // Responsive state for absolute coordinate snapping representation
  // Starts safely at a guaranteed visible status, then snaps instantly relative to true client viewport on mount
  const mx = useMotionValue(100);
  const my = useMotionValue(100);

  // 1. Force position to the absolute right side of the actual browser screen on mount
  useEffect(() => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (w > 50 && h > 50) {
      const buttonSize = 56;
      mx.set(w - buttonSize); // force right edge on initial mount
      my.set(Math.max(0, Math.min(h - buttonSize, h - 185))); // safe bottom right
    }
  }, []);

  // 2. Continuous smart clamping to keep the button fully visible and snapped to the nearest edge
  useEffect(() => {
    const clampToVisibleScreen = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      
      if (w > 50 && h > 50) {
        const buttonSize = 56;

        const cx = mx.get();
        const cy = my.get();

        const dLeft = cx;
        const dRight = w - buttonSize - cx;
        const dTop = cy;
        const dBottom = h - buttonSize - cy;

        const minDist = Math.min(dLeft, dRight, dTop, dBottom);

        let targetX = cx;
        let targetY = cy;

        if (minDist === dLeft) {
          targetX = 0;
          targetY = Math.max(0, Math.min(h - buttonSize, cy));
        } else if (minDist === dRight) {
          targetX = w - buttonSize;
          targetY = Math.max(0, Math.min(h - buttonSize, cy));
        } else if (minDist === dTop) {
          targetY = 0;
          targetX = Math.max(0, Math.min(w - buttonSize, cx));
        } else {
          targetY = h - buttonSize;
          targetX = Math.max(0, Math.min(w - buttonSize, cx));
        }

        if (targetX !== cx || targetY !== cy) {
          mx.set(targetX);
          my.set(targetY);
        }
      }
    };

    // Listen to resize events
    window.addEventListener('resize', clampToVisibleScreen);

    // Progressive checking intervals to compensate for slow loads, sub-iframes, and dynamic viewports on Render
    const checks = [100, 300, 600, 1200, 2500, 4500];
    const timers = checks.map(t => setTimeout(clampToVisibleScreen, t));

    return () => {
      window.removeEventListener('resize', clampToVisibleScreen);
      timers.forEach(clearTimeout);
    };
  }, []);

  const [isDraggingState, setIsDraggingState] = useState(false);
  const dragStartOffset = useRef({ x: 0, y: 0 });
  const startPointerCoords = useRef({ x: 0, y: 0 });

  const [isChatDragging, setIsChatDragging] = useState(false);
  const chatDragStart = useRef({ x: 0, y: 0 });

  const handleDragEnd = (e: any, info: any) => {
    setIsDraggingState(false);
    
    // Snapping logic (Always snap to left or right edges like native iOS AssistiveTouch)
    const w = window.innerWidth;
    const h = window.innerHeight;
    const buttonSize = 56;
    const margin = 16;
    
    // Header spacing and footer spacing limits
    const topLimit = margin + 64; 
    const bottomLimit = h - buttonSize - margin - 50;

    let targetX = mx.get();
    let targetY = my.get();

    // Snap to nearest horizontal edge
    if (targetX < w / 2) {
      targetX = margin; // Left
    } else {
      targetX = w - buttonSize - margin; // Right
    }

    // Clamp vertical position strictly within view bounds
    targetY = Math.max(topLimit, Math.min(bottomLimit, targetY));

    animate(mx, targetX, { type: 'spring', stiffness: 450, damping: 25 });
    animate(my, targetY, { type: 'spring', stiffness: 450, damping: 25 });
  };

  const handleChatDragEnd = (e: any, info: any) => {
    setIsChatDragging(false);

    const chatWidth = 270;
    const chatHeight = 360;
    const pad = 12;
    const footerLimit = 100;

    const targetX = mx.get();
    const targetY = my.get();

    // Boundary constraints matching the UI viewport bounds perfectly
    const finalX = Math.max(pad, Math.min(window.innerWidth - chatWidth - pad, targetX));
    const finalY = Math.max(80, Math.min(window.innerHeight - footerLimit - chatHeight, targetY));

    animate(mx, finalX, { type: 'spring', stiffness: 350, damping: 25 });
    animate(my, finalY, { type: 'spring', stiffness: 350, damping: 25 });
  };

  // Auto scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isChatOpen) {
      setTimeout(scrollToBottom, 60);
    }
  }, [messages, isChatOpen]);

  // Execute API commands returned from Gemini in real-time
  const executeCommandsList = async (commands: any[]): Promise<string[]> => {
    const executedLogs: string[] = [];
    if (!commands || commands.length === 0) return executedLogs;

    for (const cmd of commands) {
      try {
        switch (cmd.type) {
          case 'SELECT_TAB':
            if (cmd.tab) {
              setActiveTab(cmd.tab);
              executedLogs.push(`🔄 Switched workflow stage to "${cmd.tab.toUpperCase()}"`);
            }
            break;

          case 'RESET_DATASET':
            onResetOriginal();
            executedLogs.push('🧹 Dynamic dataset restored to original RAW template');
            break;

          case 'DROP_COLUMN':
            if (activeDataset && cmd.column) {
              const targetColName = activeDataset.columns.find(
                c => c.name.toLowerCase() === cmd.column.toLowerCase()
              )?.name;

              if (targetColName) {
                const refreshedCols = activeDataset.columns.filter(c => c.name !== targetColName);
                const refreshedRows = activeDataset.rows.map(row => {
                  const copy = { ...row };
                  delete copy[targetColName];
                  return copy;
                });

                onUpdateDataset({
                  ...activeDataset,
                  columns: refreshedCols,
                  rows: refreshedRows,
                  rowCount: refreshedRows.length
                });
                executedLogs.push(`🗑️ Dropped column "${targetColName}" from active coordinates`);
              } else {
                executedLogs.push(`⚠️ Column "${cmd.column}" not found in current structures`);
              }
            }
            break;

          case 'FILL_MISSING':
            if (activeDataset && cmd.column) {
              const targetCol = activeDataset.columns.find(
                c => c.name.toLowerCase() === cmd.column.toLowerCase()
              );

              if (targetCol) {
                const strategy = cmd.strategy || 'mean';
                let fillVal: any = 0;

                if (targetCol.type === 'numeric') {
                  if (strategy === 'mean') fillVal = targetCol.statistics.mean || 0;
                  else if (strategy === 'median') fillVal = targetCol.statistics.median || 0;
                  else fillVal = 0;
                } else {
                  fillVal = targetCol.statistics.mostCommon?.[0]?.value || 'Unknown';
                }

                const cleanedRows = activeDataset.rows.map(row => {
                  const val = row[targetCol.name];
                  if (val === null || val === undefined || val === '') {
                    return { ...row, [targetCol.name]: fillVal };
                  }
                  return row;
                });

                const updatedCols = activeDataset.columns.map(col => {
                  if (col.name === targetCol.name) {
                    return { ...col, missingCount: 0 };
                  }
                  return col;
                });

                onUpdateDataset({
                  ...activeDataset,
                  columns: updatedCols,
                  rows: cleanedRows
                });
                executedLogs.push(`✨ Imputed missing values in "${targetCol.name}" using strategy "${strategy}" (${fillVal})`);
              } else {
                executedLogs.push(`⚠️ Column "${cmd.column}" not found to perform imputation`);
              }
            }
            break;

          case 'RUN_EDA_SCAN':
            // Async background call
            triggerAIScan();
            executedLogs.push('🧠 Neural exploratory scanning started in the background');
            break;

          case 'RUN_ML':
            if (activeDataset && cmd.targetColumn) {
              const foundTarget = activeDataset.columns.find(
                c => c.name.toLowerCase() === cmd.targetColumn.toLowerCase()
              )?.name || activeDataset.columns[activeDataset.columns.length - 1]?.name;

              const featCols = Array.isArray(cmd.featureColumns) 
                ? cmd.featureColumns.map(fc => activeDataset.columns.find(c => c.name.toLowerCase() === fc.toLowerCase())?.name).filter(Boolean) as string[]
                : activeDataset.columns.filter(c => c.name !== foundTarget).map(c => c.name).slice(0, 4);

              const modelType = cmd.modelType || (foundTarget.toLowerCase().includes('churn') ? 'classification' : 'regression');
              const hParams = cmd.hyperparameters || { n_estimators: 100, max_depth: 8 };

              triggerPrediction(foundTarget, featCols, modelType, hParams);
              executedLogs.push(`🔮 Machine Learning pipeline trained on "${foundTarget}" using features [${featCols.join(', ')}]`);
            }
            break;

          case 'FILTER_ROWS':
            if (activeDataset && cmd.column && cmd.operator) {
              const targetColName = activeDataset.columns.find(
                c => c.name.toLowerCase() === cmd.column.toLowerCase()
              )?.name;

              if (targetColName) {
                const operator = cmd.operator;
                const filterVal = cmd.value;

                const filteredRows = activeDataset.rows.filter(row => {
                  const val = row[targetColName];
                  if (val === null || val === undefined) return false;

                  if (operator === '==') return String(val).toLowerCase() === String(filterVal).toLowerCase();
                  if (operator === '!=') return String(val).toLowerCase() !== String(filterVal).toLowerCase();
                  if (operator === '>') return Number(val) > Number(filterVal);
                  if (operator === '<') return Number(val) < Number(filterVal);
                  return true;
                });

                onUpdateDataset({
                  ...activeDataset,
                  rows: filteredRows,
                  rowCount: filteredRows.length
                });

                executedLogs.push(`🔍 Filtered dataset: Kept rows where "${targetColName}" ${operator} "${filterVal}" (${filteredRows.length} left)`);
              }
            }
            break;

          case 'ADD_COLUMN':
            if (activeDataset && cmd.column) {
              const newColName = cmd.column;
              const newColType = cmd.columnType || 'categorical';
              const defaultVal = cmd.value !== undefined ? cmd.value : '';

              if (!activeDataset.columns.some(c => c.name.toLowerCase() === newColName.toLowerCase())) {
                const newColMeta = {
                  name: newColName,
                  type: newColType as any,
                  missingCount: 0,
                  distinctCount: 1,
                  statistics: {
                    min: newColType === 'numeric' ? Number(defaultVal) || 0 : undefined,
                    max: newColType === 'numeric' ? Number(defaultVal) || 0 : undefined,
                    mean: newColType === 'numeric' ? Number(defaultVal) || 0 : undefined,
                    median: newColType === 'numeric' ? Number(defaultVal) || 0 : undefined,
                    mostCommon: [{ value: String(defaultVal), count: activeDataset.rows.length }]
                  }
                };

                const updatedRows = activeDataset.rows.map(row => ({
                  ...row,
                  [newColName]: defaultVal
                }));

                onUpdateDataset({
                  ...activeDataset,
                  columns: [...activeDataset.columns, newColMeta],
                  rows: updatedRows
                });
                executedLogs.push(`➕ Added column "${newColName}" of type "${newColType}" filled with "${defaultVal}"`);
              } else {
                executedLogs.push(`⚠️ Column "${newColName}" already exists`);
              }
            }
            break;

          case 'ADD_ROW':
            if (activeDataset) {
              const newRow: Record<string, any> = {};
              activeDataset.columns.forEach(col => {
                if (col.type === 'numeric') {
                  newRow[col.name] = 0;
                } else if (col.type === 'boolean') {
                  newRow[col.name] = false;
                } else {
                  newRow[col.name] = '';
                }
              });

              // Optional support for custom values filled by AI
              if (cmd.values && typeof cmd.values === 'object') {
                Object.keys(cmd.values).forEach(k => {
                  const matchCol = activeDataset.columns.find(col => col.name.toLowerCase() === k.toLowerCase());
                  if (matchCol) {
                    newRow[matchCol.name] = cmd.values[k];
                  }
                });
              }

              const updatedRows = [...activeDataset.rows, newRow];
              onUpdateDataset({
                ...activeDataset,
                rows: updatedRows,
                rowCount: updatedRows.length
              });
              executedLogs.push(`➕ Appended new raw row matching structure`);
            }
            break;

          case 'DELETE_ROW':
            if (activeDataset && cmd.index !== undefined) {
              const idxToDelete = Number(cmd.index);
              if (idxToDelete >= 0 && idxToDelete < activeDataset.rows.length) {
                const refreshedRows = activeDataset.rows.filter((_, idx) => idx !== idxToDelete);
                onUpdateDataset({
                  ...activeDataset,
                  rows: refreshedRows,
                  rowCount: refreshedRows.length
                });
                executedLogs.push(`🗑️ Deleted dataset row index ${idxToDelete}`);
              } else {
                executedLogs.push(`⚠️ Invalid row index ${idxToDelete} specified`);
              }
            }
            break;

          case 'EXECUTE_DATASET_JS':
            if (activeDataset && cmd.jsCode) {
              try {
                // Run this LLM/user-generated code inside an isolated,
                // opaque-origin sandboxed iframe rather than this page's own
                // privileged context. A dataset cell can carry a prompt-
                // injection payload that tricks the chat model into emitting
                // jsCode here, so it must never get direct access to this
                // page's cookies, localStorage, or DOM.
                const updated = await runSandboxedDatasetJs(cmd.jsCode, activeDataset);
                if (updated && Array.isArray(updated.rows)) {
                  onUpdateDataset({
                    ...activeDataset,
                    ...updated,
                    rowCount: updated.rows.length
                  });
                  let logMsg = `📊 SQL Action Applied`;
                  if (cmd.sqlQuery) logMsg += ` [Query: "${cmd.sqlQuery}"]`;
                  if (cmd.explanation) logMsg += `: ${cmd.explanation}`;
                  executedLogs.push(logMsg);
                } else {
                  executedLogs.push(`⚠️ Action executed successfully but returned incomplete data arrays`);
                }
              } catch (evalErr: any) {
                console.error("Evaluation error:", evalErr);
                executedLogs.push(`❌ Action Compiler Error: ${evalErr.message || String(evalErr)}`);
              }
            }
            break;

          default:
            break;
        }
      } catch (err: any) {
        executedLogs.push(`❌ Error executing command [${cmd.type}]: ${err.message || String(err)}`);
      }
    }

    return executedLogs;
  };

  const handleSendMessage = async (customText?: string) => {
    const textToSendRaw = customText ? customText.trim() : inputText.trim();
    if (!textToSendRaw || isGenerating) return;

    if (!customText) {
      setInputText('');
    }

    let textToSend = textToSendRaw;
    if (!customText) {
      if (inputMode === 'sql') {
         textToSend = `[SQL COMMAND]\n${textToSendRaw}`;
      } else if (inputMode === 'python') {
         textToSend = `[PYTHON/JS SCRIPT]\n${textToSendRaw}`;
      }
    }

    const newMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMsg]);
    setIsGenerating(true);

    try {
      // Build dataset context schema to share with the Gemini Bot
      const datasetContext = activeDataset ? {
        filename: activeDataset.filename,
        rowCount: activeDataset.rowCount,
        columns: activeDataset.columns.map(c => ({
          name: c.name,
          type: c.type,
          missingCount: c.missingCount,
          distinctCount: c.distinctCount,
          statistics: {
            mean: c.statistics.mean,
            median: c.statistics.median,
            mostCommon: c.statistics.mostCommon?.slice(0, 3)
          }
        })),
        sampleRows: activeDataset.rows.slice(0, 3)
      } : null;

      // Extract brief history of latest 8 messages to keep token counts tight
      const messageHistory = messages.slice(-8).map(m => ({
        role: m.role,
        content: m.content
      }));

      let data;
      try {
        const res = await fetch('/api/chat-bot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: textToSend,
            history: messageHistory,
            datasetContext,
            activeTab
          })
        });

        const resText = await res.text();
        if (resText.trim().startsWith('<!doctype html') || resText.trim().startsWith('<html') || !res.ok) {
          throw new Error('Fallback to local analyzer');
        }
        data = JSON.parse(resText);
      } catch (e) {
        console.warn('Backend server returned invalid or HTML response. running local copilot fallback...', e);
        data = getClientSideChatFallback(textToSend, activeDataset, activeTab);
      }
      
      // Execute any commands returned by the model
      const executedLogs = await executeCommandsList(data.commands || []);

      const botMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.message || `Processed your instruction successfully.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        executedCommands: executedLogs
      };

      setMessages(prev => [...prev, botMsg]);

      // Trigger temporary glowing system overlay notification on screen
      if (executedLogs.length > 0) {
        setLastExecutedTaskMsg(executedLogs.join(' | '));
        setTimeout(() => setLastExecutedTaskMsg(null), 5000);
      }

    } catch (err: any) {
      console.error(err);
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `⚠️ Failed to execute task: ${err.message || 'connection timeout'}. Please verify server pipelines are running.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSuggestedPromptClick = (promptStr: string) => {
    handleSendMessage(promptStr);
  };

  // Drag container Ref bounding calculations
  useEffect(() => {
    // Generate simple boundary constraints handler
  }, []);

  return (
    <>
      {/* Dynamic Screen Overlay for Executed Tasks */}
      <AnimatePresence>
        {lastExecutedTaskMsg && (
          <motion.div
            initial={{ opacity: 0, y: -25, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -25, scale: 0.95 }}
            className="fixed top-20 left-1/2 -translate-x-1/2 z-50 bg-[#10b981]/15 backdrop-blur-md border border-[#10b981]/30 text-[#10b981] font-mono text-xs px-5 py-3 rounded-full flex items-center gap-2 shadow-2xl tracking-tight leading-relaxed font-bold select-none max-w-lg text-center"
          >
            <div className="w-2 h-2 rounded-full bg-[#10b981] animate-ping" />
            <span>{lastExecutedTaskMsg}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* AssistiveTouch Floating Base Area */}
      <div 
        ref={dragContainerRef}
        className="fixed inset-0 pointer-events-none z-[9999] select-none overflow-hidden" 
        id="assistive_draggable_universe"
      >
        <AnimatePresence mode="wait">
          {!isChatOpen ? (
            /* AssistiveTouch Button Mode - Visible only when Chatbox is closed */
            <motion.div
              key="assistive-icon"
              style={{ x: mx, y: my, touchAction: 'none' }}
              initial={{ scale: 1, opacity: 1 }}
              exit={{ scale: 1, opacity: 0 }}
              className="fixed top-0 left-0 pointer-events-auto cursor-grab active:cursor-grabbing z-50 flex flex-col items-center justify-center p-1"
              drag
              dragMomentum={false}
              dragElastic={0.1}
              onDragStart={() => setIsDraggingState(true)}
              onDragEnd={handleDragEnd}
              onTap={() => {
                const chatWidth = 270;
                const chatHeight = 360;
                const pad = 12;

                let cx = mx.get();
                let cy = my.get();

                if (cx + chatWidth > window.innerWidth) {
                  cx = window.innerWidth - chatWidth - pad;
                }
                if (cx < pad) cx = pad;

                if (cy + chatHeight > window.innerHeight) {
                  cy = window.innerHeight - chatHeight - pad;
                }
                if (cy < pad) cy = pad;

                animate(mx, cx, { type: 'spring', stiffness: 400, damping: 30 });
                animate(my, cy, { type: 'spring', stiffness: 400, damping: 30 });
                setIsChatOpen(true);
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {/* iOS AssistiveTouch Iconic Circle Shell with continuous 3D Live Aura */}
              <div className="w-14 h-14 bg-[#0A0D14]/90 backdrop-blur-md rounded-full border border-slate-700/50 flex items-center justify-center shadow-[0_0_20px_rgba(59,200,200,0.25)] relative group overflow-hidden select-none animate-neon-breath preserve-3d">
                <div className="absolute inset-0 bg-[#070A13]/85 rounded-full pointer-events-none" />

                {/* Ambient dynamic grid or stardust backdrops */}
                <div className="absolute inset-0 opacity-40 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-indigo-500/20 via-transparent to-transparent pointer-events-none animate-quantum-pulsar" />

                {/* 3D Holographic Orbiting Rings (Continuous) */}
                <div className="absolute w-12 h-12 rounded-full border border-dashed border-indigo-400/40 animate-live-ring-1 pointer-events-none" style={{ transformStyle: 'preserve-3d' }} />
                <div className="absolute w-11 h-11 rounded-full border border-double border-teal-400/30 animate-live-ring-2 pointer-events-none" style={{ transformStyle: 'preserve-3d' }} />
                <div className="absolute w-12 h-12 rounded-full border border-dotted border-rose-450/30 animate-live-ring-3 pointer-events-none" style={{ transformStyle: 'preserve-3d' }} />

                {/* Middle Ring Holder */}
                <div className="w-11 h-11 rounded-full border border-slate-600/30 flex items-center justify-center bg-slate-900/40 relative z-10 preserve-3d">
                  
                  {/* Central Round AskDeepakAI Logo Design - Spinning in 3D perspective continuously */}
                  <div className="w-8 h-8 rounded-full bg-[#050811]/95 flex flex-col justify-between p-1.5 border border-slate-700/60 shadow-[0_0_15px_rgba(59,200,200,0.4)] relative z-20 select-none overflow-hidden animate-live-3d-spin preserve-3d">
                    
                    {/* Glossy Reflection Sweep across the 3D rotating coin */}
                    <div className="absolute top-0 bottom-0 w-[40%] bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 animate-shiny-sheen pointer-events-none" />

                    {/* Top Bar with 3D layer translated layout */}
                    <div className="h-[24%] w-full flex rounded-full overflow-hidden" style={{ transform: 'translateZ(4px)' }}>
                      <div className="w-[18%] h-full bg-[#1b5bd2]"></div>
                      <div className="w-[82%] h-full bg-[#3bc8c8]"></div>
                    </div>
                    {/* Middle Bar with 3D layer translated layout */}
                    <div className="h-[24%] w-full flex rounded-full overflow-hidden" style={{ transform: 'translateZ(8px)' }}>
                      <div className="w-[18%] h-full bg-[#3bc8c8]"></div>
                      <div className="w-[82%] h-full bg-[#ef7222]"></div>
                    </div>
                    {/* Bottom Bar with 3D layer translated layout */}
                    <div className="h-[24%] w-[64%] flex rounded-full overflow-hidden" style={{ transform: 'translateZ(6px)' }}>
                      <div className="w-[32%] h-full bg-[#dfa435]"></div>
                      <div className="w-[68%] h-full bg-[#b22038]"></div>
                    </div>

                    {/* Inner 3D Backlight ring */}
                    <div className="absolute inset-0 border border-white/5 rounded-full pointer-events-none" />
                  </div>
                </div>

                {/* Dynamic corner specular flare elements */}
                <span className="absolute top-1 left-2 w-1 h-1 rounded-full bg-white opacity-40 blur-[0.5px]" />
                <span className="absolute bottom-1 right-2 w-1.5 h-1.5 rounded-full bg-indigo-400 opacity-30 blur-[1px]" />

                {/* Ambient horizontal indicator glow */}
                <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-[#1b5bd2]/60 via-[#3bc8c8]/60 to-[#ef7222]/60 opacity-80 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
              </div>
              <span className="text-[9px] font-mono font-bold text-slate-400 mt-1.5 bg-[#0B0F19]/90 border border-slate-800/80 px-2 py-0.5 rounded-full select-none shadow-md hidden sm:inline tracking-wide">
                AskAI
              </span>
            </motion.div>
          ) : (
            /* Chat Box Mode (65% Scaler Size applied, Draggable across coordinates) */
            <motion.div
              key="assistive-chat"
              style={{ x: mx, y: my, touchAction: 'none' }}
              animate={{ scale: 1, opacity: 1 }}
              initial={{ scale: 0.8, opacity: 0 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{
                type: 'spring',
                damping: 25,
                stiffness: 220
              }}
              className="fixed top-0 left-0 w-[300px] h-[400px] bg-slate-900/70 border border-slate-700/50 rounded-3xl shadow-[0_0_40px_rgba(0,0,0,0.8),inset_0_1px_0_rgba(255,255,255,0.1)] flex flex-col z-50 pointer-events-auto overflow-hidden backdrop-blur-2xl select-text transition-shadow"
              id="assistive_chat_panel"
              drag
              dragMomentum={false}
              dragElastic={0}
              onDragStart={(e) => {
                // Only drag from the header element dragging target, exclude close buttons etc
                const target = e.target as HTMLElement;
                if (target.closest('button') || target.closest('svg')) {
                  e.preventDefault();
                  return;
                }
                setIsChatDragging(true);
              }}
              onDragEnd={handleChatDragEnd}
            >
              {/* Header Details */}
              <div
                className="bg-transparent px-4 py-3 border-b border-slate-700/50 flex items-center justify-between z-10 shrink-0 cursor-move select-none relative"
              >
                <div className="absolute top-0 right-0 w-[150px] h-[40px] bg-indigo-500/10 rounded-full blur-2xl pointer-events-none" />
                <div className="flex items-center gap-3 relative z-10">
                  <div className="w-7 h-7 bg-[#070A13] rounded-full flex flex-col justify-between p-1 border border-slate-700/40 relative select-none shadow-md shrink-0 overflow-hidden">
                    {/* Top Bar */}
                    <div className="h-[24%] w-full flex rounded-full overflow-hidden">
                      <div className="w-[18%] h-full bg-[#1b5bd2]"></div>
                      <div className="w-[82%] h-full bg-[#3bc8c8]"></div>
                    </div>
                    {/* Middle Bar */}
                    <div className="h-[24%] w-full flex rounded-full overflow-hidden">
                      <div className="w-[18%] h-full bg-[#3bc8c8]"></div>
                      <div className="w-[82%] h-full bg-[#ef7222]"></div>
                    </div>
                    {/* Bottom Bar */}
                    <div className="h-[24%] w-[64%] flex rounded-full overflow-hidden">
                      <div className="w-[32%] h-full bg-[#dfa435]"></div>
                      <div className="w-[68%] h-full bg-[#b22038]"></div>
                    </div>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 border border-[#0A0F1D] absolute bottom-0.5 right-0.5" />
                  </div>
                  <div>
                    <h4 className="font-extrabold text-[11px] text-white flex items-center gap-1 leading-none whitespace-nowrap">
                      AskAI <span className="text-[8px] text-teal-400 font-bold font-mono bg-[#3bc8c8]/10 px-1 rounded border border-[#3bc8c8]/20 shrink-0">DeepakAI</span>
                    </h4>
                    <p className="text-[8px] text-slate-400 mt-0.5 font-medium leading-none whitespace-nowrap">Interactive Data Copilot</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Close the chat and snap the AssistiveTouch button back to nearest screen edge
                    setIsChatOpen(false);
                    setTimeout(() => {
                      const midX = window.innerWidth / 2;
                      const paddingX = 16;
                      const buttonSize = 56;
                      let snapX = mx.get() < midX ? paddingX : window.innerWidth - buttonSize - paddingX;
                      animate(mx, snapX, { type: 'spring', stiffness: 400, damping: 30 });
                    }, 50);
                  }}
                  className="p-1 px-1.5 text-slate-400 hover:text-white hover:bg-slate-800/80 rounded transition-colors text-[10px] font-bold font-mono border border-slate-850 bg-[#070b14]/60 cursor-pointer"
                >
                  ✕
                </button>
              </div>

              {/* Chat Thread Messages Stream (65% Scaled Height & Text) */}
              <div className="flex-1 p-3 overflow-y-auto space-y-3 bg-transparent scrollbar-thin select-text" style={{ WebkitOverflowScrolling: 'touch' }}>
                {messages.map((m) => {
                  const isUser = m.role === 'user';
                  return (
                    <div 
                      key={m.id} 
                      className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-end gap-1.5`}
                    >
                      {!isUser && (
                        <div className="w-5.5 h-5.5 rounded-full bg-[#070A13] flex flex-col justify-between p-1 border border-slate-800 shrink-0 self-start mt-0.5 select-none overflow-hidden">
                          {/* Top Bar */}
                          <div className="h-[24%] w-full flex rounded-full overflow-hidden">
                            <div className="w-[18%] h-full bg-[#1b5bd2]"></div>
                            <div className="w-[82%] h-full bg-[#3bc8c8]"></div>
                          </div>
                          {/* Middle Bar */}
                          <div className="h-[24%] w-full flex rounded-full overflow-hidden">
                            <div className="w-[18%] h-full bg-[#3bc8c8]"></div>
                            <div className="w-[82%] h-full bg-[#ef7222]"></div>
                          </div>
                          {/* Bottom Bar */}
                          <div className="h-[24%] w-[64%] flex rounded-full overflow-hidden">
                            <div className="w-[32%] h-full bg-[#dfa435]"></div>
                            <div className="w-[68%] h-full bg-[#b22038]"></div>
                          </div>
                        </div>
                      )}
                      
                      <div className="space-y-1 max-w-[85%]">
                        <div 
                          className={`rounded-xl p-2.5 text-[10px] inline-block leading-relaxed border relative shadow-md ${
                            isUser 
                              ? 'bg-indigo-600 border-indigo-500 text-white rounded-br-none' 
                              : 'bg-slate-900/80 border-slate-800 text-slate-200 rounded-bl-none space-y-1'
                          }`}
                        >
                          <div className="whitespace-pre-wrap font-sans">
                            {m.content.split('\n').map((line, idx) => {
                              let formatted = line;
                              
                              // Translate markdowns bold text **abc** inside scaled thread
                              const boldRegex = /\*\*(.*?)\*\*/g;
                              const matches = [...formatted.matchAll(boldRegex)];
                              
                              if (matches.length > 0) {
                                return (
                                  <div key={idx} className="mb-0.5 leading-relaxed">
                                    {formatted.split(/\*\*.*?\*\*/g).map((part, pIdx) => {
                                      const boldText = matches[pIdx]?.[1];
                                      return (
                                        <React.Fragment key={pIdx}>
                                          {part}
                                          {boldText && <strong className="text-white font-extrabold">{boldText}</strong>}
                                        </React.Fragment>
                                      );
                                    })}
                                  </div>
                                );
                              }

                              if (formatted.trim().startsWith('-')) {
                                return (
                                  <li key={idx} className="list-disc ml-3 text-slate-350 mt-0.5 font-medium leading-normal">
                                    {formatted.replace('-', '').trim()}
                                  </li>
                                );
                              }

                              if (formatted.includes('`')) {
                                return (
                                  <div key={idx} className="mb-0.5 inline-block">
                                    {formatted.split('`').map((cPart, cIdx) => (
                                      cIdx % 2 === 1 
                                        ? <code key={cIdx} className="bg-slate-950 px-1 py-0.5 rounded text-amber-300 font-mono text-[9px] font-bold">{cPart}</code>
                                        : <span key={cIdx}>{cPart}</span>
                                    ))}
                                  </div>
                                );
                              }

                              return <p key={idx} className="mb-0.5 font-medium">{formatted}</p>;
                            })}
                          </div>

                          {m.executedCommands && m.executedCommands.length > 0 && (
                            <div className="mt-1.5 pt-1.5 border-t border-slate-800/85 font-mono text-[8.5px] space-y-0.5 text-emerald-450">
                              <span className="block font-bold text-slate-400 text-[8px] uppercase">⚡ Applied Actions:</span>
                              {m.executedCommands.map((log, lIdx) => (
                                <div key={lIdx} className="flex items-start gap-0.5 text-slate-350">
                                  <span className="text-emerald-400">✓</span>
                                  <span>{log}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                        <span className="text-[8px] text-slate-500 font-mono block px-1 text-right leading-none">
                          {m.timestamp}
                        </span>
                      </div>
                    </div>
                  );
                })}

                {/* Loading indicator scale down */}
                {isGenerating && (
                  <div className="flex justify-start items-center gap-1.5">
                    <div className="w-5.5 h-5.5 rounded-full bg-[#070A13] flex flex-col justify-between p-1 border border-slate-800 shrink-0 select-none overflow-hidden animate-spin">
                      <div className="h-[24%] w-full flex rounded-full overflow-hidden">
                        <div className="w-[18%] h-full bg-[#1b5bd2]"></div>
                        <div className="w-[82%] h-full bg-[#3bc8c8]"></div>
                      </div>
                      <div className="h-[24%] w-full flex rounded-full overflow-hidden">
                        <div className="w-[18%] h-full bg-[#3bc8c8]"></div>
                        <div className="w-[82%] h-full bg-[#ef7222]"></div>
                      </div>
                    </div>
                    <div className="bg-slate-900/60 border border-slate-850 rounded-xl rounded-bl-none p-2 px-3 inline-block">
                      <div className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 inline-block animate-bounce" />
                        <span className="w-1.5 h-1.5 rounded-full bg-teal-500 inline-block animate-bounce" />
                        <span className="text-[9px] text-slate-450 font-mono font-medium ml-1">Computing...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Suggestions Chips Slider Row */}
              <div className="py-1 px-2 bg-[#0a0f1d] border-t border-slate-800/65 flex gap-1.5 overflow-x-auto scrollbar-none shrink-0 select-none" id="chat_suggestions">
                {[
                  { label: '🪄 Plan CRM', prompt: 'Analyze this dataset with an EDA scan' },
                  { label: '🧹 Impute Mean', prompt: 'Fill missing values with mean' },
                  { label: '📊 Dashboard', prompt: 'Show the stakeholder dashboard' },
                  { label: '🔌 SQL Run', prompt: 'SELECT * FROM dataset WHERE tenure > 35' },
                  { label: '🔄 Reset', prompt: 'Reset dataset' }
                ].map((s, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={(e) => { e.stopPropagation(); handleSuggestedPromptClick(s.prompt); }}
                    className="px-2 py-1 rounded-full bg-slate-900 hover:bg-slate-850 text-[9px] text-slate-300 border border-slate-800 transition-colors cursor-pointer select-none font-medium whitespace-nowrap shrink-0"
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              {/* Mode Selector */}
              <div className="flex bg-[#070A13] border-t border-slate-800/80 shrink-0">
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setInputMode('english'); }}
                  className={`flex-1 py-1.5 text-[9px] font-bold uppercase transition-colors text-center border-r border-slate-800/50 ${inputMode === 'english' ? 'bg-indigo-600/20 text-indigo-300 relative after:absolute after:bottom-0 after:left-0 after:w-full after:h-[2px] after:bg-indigo-500' : 'text-slate-500 hover:text-slate-300'}`}
                >
                  Ask AI
                </button>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setInputMode('sql'); }}
                  className={`flex-1 py-1.5 text-[9px] font-bold uppercase transition-colors text-center border-r border-slate-800/50 ${inputMode === 'sql' ? 'bg-emerald-600/20 text-emerald-300 relative after:absolute after:bottom-0 after:left-0 after:w-full after:h-[2px] after:bg-emerald-500' : 'text-slate-500 hover:text-slate-300'}`}
                >
                  SQL
                </button>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setInputMode('python'); }}
                  className={`flex-1 py-1.5 text-[9px] font-bold uppercase transition-colors text-center ${inputMode === 'python' ? 'bg-amber-600/20 text-amber-300 relative after:absolute after:bottom-0 after:left-0 after:w-full after:h-[2px] after:bg-amber-500' : 'text-slate-500 hover:text-slate-300'}`}
                >
                  Python/JS
                </button>
              </div>

              {/* Chat Send Input Form */}
              <form 
                onSubmit={(e) => { e.preventDefault(); e.stopPropagation(); handleSendMessage(); }}
                className="p-1.5 bg-[#090D17] flex flex-col gap-1.5 shrink-0 relative select-none"
              >
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder={inputMode === 'english' ? 'Ask anything in simple English...' : inputMode === 'sql' ? 'User compact SQL command...' : 'User Python/JS script...'}
                    disabled={isGenerating}
                    className="flex-1 bg-[#060A10]/95 text-[10.5px] h-8 text-slate-100 placeholder-slate-500 border border-slate-800 flex-shrink focus:border-indigo-500 focus:outline-none rounded-xl py-1 px-2.5 transition-colors disabled:opacity-40"
                  />
                  <button
                    type="submit"
                    disabled={!inputText.trim() || isGenerating}
                    className={`w-8 h-8 items-center justify-center flex rounded-lg disabled:bg-slate-850 disabled:text-slate-600 text-white transition-all cursor-pointer shadow-md shrink-0 ${inputMode === 'english' ? 'bg-indigo-600 hover:bg-indigo-500' : inputMode === 'sql' ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-amber-600 hover:bg-amber-500'}`}
                  >
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </div>
              </form>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}

// Executes an LLM/user-generated dataset-transform function inside a fresh,
// sandboxed iframe (allow-scripts only, no allow-same-origin) instead of this
// page's own realm. Because the iframe gets an opaque origin, code running
// inside it cannot reach this page's document, cookies, or localStorage, and
// cannot navigate or deface the real page - it can only see the exact
// `dataset` value it is handed and return a replacement. This does not stop
// that code from exfiltrating the dataset slice it was given (it must see
// that data to transform it), but it removes the far larger blast radius of
// running attacker-influenced code with full access to the live session.
function runSandboxedDatasetJs(jsCode: string, dataset: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const sandboxHtml = [
      '<!DOCTYPE html><html><body><script>',
      'window.addEventListener("message", function(event) {',
      '  try {',
      '    var fn = new Function("dataset", "return (" + event.data.jsCode + ")(dataset);");',
      '    var result = fn(event.data.dataset);',
      '    parent.postMessage({ __sandboxResult: true, ok: true, result: result }, "*");',
      '  } catch (e) {',
      '    parent.postMessage({ __sandboxResult: true, ok: false, error: (e && e.message) ? e.message : String(e) }, "*");',
      '  }',
      '});',
      '</' + 'script></body></html>'
    ].join('\n');

    const iframe = document.createElement('iframe');
    iframe.setAttribute('sandbox', 'allow-scripts');
    iframe.style.display = 'none';
    iframe.srcdoc = sandboxHtml;

    let settled = false;
    const cleanup = () => {
      window.removeEventListener('message', onMessage);
      clearTimeout(timer);
      iframe.parentNode?.removeChild(iframe);
    };

    const onMessage = (event: MessageEvent) => {
      if (event.source !== iframe.contentWindow || !event.data || !event.data.__sandboxResult) return;
      if (settled) return;
      settled = true;
      cleanup();
      if (event.data.ok) resolve(event.data.result);
      else reject(new Error(event.data.error || 'Sandboxed execution failed'));
    };

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error('Sandboxed script execution timed out after 5s'));
    }, 5000);

    window.addEventListener('message', onMessage);

    iframe.onload = () => {
      iframe.contentWindow?.postMessage({ jsCode, dataset }, '*');
    };

    document.body.appendChild(iframe);
  });
}

function getClientSideChatFallback(userMsg: string, ds: any, activeTab: string) {
  const msgLower = userMsg.toLowerCase();
  let reply = "";
  let commands: any[] = [];

  const colNames = ds && Array.isArray(ds.columns) ? ds.columns.map((c: any) => c.name) : [];
  const colNamesList = colNames.join(', ');
  const rowCount = ds ? ds.rowCount : 0;

  // Prioritized SQL Command parser fallback
  if (msgLower.includes('select') || msgLower.includes('update') || msgLower.includes('delete') || msgLower.includes('insert') || msgLower.includes('alter') || msgLower.includes('multiply') || msgLower.includes('double') || msgLower.includes('calculate')) {
    let isQueryProcessed = false;
    let jsCode = "";
    let sqlQuery = userMsg;
    let explanation = "";

    // 1. SELECT query parsing (e.g., SELECT * FROM dataset WHERE Age > 30)
    if (msgLower.includes('select') && msgLower.includes('where')) {
      const match = msgLower.match(/where\s+(\w+)\s*(=|>|<|!=)\s*(['"]?[\w\s.-]+['"]?)/i);
      if (match) {
        const col = colNames.find((c: string) => c.toLowerCase() === match[1].toLowerCase()) || match[1];
        const op = match[2] === '=' ? '===' : match[2];
        const val = match[3].replace(/['"]/g, '').trim();
        const isNum = !isNaN(Number(val));
        const compareVal = isNum ? Number(val) : `'${val}'`;

        jsCode = `(dataset) => {
          const filteredRows = dataset.rows.filter(row => {
            const rowVal = row['${col}'];
            if (rowVal === undefined || rowVal === null) return false;
            return ${isNum ? 'Number(rowVal)' : 'String(rowVal).toLowerCase()'} ${op} ${isNum ? compareVal : `'${val.toLowerCase()}'`};
          });
          return {
            ...dataset,
            rows: filteredRows,
            rowCount: filteredRows.length
          };
        }`;
        explanation = `Filtered rows where "${col}" ${match[2]} "${val}"`;
        isQueryProcessed = true;
      }
    }

    // 2. UPDATE query parsing (e.g., UPDATE dataset SET churn = 1 WHERE tenure < 5)
    if (msgLower.includes('update') && msgLower.includes('set')) {
      const matchSet = msgLower.match(/set\s+(\w+)\s*=\s*([^where]+)/i);
      if (matchSet) {
        const colToUpdate = colNames.find((c: string) => c.toLowerCase() === matchSet[1].toLowerCase()) || matchSet[1];
        let expr = matchSet[2].trim();
        
        let whereCol: string | null = null;
        let whereOp = "===";
        let whereVal = "";
        let whereIsNum = false;

        const whereIndex = msgLower.indexOf('where');
        if (whereIndex !== -1) {
          const wherePart = userMsg.slice(whereIndex + 5).trim();
          const whereMatch = wherePart.match(/(\w+)\s*(=|>|<|!=)\s*(['"]?[\w\s.-]+['"]?)/i);
          if (whereMatch) {
            whereCol = colNames.find((c: string) => c.toLowerCase() === whereMatch[1].toLowerCase()) || whereMatch[1];
            whereOp = whereMatch[2] === '=' ? '===' : whereMatch[2];
            whereVal = whereMatch[3].replace(/['"]/g, '').trim();
            whereIsNum = !isNaN(Number(whereVal));
          }
        }

        jsCode = `(dataset) => {
          const updatedRows = dataset.rows.map(row => {
            const copy = { ...row };
            let shouldUpdate = true;
            if ('${whereCol || ''}') {
              const rowWhereVal = row['${whereCol || ''}'];
              if (rowWhereVal === undefined || rowWhereVal === null) {
                shouldUpdate = false;
              } else {
                const compVal = ${whereIsNum ? 'Number' : 'String'}(rowWhereVal);
                const targetComp = ${whereIsNum ? whereVal : `'${whereVal}'.toLowerCase()`};
                shouldUpdate = ${whereIsNum ? 'compVal' : 'compVal.toLowerCase()'} ${whereOp} targetComp;
              }
            }

            if (shouldUpdate) {
              let newVal = copy['${colToUpdate}'];
              const rawExpr = '${expr}';
              if (rawExpr.includes('+')) {
                const parts = rawExpr.split('+');
                const addVal = Number(parts[1].trim());
                newVal = isNaN(addVal) ? rawExpr.replace(/['"]/g, '') : Number(copy['${colToUpdate}']) + addVal;
              } else if (rawExpr.includes('-')) {
                const parts = rawExpr.split('-');
                const subVal = Number(parts[1].trim());
                newVal = isNaN(subVal) ? rawExpr.replace(/['"]/g, '') : Number(copy['${colToUpdate}']) - subVal;
              } else if (rawExpr.includes('*')) {
                const parts = rawExpr.split('*');
                const multVal = Number(parts[1].trim());
                newVal = isNaN(multVal) ? copy['${colToUpdate}'] : Number(copy['${colToUpdate}']) * multVal;
              } else if (!isNaN(Number(rawExpr))) {
                newVal = Number(rawExpr);
              } else {
                newVal = rawExpr.replace(/['"]/g, '');
              }
              copy['${colToUpdate}'] = newVal;
            }
            return copy;
          });

          return {
            ...dataset,
            rows: updatedRows
          };
        }`;
        explanation = `Updated values of "${colToUpdate}" matching query criteria`;
        isQueryProcessed = true;
      }
    }

    // 3. DELETE query parsing (e.g., DELETE FROM dataset WHERE age < 18)
    if (msgLower.includes('delete') && msgLower.includes('where')) {
      const match = msgLower.match(/where\s+(\w+)\s*(=|>|<|!=)\s*(['"]?[\w\s.-]+['"]?)/i);
      if (match) {
        const col = colNames.find((c: string) => c.toLowerCase() === match[1].toLowerCase()) || match[1];
        const op = match[2] === '=' ? '===' : match[2];
        const val = match[3].replace(/['"]/g, '').trim();
        const isNum = !isNaN(Number(val));
        const compareVal = isNum ? Number(val) : `'${val}'`;

        jsCode = `(dataset) => {
          const filteredRows = dataset.rows.filter(row => {
            const rowVal = row['${col}'];
            if (rowVal === undefined || rowVal === null) return true;
            const matchesCondition = ${isNum ? 'Number(rowVal)' : 'String(rowVal).toLowerCase()'} ${op} ${isNum ? compareVal : `'${val.toLowerCase()}'`};
            return !matchesCondition;
          });
          return {
            ...dataset,
            rows: filteredRows,
            rowCount: filteredRows.length
          };
        }`;
        explanation = `Deleted rows where "${col}" ${match[2]} "${val}"`;
        isQueryProcessed = true;
      }
    }

    if (!isQueryProcessed) {
      // Fallback custom mutator for general commands e.g. "double MonthlyCharges" or "multiply tenure by 10"
      let foundCol = colNames.find((c: string) => msgLower.includes(c.toLowerCase()));
      if (foundCol) {
        let scale = 1;
        if (msgLower.includes('double')) scale = 2;
        else if (msgLower.includes('triple')) scale = 3;
        else {
          const numMatch = msgLower.match(/\d+/);
          if (numMatch) scale = Number(numMatch[0]);
        }

        jsCode = `(dataset) => {
          const updatedRows = dataset.rows.map(row => {
            const copy = { ...row };
            if (copy['${foundCol}'] !== undefined) {
              copy['${foundCol}'] = Number(copy['${foundCol}']) * ${scale};
            }
            return copy;
          });
          return {
            ...dataset,
            rows: updatedRows
          };
        }`;
        explanation = `Scaled column "${foundCol}" by factor ${scale}`;
        isQueryProcessed = true;
      }
    }

    if (isQueryProcessed) {
      reply = `I have decoded your action request! Running dataset operations pipeline.\n- **Transformed query**: \`${sqlQuery}\`\n- **Database Action**: Executes dynamic row-set corrections seamlessly.`;
      commands.push({
        type: 'EXECUTE_DATASET_JS',
        jsCode,
        sqlQuery,
        explanation
      });
      commands.push({ type: 'SELECT_TAB', tab: 'clean' });
      return { message: reply, commands };
    }
  }

  if (msgLower.includes('drop') || msgLower.includes('remove column')) {
    // Find which column to drop
    const foundCol = colNames.find((c: string) => msgLower.includes(c.toLowerCase()));
    if (foundCol) {
      reply = `I have successfully analyzed your command to drop column. Dropping **"${foundCol}"** and updating active pipelines.`;
      commands.push({ type: 'DROP_COLUMN', column: foundCol });
      commands.push({ type: 'SELECT_TAB', tab: 'clean' });
    } else {
      reply = `I can drop columns for you, but I couldn't identify which column you wanted to drop. Available columns: ${colNamesList || 'No dataset loaded'}.`;
    }
  } else if (msgLower.includes('fill') || msgLower.includes('impute') || msgLower.includes('missing')) {
    const foundCol = colNames.find((c: string) => msgLower.includes(c.toLowerCase())) || colNames[0];
    let strategy = 'mean';
    if (msgLower.includes('median')) strategy = 'median';
    else if (msgLower.includes('zero') || msgLower.includes('0')) strategy = 'zero';
    else if (msgLower.includes('mode') || msgLower.includes('common')) strategy = 'mode';

    if (foundCol) {
      reply = `I am executing an data imputation task on column **"${foundCol}"** using the **"${strategy}"** strategy to clean the dataset.`;
      commands.push({ type: 'FILL_MISSING', column: foundCol, strategy });
      commands.push({ type: 'SELECT_TAB', tab: 'clean' });
    } else {
      reply = `Imputation can only be executed when a column is specified. Available columns: ${colNamesList || 'N/A'}`;
    }
  } else if (msgLower.includes('eda') || msgLower.includes('scan') || msgLower.includes('analyze') || msgLower.includes('exploratory')) {
    reply = "Starting Exploratory Data Scan and Statistical Analysis on active datasets using our intelligent analytics engine!";
    commands.push({ type: 'SELECT_TAB', tab: 'eda' });
    commands.push({ type: 'RUN_EDA_SCAN' });
  } else if (msgLower.includes('model') || msgLower.includes('predict') || msgLower.includes('ml') || msgLower.includes('train')) {
    // Intelligently infer target and features
    const targetCol = colNames.find((c: string) => msgLower.includes(c.toLowerCase()) && (c.toLowerCase().includes('target') || c.toLowerCase().includes('churn') || c.toLowerCase().includes('probability'))) || colNames[colNames.length - 1] || 'target';
    const features = colNames.filter((c: string) => c !== targetCol).slice(0, 4);
    const mType = msgLower.includes('class') || targetCol.toLowerCase().includes('churn') ? 'classification' : 'regression';
    
    reply = `I've configured and triggered an automated Machine Learning pipeline for you!\n- **Stage**: ML Modeling\n- **Target Column**: \`${targetCol}\`\n- **Features**: ${JSON.stringify(features)}\n- **Model Type**: \`${mType}\`\n\nTraining starting now...`;
    commands.push({ type: 'SELECT_TAB', tab: 'ml' });
    commands.push({ type: 'RUN_ML', targetColumn: targetCol, featureColumns: features, modelType: mType });
  } else if (msgLower.includes('dashboard') || msgLower.includes('chart') || msgLower.includes('metric') || msgLower.includes('slicer')) {
    reply = "Right away! Moving you to the **Stakeholder Dashboard** stage where you can filter columns and monitor business outcomes.";
    commands.push({ type: 'SELECT_TAB', tab: 'dashboard' });
  } else if (msgLower.includes('report') || msgLower.includes('brief') || msgLower.includes('pdf') || msgLower.includes('hub')) {
    reply = "Transitioning to **Strategic Reports Hub** stage. You can compile, view, and export executive analysis briefs here.";
    commands.push({ type: 'SELECT_TAB', tab: 'reports' });
  } else if (msgLower.includes('ingest') || msgLower.includes('upload') || msgLower.includes('csv')) {
    reply = "Opening **Data Ingestion** panel so you can upload or template a dataset.";
    commands.push({ type: 'SELECT_TAB', tab: 'ingest' });
  } else if (msgLower.includes('add column') || msgLower.includes('create column')) {
    let label = 'NewDimension';
    const parts = userMsg.split(/add column|create column/i);
    if (parts[1]) {
      const potentialName = parts[1].trim().split(' ')[0].replace(/[^a-zA-Z0-9_]/g, '');
      if (potentialName) label = potentialName;
    }
    reply = `I am executing a pipeline task to add a new column named **"${label}"** with default values. Checking structures...`;
    commands.push({ type: 'ADD_COLUMN', column: label, columnType: 'categorical', value: 'DefaultVal' });
    commands.push({ type: 'SELECT_TAB', tab: 'clean' });
  } else if (msgLower.includes('add row') || msgLower.includes('insert row')) {
    reply = `Instructing the pipeline studio to append a new default row with placeholder entries!`;
    commands.push({ type: 'ADD_ROW' });
    commands.push({ type: 'SELECT_TAB', tab: 'clean' });
  } else if (msgLower.includes('delete row') || msgLower.includes('remove row')) {
    const match = userMsg.match(/\d+/);
    const index = match ? parseInt(match[0]) : 0;
    reply = `Applying dataset correction: Deleting active row at index **#${index}**.`;
    commands.push({ type: 'DELETE_ROW', index });
    commands.push({ type: 'SELECT_TAB', tab: 'clean' });
  } else if (msgLower.includes('group by') || msgLower.includes('groupby')) {
    const foundCol = colNames.find((c: string) => msgLower.includes(c.toLowerCase())) || colNames[0];
    if (foundCol) {
      reply = `I am executing a group-by operation. Grouping the active dataset by **"${foundCol}"** and displaying aggregation breakdown in the Cleaning Studio.`;
      commands.push({ type: 'SELECT_TAB', tab: 'clean' });
    } else {
      reply = `I can group your columns for aggregate summaries, but please specify one from: ${colNamesList || 'N/A'}`;
    }
  } else if (msgLower.includes('reset') || msgLower.includes('restore') || msgLower.includes('original')) {
    reply = "I've reset the active worksheet back to its original raw state. All values restored successfully!";
    commands.push({ type: 'RESET_DATASET' });
    commands.push({ type: 'SELECT_TAB', tab: 'clean' });
  } else if (msgLower.includes('hello') || msgLower.includes('hi') || msgLower.includes('who are you') || msgLower.includes('creater') || msgLower.includes('deepak')) {
    reply = `Hello! I am **AskAI**, acting as your interactive **AskDeepakAI** co-pilot built by **Gorisi Deepak Reddy**. 
    
I can analyze your dataset, clean missing cells, add or delete rows and columns, compute group-by metrics, run ML prediction models, and manage tabs. Try prompts like:
- *"Add column PremiumCustomer"*
- *"Insert a blank row"*
- *"Delete row 3"*
- *"Group by Country"*
- *"Drop column PaymentMethod"*
- *"Impute missing Age with median"*
- *"Run classification models for target Churn"*
- *"Reset dataset"*`;
  } else {
    reply = `I have received your message: "${userMsg}". 
    
As your AI code copilot, I can read and write the active dataset! I can add/delete rows & columns, perform group values, and execute smart operations. Current dataset has **${rowCount}** rows with columns: ${colNamesList || 'None'}.`;
  }

  return { message: reply, commands };
}
