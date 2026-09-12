/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Power, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';

type WakeState = 'idle' | 'waking' | 'online' | 'error';

// Render's free tier puts an instance to sleep after ~15 minutes without
// traffic. Re-probing a little inside that window keeps it warm for as long as
// the toggle is on, so the first real training request never pays a cold start.
const KEEP_WARM_INTERVAL_MS = 10 * 60 * 1000;

export default function MLServiceWakeButton() {
  const [state, setState] = useState<WakeState>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [detail, setDetail] = useState<string>('');

  // Held in refs so the timers survive re-renders and are always cleanable.
  const keepWarmRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const clearTimers = useCallback(() => {
    if (keepWarmRef.current) { clearInterval(keepWarmRef.current); keepWarmRef.current = null; }
    if (tickRef.current) { clearInterval(tickRef.current); tickRef.current = null; }
  }, []);

  // Never leave a timer or an in-flight request behind on unmount.
  useEffect(() => () => {
    clearTimers();
    abortRef.current?.abort();
  }, [clearTimers]);

  const startKeepWarm = useCallback(() => {
    if (keepWarmRef.current) return;
    keepWarmRef.current = setInterval(() => {
      // Cheap no-retry probe purely to register activity against the host.
      fetch('/api/ml-service/status').catch(() => { /* transient - next tick retries */ });
    }, KEEP_WARM_INTERVAL_MS);
  }, []);

  const powerOff = useCallback(() => {
    clearTimers();
    abortRef.current?.abort();
    abortRef.current = null;
    setState('idle');
    setElapsed(0);
    setDetail('');
  }, [clearTimers]);

  const powerOn = useCallback(async () => {
    const controller = new AbortController();
    abortRef.current = controller;

    setState('waking');
    setDetail('');
    setElapsed(0);
    tickRef.current = setInterval(() => setElapsed(e => e + 1), 1000);

    try {
      // Runs entirely in the background - no navigation, no popup, nothing
      // rendered from the service's own response.
      const response = await fetch('/api/ml-service/wake', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });
      const body = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(body?.error || `Wake failed (HTTP ${response.status}).`);
      }

      if (tickRef.current) { clearInterval(tickRef.current); tickRef.current = null; }
      setState('online');
      setDetail(
        body?.coldStart
          ? `Cold start finished in ${Math.round((body.elapsedMs ?? 0) / 1000)}s. Staying warm while this is on.`
          : 'Service was already warm. Staying warm while this is on.'
      );
      startKeepWarm();
    } catch (err: any) {
      if (err?.name === 'AbortError') return; // user toggled off mid-flight
      if (tickRef.current) { clearInterval(tickRef.current); tickRef.current = null; }
      setState('error');
      setDetail(err?.message || 'Could not reach the ML service.');
    }
  }, [startKeepWarm]);

  const onClick = () => {
    if (state === 'waking' || state === 'online') powerOff();
    else powerOn();
  };

  const view = {
    idle:   { Icon: Power,        label: 'Wake ML Service', cls: 'border-[#3bc8c8]/20 bg-[#3bc8c8]/5 hover:bg-[#3bc8c8]/15 hover:border-[#3bc8c8]/40 text-[#3bc8c8]' },
    waking: { Icon: Loader2,      label: `Waking… ${elapsed}s`, cls: 'border-amber-400/30 bg-amber-400/10 text-amber-300' },
    online: { Icon: CheckCircle2, label: 'ML Service On', cls: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300' },
    error:  { Icon: AlertTriangle,label: 'Wake Failed', cls: 'border-rose-400/30 bg-rose-400/10 text-rose-300' },
  }[state];

  const { Icon } = view;

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={state === 'online'}
      aria-busy={state === 'waking'}
      title={
        detail ||
        (state === 'idle'
          ? 'Spin up the free-tier ML compute service in the background before training.'
          : 'Click to stop keeping the ML service warm.')
      }
      className={`flex items-center gap-2 px-4 py-2 rounded-xl border transition-all duration-300 text-xs font-semibold select-none cursor-pointer font-display ${view.cls}`}
    >
      <Icon className={`w-3.5 h-3.5 shrink-0 ${state === 'waking' ? 'animate-spin' : ''}`} />
      <span className="tracking-wide whitespace-nowrap">{view.label}</span>
      <span className="sr-only" aria-live="polite">{detail}</span>
    </button>
  );
}
