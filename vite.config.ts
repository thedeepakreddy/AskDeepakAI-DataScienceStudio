import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    'process.env.GOOGLE_MAPS_PLATFORM_KEY': JSON.stringify(process.env.GOOGLE_MAPS_PLATFORM_KEY || '')
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
  server: {
    hmr: process.env.DISABLE_HMR !== 'true',
    // deepakllm_training_data.jsonl is a runtime log the server appends to while
    // the app is used, not source code - without this it triggers a disruptive
    // full-page reload (and lost UI state) on every chat/training interaction.
    watch: process.env.DISABLE_HMR === 'true' ? null : { ignored: ['**/deepakllm_training_data.jsonl'] },
  },
});
