import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';

// Non-type-checked recommended rules only (fast, no tsconfig project wiring
// needed in CI). This is a first pass for a codebase that had no linter
// history before Phase 3 - see README/CI workflow for why it currently runs
// as an informational, non-blocking check rather than a merge gate.
export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'mlops_service/**', 'app/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
);
