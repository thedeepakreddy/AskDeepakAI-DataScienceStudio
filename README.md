# 🧠 AskDeepakAI - Data Scientist Agent
[![CI](https://github.com/thedeepakreddy/AskDeepakAI-DataScientist/actions/workflows/ci.yml/badge.svg)](https://github.com/thedeepakreddy/AskDeepakAI-DataScientist/actions/workflows/ci.yml)

> Interactive, telemetry-free Data Science Workstation & Automated Machine Learning modeling engine powered by Google Gemini.

---

## 🎯 The Problem

Most "AutoML" side projects fall into one of two traps: an LLM narrates plausible-sounding metrics instead of computing them, or the "platform" is a thin wrapper around a single hardcoded `sklearn` call. Neither actually does machine learning, and neither survives a technical interviewer asking "wait, where does that number come from?"

This project was rebuilt to close that gap. Every metric a user sees — accuracy, R², the cross-validation scores behind model selection, SHAP feature importances, KS-test drift statistics — is produced by a real Python microservice running real `scikit-learn` / `XGBoost` training and real `pandas` / `NumPy` analysis on the actual uploaded dataset. Google Gemini is deliberately confined to narration and UX (chat, plain-English summaries, code generation): its response schemas structurally omit numeric fields, so it is not just discouraged but *unable* to be the source of a number anyone could make a decision on. Where Gemini is unavailable, the app falls back to template narration built from those same real numbers rather than degrading silently.

---

## ⏱️ 10-Second Recruiter Summary
*   **What is it?** A Full-Stack Data Science, AI, and MLOps workstation that automates data ingestion, cleaning, EDA, and machine learning pipelines entirely in the browser.
*   **The Big Innovation:** Features dual interfaces. **Beginner Mode** abstracts complex math away for business users, while **Expert Mode** unlocks production-grade MLOps configurations, ETL pipeline exports, and Python FastAPI integrations.
*   **Tech Stack:** React (TypeScript, Vite, Tailwind, Recharts) frontend, Node.js/Express orchestration backend, Python (FastAPI, Scikit-learn, XGBoost) microservice, and Google Gemini API for semantic AI reasoning.
*   **Key Features:** Visual SQL builder, zero-telemetry local CSV processing, data drift monitoring (KS-Tests), Hyperparameter tuning, and a 3D-orbiting intelligent Chatbot assistant.

---

## 📖 In-Depth Overview

**AskDeepakAI Data Scientist** is an end-to-end full-stack data intelligence platform empowering data engineers, analysts, and business decision-makers to transform raw datasets into clean, analyzed, modeled, and stakeholder-ready resources. 

Unlike locked-down SaaS BI tools or heavy CLI-only Python environments, this project couples an incredibly responsive **React (TypeScript) visual workspace** with a secure **Node.js (Express) backend engine** and a dedicated **Python FastAPI MLOps Microservice**.

---

## 🎭 Two Experiences: Beginner vs. Expert Mode

The platform is dynamically architected to serve two entirely different user personas at the toggle of a switch in the header:

### 🟢 Beginner Mode (Streamlined for Business)
Designed for Product Managers, Students, and Business Analysts who need quick insights without getting bogged down by statistical jargon or code.
*   **Simplified Ingestion:** Focuses purely on easy drag-and-drop local CSV uploads.
*   **Automated Cleaning & EDA:** The AI acts as your pilot, automatically detecting schema issues, suggesting a business hypothesis, and displaying only top-level data health summaries.
*   **One-Click Auto-ML:** Train AI models with a single click. Complex parameters are hidden. Outputs are presented in simple "Good/Bad" business terms (e.g., plain English accuracy summaries).
*   **Basic Dashboarding:** The MLOps dashboard displays simple, reassuring green/red badges indicating whether data health is "Stable" or "Drifted", hiding the complex mathematics behind the scenes.

### 🔴 Expert Mode (Unrestricted MLOps & Data Engineering)
Unlocks the true power of the platform. Designed for Senior Data Scientists and ML Engineers who demand total control over their algorithms and data pipelines.
*   **Database Syncs & SQL:** Unlocks the Visual Pipeline Builder allowing direct connection to PostgreSQL and Snowflake databases via scheduled `node-cron` jobs, plus an interactive SQL Assistant for querying.
*   **Advanced Analytics Labs:** Unlocks the Hypothesis Lab, A/B Testing Interpreters, Feature Correlation Matrix, and multi-dimensional Distribution Charts.
*   **Granular ML Control & Leakage Audits:** Unlocks Custom Hyperparameter Tuning (Estimators, depths, CV ratios, tree constraints) and rigorous Data Leakage Risk monitors.
*   **Python MLOps & Data Drift:** Deploys Native Python compute microservices for training `scikit-learn` models. Unlocks advanced Recharts visualizations for **Data Drift**, comparing production inference payloads against training distributions via Kolmogorov-Smirnov (KS) statistical tests.
*   **Developer Exports:** Unlocks Executive PDF generation, ETL Python code snippet generation, customizable Legacy Dashboards, and real-time backend operational Terminal Logs.

---

## 🏢 Industry-Grade Architecture & Enterprise Readiness

This application is engineered to meet strict industry standards for reliability, security, and scalability:

*   **Offline-First & Zero-Telemetry Data Privacy:** CSV parsing and data transformation occur strictly client-side via secure Web Workers. No proprietary row-level data is ever transmitted to remote AI endpoints, preventing highly sensitive enterprise data leakage (PII/PHI).
*   **Microservice Decoupling:** The heavy lifting of statistical computation, inference generation, and hyperparameter cross-validation is offloaded to a designated Python FastAPI microservice, preserving the Node.js orchestrator strictly for state management and API proxying. 
*   **Production Code Generation:** It behaves as an ETL compiler. The platform doesn't just display insights; it generates 1-to-1 deployable Python scripts detailing explicit cleaning drops, target variables, and ML hyperparameters, immediately verifiable by Data Engineering teams.
*   **Continuous Data Drift Monitoring:** Implements production-grade Kolmogorov-Smirnov (KS) tests to instantly flag feature distribution shifts (Data Drift) between historical training baselines and real-world incoming inferences.

---

## 🛡️ Comprehensive Feature Breakdown

### Data Engineering (Automated Ingestion & Sync)
*   **External Database Connectors**: Integrates securely with Snowflake, PostgreSQL, and other cloud databases for automated data pooling.
*   **Cron-Scheduled Ingestion Jobs**: Implements a robust `node-cron` orchestrator to fetch data directly from databases at a given time interval.
*   **Visual Pipeline Builder**: Drag-and-drop React interface abstracting API connections. Users simply select a timeline dropdown and provide connection strings to generate a fully automated ingestion pipeline.

### Business Intelligence (BI & Semantic Modeling)
*   **AI-Driven SQL Semantic Modeling**: Interrogates incoming schema changes automatically, using Gemini to enforce column validation mapping, detecting NULL variance or data type changes before persisting ingestion.
*   **Interactive Multi-Metric Slicers**: Fast segment slicer panels let you slice complex numerical properties or categorical states dynamically.
*   **Dynamic Data Visualizations**: Real-time visual plots using `Recharts` output scatter point graphs, line summaries, bar groupings, and percentage-based pie aggregations.

### MLOps Workstation (Python Compute & Drift Monitoring)
*   **Native Python ML Microservice**: A brand new dedicated FastAPI inference and computation layer using `scikit-learn` and `XGBoost`. Exceeds API limits by keeping core algorithmic mathematics running natively.
*   **One-Click Deployment**: Directly from the React frontend, users hit '1-Click Deploy', which serializes models into persistent `joblib` artifacts via the backend API.
*   **Data Drift Monitoring**: Features an interactive visual data drift monitor powered by Recharts comparing real-time inference payload distributions against training data baseline splits via Kolmogorov-Smirnov statistical tests.

### Stage 1: Data Ingestion
*   **Zero-Telemetry CSV Parsing**: Fluid file processor handles your raw matrices and CSV uploads completely inside your browser locally, guaranteeing no sensitive files hit external remote storage.

### Stage 2: Cleaning Studio (Quality Audit & Resolution)
*   **Automated Quality Audit System**: Single-click diagnostic scan runs across your dataset to discover Missing Values, Outliers, and Identical Duplicates.
*   **Missing Value Imputations**: Resolve empty row issues interactively. You can Drop Missing values entirely, or intelligently Fill them (using Mean, Median, Mode, Forward Fill, or Backward Fill).
*   **Outlier Normalization**: Utilizes standard robust statistical boundaries (1.5x IQR method) to automatically trim or cap extreme numeric anomalies that skew predictive machines.
*   **Duplicate Elimination**: One-click deduplication instantly drops redundant rows to maintain uncorrupted data validity.

### Stage 3: Exploration (EDA & Hypotheses)
*   **Semantic Data Profiling**: Connects to Google Gemini to interpret column name contexts, synthesize underlying business vectors, and deliver descriptive data commentary natively mapping back to your initial business goal.
*   **Hypothesis Testing Lab**:
    *   **Auto-Generate Deep Hypotheses**: Click the generator and Gemini proposes testable business hypotheses with the real columns they involve, pre-wired to the same real statistical engine below.
    *   **Test Your Own Hypothesis**: Write your question in plain text (e.g., "Do customers with high tenure have lower churn rates?"), pick the two real columns it's about, and AskDeepakAI runs a genuine statistical test between them — Pearson correlation, a Welch's t-test, one-way ANOVA, or a chi-squared test of independence, chosen automatically from the columns' real data types via `scipy.stats` — then has Gemini narrate what the real result means in plain English.

### Stage 4: Machine Learning Modeling (AutoML)
*   **Feature Architect & Optimizer**: Easily review columns to tag high-importance predictive variables, set specific Target variables, or selectively uncheck noise variables. The AI acts as a co-pilot, pre-selecting optimal Target metrics.
*   **Interactive Simulation Parameters**: Custom hyperparameters let you seamlessly tune tree structural constraints, Random Forest iterations, estimator depths, and custom Cross-Validation Testing ratios (Test Splits).
*   **Algorithmic Benchmarking Outputs**: Depending on the target column class, the workspace auto-detects Regression or Classification algorithms natively. Generates visual scorecards of metrics including **R-squared ($R^2$)**, **Mean Squared Error (MSE)**, **Accuracy**, **Precision**, and **F1-Score**.
*   **Model Explainer Visualizer**: Generates interactive global feature importance metrics in a horizontal visualization, illustrating how specific parameters shift the specific algorithmic logic weights.

### Stage 5: Stakeholder Dashboard
*   **Interactive Multi-Metric Slicers**: Fast segment slicer panels let you slice complex numerical properties or categorical states dynamically.
*   **Dynamic Data Visualizations**: Real-time visual plots using `Recharts` output scatter point graphs, line summaries, bar groupings, and percentage-based pie aggregations.
*   **Business Translation Overlays**: Auto-generated strategic business indicators, metric impact translations, Risk Registers, and tactical recommendations designed for executives who don't read raw statistics.

### Stage 6: Reports Hub
*   **Data Dictionary Catalog**: Instantly renders out a well-typed data dictionary schema of the original and transformed feature mappings.
*   **ETL Code Exporter**: Generates 1-to-1 reproducible native Python Pipeline code mimicking the precise data cleaning drops, imputations, and predictive states currently cached in your application.
*   **Executive PDF Pipeline (Real-Data Business Briefing)**: Prints a polished, structured narrative — built from the actual dataset profile and real trained-model metrics, narrated by Gemini for a business audience — framing the ML mechanics as an executive business case briefing.

### 🤖 AskDeepakAI Chat & Interactive AssistiveTouch Bot
The application features a revolutionary interactive **3D-rotating robot assistant** orbiting the user's workspace on all screen layouts globally.
*   **AI Prompt in Simple English**: Formulate standard conversational prompts ("Clean the data", "Remove the 'Id' column"). The agent translates prompt intent mathematically and executes dataset updates instantly.
*   **Run Compact SQL Commands**: Execute direct logic commands directly against the frontend browser-state SQL-like runtime structures! Filter tables via user compact SQL directly in chat.
*   **Run Python / JS Scripts**: Directly issue code block commands through the input field to alter the current row mappings with your custom developer logic.
*   **3D Quantum Orbit Design**: The hovering bot incorporates multi-layered reflective graphics with active orbital ring visuals, responsive viewport-edge snapping physics, and simple interactive drag.

### 🌐 Workspace Adaptability & UX
*   **Beginner vs. Expert User Context Toggle**: Switch seamlessly from 'Beginner' to 'Expert' modes right from the header. This dynamically switches the complexity of visual data! In "Beginner mode", the MLOps dashboard shows simple green/red badge indicators for data drift health. In "Expert Mode", it reveals complex Recharts line charts and statistical distribution metrics (KS-Tests).
*   **Intelligent Stage Headers**: Contextually aware top-headers that provide a clear mission statement and description for every individual stage, anchoring the user on the specific task.
*   **Advanced Dynamic Pipeline Progression (Stage Navigation)**: A powerful left-centered sticky navigation pill and bottom page navigation footers that collapse neatly out of the way, tracing locked statuses, completed dependencies, and transitioning elegantly fluidly up and down the data transformation chain without complex paginations.
*   **Pipeline Progress Bar**: A sleek visual completion tracker showing absolute progression towards the stakeholder dashboard reporting.
*   **Mobile Top-Bar Experience**: Completely responsive to horizontal tablet arrays and vertical smartphone screens.

---

## 📐 System Architecture Flow

The workstation follows a modern client-server architecture, acting as an enterprise MLOps platform:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                             REACT FRONTEND                               │
│  - Visual Pipeline Builder                  - MLOps Real-time Dashboard  │
│  - Beginner/Expert Toggle                   - Recharts Visualizations    │
└─────────────────────┬───────────────────────────────▲────────────────────┘
                      │                               │
                      │ 1-Click Deploy / Config       │ Analytics, Drift Metrics,
                      │ Connection strings            │ AI summaries
                      ▼                               │
┌─────────────────────┴───────────────────────────────┴────────────────────┐
│                             NODE ORCHESTRATOR                            │
│  - Express server executing node-cron scheduling jobs                    │
│  - BI Semantic Modeling Engine (Data schema validation)                  │
└────────────┬────────────────────────┬────────────────────────────────────┘
             │                        │
  Cloud Data │                        │ Secure requests to APIs & ML microservice
             ▼                        ▼
┌─────────────────────────┐   ┌─────────────────────────────────────────┐
│     CLOUD DATA POOLS    │   │         PYTHON MLOPS MICROSERVICE       │
│ - PostgreSQL            │   │ - FastAPI Endpoints (/api/train)        │
│ - Snowflake             │   │ - Scikit-learn / XGBoost training       │
│                         │   │ - Data Drift Monitor (KS-Tests)         │
└─────────────────────────┘   └─────────────────────────────────────────┘
```

This isn't just a diagram — it's backed by a [pytest suite](mlops_service/tests) that asserts real metric ranges and cross-checks EDA numbers against independent pandas computation, and a [CI workflow](.github/workflows/ci.yml) that runs it on every push.

---

## 🎬 Live Demo

🔗 **Hosted demo:** not currently deployed — the fastest way to see it live is the [Docker one-liner](#-alternative-one-command-local-dev-with-docker) below (`docker compose up --build`, then open `localhost:3000`).

The clip below is a real, unedited run captured directly against the running application (locally, via the Docker setup above) — a sample dataset going through server-side EDA, real 5-model AutoML training (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and an MLP, ranked by cross-validation, not just a single test-set score), and real SHAP feature importance for the winning model. Nothing shown is scripted or mocked.

![Demo: real server-side EDA, multi-model AutoML comparison, and SHAP explainability](docs/screenshots/demo.gif)

<details>
<summary>Static screenshots</summary>

**Real, server-side EDA** — pandas/NumPy statistics (mean, median, skew, IQR outliers, missing-value report, correlation matrix) computed on the actual uploaded data, with an AI-written summary that is structurally barred from inventing its own numbers:
![Real server-side EDA](docs/screenshots/02-real-server-side-eda.png)

**Real multi-model AutoML comparison** — five candidate models trained and ranked by 5-fold cross-validation mean (with fold-to-fold std shown, not hidden), including the losers, with an explicit, visible reason for why the champion was chosen:
![Real AutoML model comparison](docs/screenshots/03-real-automl-model-comparison.png)

**Real SHAP explainability** — feature importance for the champion model computed via SHAP (TreeExplainer / LinearExplainer depending on the algorithm), not a hardcoded or LLM-guessed ranking:
![Real SHAP feature importance](docs/screenshots/04-real-shap-feature-importance.png)

</details>

📄 Curious what was actually hard to build (and what's still fake vs. real)? See **[CASE_STUDY.md](CASE_STUDY.md)**.

---

## 🛠️ Technology Stack

| Component | Technology | Version / Focus |
| :--- | :--- | :--- |
| **Frontend UI** | **React** | v19.0.x — Modern functional hooks with concurrent rendering |
| **Bundler / Build** | **Vite** | v6.x — Sub-millisecond dev starts with ES Module hot reloading |
| **Styles** | **Tailwind CSS** | v4.x — High-fidelity utility classes & themes |
| **Graphs & Charts** | **Recharts & Lucide** | Interactive visualizations, scatter plots, trend lines, SVG Icons |
| **Server Engine** | **Node.js / Express** | Compact REST API wrapper and static web asset middleware, proxies to the Python ML microservice |
| **AI Processing** | **Google GenAI SDK** | `@google/genai` v2.4.x — Gemini model API integrations, narration-only (see [The Problem](#-the-problem)) |
| **Server Compiler** | **esbuild** | Ultra-fast JS/TS server compiler bundling to standalone CommonJS |
| **ML Microservice** | **FastAPI** | Python compute engine — the source of every real number in the app |
| **Model Training** | **scikit-learn / XGBoost** | Real `Pipeline`s (imputation + scaling/encoding + estimator), 5 candidate models, k-fold cross-validation |
| **Explainability** | **SHAP** | `TreeExplainer` / `LinearExplainer` per algorithm — real feature attribution, not a guessed ranking |
| **Data Analysis** | **pandas / NumPy / SciPy** | Real EDA (correlation, skew, IQR outliers), KS-test drift detection |
| **Testing** | **pytest** | 21 tests: real metric-range assertions, EDA numbers cross-checked against independent pandas computation |
| **Lint / CI** | **ruff, tsc, GitHub Actions** | Lint + test + typecheck + build on every push — see the badge at the top of this README |
| **Containerization** | **Docker / docker-compose** | One-command local dev for the full stack — see [Local Development Setup](#-local-development-setup) |

---

## 🚀 Local Development Setup

To test, debug, or run the application on your computer:

### Prerequisites
*   **Node.js**: `v18.x` through `v22.x` (or later) recommended
*   **Google Gemini API Key**: Obtain a developer key from [Google AI Studio](https://aistudio.google.com/)

### Command Line Instruction

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/thedeepakreddy/AskDeepakAI-DataScientist.git
    cd AskDeepakAI-DataScientist
    ```

2.  **Install Node packages**:
    ```bash
    npm install
    ```

3.  **Configure Environment Variables**:
    Create a `.env` file in your project's root folder:
    ```bash
    touch .env
    ```
    Populate it with your Gemini API key:
    ```env
    GEMINI_API_KEY=your_actual_google_gemini_api_key_here
    NODE_ENV=development
    ```

4.  **Boot Development Servers (Frontend / Orchestrator)**:
    ```bash
    npm run dev
    ```
    *   This boots up the integrated Node-Vite development runtime port `3000`.

5.  **Boot the MLOps Python Microservice**:
    Open a second terminal window in the project directory:
    ```bash
    cd mlops_service
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
    ```
    *   This boots up the Native Compute Engine for the 1-Click Deploy endpoints.

### 🐳 Alternative: One-Command Local Dev with Docker

If you have Docker installed, you can skip steps 2, 4, and 5 above and run the whole stack (frontend/orchestrator + Python ML microservice) with a single command:

```bash
GEMINI_API_KEY=your_actual_google_gemini_api_key_here docker compose up --build
```

*   The web app is served at `http://localhost:3000` and the ML microservice at `http://localhost:8000`.
*   The two containers talk to each other over the Docker Compose network (`ML_SERVICE_URL=http://ml-service:8000`), so no manual wiring is needed.
*   Dockerfiles live at `Dockerfile` (Node frontend/orchestrator, multi-stage build) and `mlops_service/Dockerfile` (Python ML microservice).

---

## 🌐 Deploying to Render.com (Production)

> **Architectural Note**: Python apps use Streamlit; full-stack TypeScript apps require standard virtual runtimes. Because **AskDeepakAI** is designed using TypeScript, Node, and React, **Render Web Services** is the perfect hosting environment.

### 1. Push your latest code modifications to GitHub
```bash
git add .
git commit -m "Deploy production update for AskDeepakAI"
git push origin main
```

### 2. Set up your Web Service on Render
1.  Log into your **[Render.com Dashboard](https://dashboard.render.com/)**.
2.  Click **New +** and select **Web Service**.
3.  Connect your GitHub repository: `AskDeepakAI-DataScientist`.
4.  Configure the environment setup settings exactly like this:
    *   **Name**: `ask-deepak-ai-datascientist`
    *   **Runtime**: `Node`
    *   **Branch**: `main`
    *   **Build Command**: `npm install && npm run build`
    *   **Start Command**: `npm start`
    *   **Instance Type**: `Free` (or premium depending on requirements)

### 3. Inject Production Environmental Secret
Scroll down to **Environment Variables** inside the Render wizard and add this pair:
*   **Key**: `GEMINI_API_KEY`
*   **Value**: `[Paste your actual secret Google Gemini token here]`

Render will automatically pull the branch, compile your React application bundle, transpile the TypeScript server, and make it available at your public URL!

---

## 🗂️ Git Directory Structure (For GitHub & Render Deployment)

To deploy cleanly on Render, your GitHub repository should contain the following structure. 

> **A Note on Tailwind CSS**: You might notice there is no `tailwind.config.js`. This is intentional! We are using **Tailwind CSS v4.x** with Vite, which is fully configured via CSS variables inside `src/index.css`.

```text
AskDeepakAI-DataScientist/
├── .gitignore            <-- ✅ Ignores dist/ and node_modules/
├── .env.example          <-- ✅ Example Template (Commit safely)
├── package.json          <-- ✅ Packages & Render Build/Start Scripts
├── package-lock.json     <-- ✅ Locks dependency versions for Render
├── tsconfig.json         <-- ✅ TypeScript Configuration
├── vite.config.ts        <-- ✅ Vite & Tailwind v4 plugin config
├── server.ts             <-- ✅ Express NodeJS entrypoint wrapper
├── mlops_service/        <-- ✅ Native Python FastAPI compute engine
│   ├── main.py
│   └── requirements.txt
└── src/                  <-- ✅ UI React Pages, Styles, and Data Components
    ├── index.css         <-- Contains Tailwind v4 @theme directives
    ├── App.tsx 
    └── components/

## 🚫 Files to IGNORE in GitHub (Add to .gitignore)
├── .env                  <-- ❌ SECRETS (Never commit your Gemini API Key)
├── node_modules/         <-- ❌ Heavy folder, let Render run 'npm install'
└── dist/                 <-- ❌ Compiled folder, let Render run 'npm run build'
```

---

*Engineered with precision. Powered by Google Gemini.*
