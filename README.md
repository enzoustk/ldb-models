# 🏀 LDB Bayesian Prediction Engine

> **Context:** This is a specialized sports analytics project designed to model and simulate basketball matches from the **LDB (Liga de Desenvolvimento de Basquete)**. Unlike simple regression models, this system utilizes **Hierarchical Bayesian Modeling** to capture latent team strengths, quarter-specific dynamics, and possession (pace) evolution.

[![Stan](https://img.shields.io/badge/Probabilistic_Prog-Stan-red)](https://mc-stan.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![ArviZ](https://img.shields.io/badge/Inference-ArviZ-purple)](https://arviz-devs.github.io/arviz/)

---

### 📋 Project Overview
This repository hosts the research and development of a statistical engine capable of simulating LDB match outcomes. By decomposing the game into fundamental components (Possessions, Shot Attempts, and Efficiency) per quarter, the model aims to predict **Spreads** (Q1/Full Game) and **Totals** with probabilistic uncertainty intervals.

### 🛠️ The Stack
A heavy-duty data science stack focused on probabilistic programming and rigorous statistical inference.

* **Core Engine:** `Stan` (via `cmdstanpy`) - State-of-the-art MCMC sampling (NUTS).
* **Inference & Diagnostics:** `ArviZ` - For posterior analysis, trace plots, and LOO-CV (Leave-One-Out Cross-Validation).
* **Data Engineering:** `Pandas`, `NumPy` - Advanced data structuring (Long-format transformations).
* **Data Acquisition:** Custom Scrapers (`BeautifulSoup`/`Requests`) for extracting box scores from LNB official site.

### ⚡ Model Architecture (Stan)
The heart of the project is the `.stan` model, which treats a basketball game as a stochastic process:

1.  **Pace Modeling (AR-1 Process):**
    * Matches are modeled as a sequence of quarters.
    * Pace (possessions) is modeled using a **Negative Binomial** distribution with an **Auto-Regressive (AR-1)** component to capture momentum and game flow state.
2.  **Shot Generation (Hierarchical):**
    * **Attempts (2PA, 3PA, FTA):** Modeled via **Negative Binomial** regression, conditioned on Pace.
    * **Efficiency (2PM, 3PM, FTM):** Modeled via **Binomial** distributions (Make/Miss) given the attempts.
3.  **Latent Variables:**
    * Estimates separate `Offensive` and `Defensive` strengths for every team.
    * Includes `Home/Away` advantages and `Quarter-Specific` effects.

### 🔄 Workflow
The system operates in a research pipeline:

1.  **Scraping:** Fetches raw box-score data and schedules.
2.  **Structuring:** Transforms game logs into "Long Format" (Team x Period rows) suitable for Bayesian inference.
3.  **Sampling:** Runs HMC (Hamiltonian Monte Carlo) chains to approximate the posterior distribution of all parameters.
4.  **Simulation:** Generates thousands of synthetic matches using the posterior predictive distribution to determine EV+ lines.

### ⚠️ Current Status & Challenges
* **Iterative Development:** Currently in **V3**.
* **Shrinkage:** The hierarchical priors are aggressive, currently causing some "shrinkage to the mean" (undervaluing extreme favorites). Adjustments to `sd_pace` and `team_raw` priors are ongoing.
* **Replication:** Requires a configured C++ toolchain (CmdStan) to compile the model binaries.

---

### 📂 Directory Structure
* `models/`: Contains the `.stan` source code and compiled binaries.
* `scraping/`: Scripts for data collection.
* `api/`: Helper functions for data validation and fetching.

---
*Research & Development by [Your Name].*
