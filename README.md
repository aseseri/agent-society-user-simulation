# CS 245 Final Project – QUICK_START

**Track:** Option 1 · Track 1 – User Behavior Modeling (AgentSociety)  

---

## 0. Overview & Quick Start Checklist

### Project Overview

**Core Objective**: Build an LLM agent that simulates realistic user behaviors on review platforms (Yelp dataset).

**Task Details**:
- **Input**: For each task, your agent receives:
  - A `user_id` (representing a user)
  - An `item_id` (e.g., a restaurant on Yelp)
- **Output**: Your agent must generate:
  1. A star rating (1-5 scale)
  2. A realistic review text that the user would write

**Evaluation Metrics**:
1. **Rating Accuracy**: Mean Absolute Error (MAE) between predicted and ground truth ratings
2. **Review Quality**:
   - Emotional tone similarity
   - Sentiment attitude alignment
   - Topic relevance (measured via cosine similarity)
3. **Overall Score**: Average of preference (rating) and review quality components

**Technical Stack**:
- **Framework**: `websocietysimulator` package (AgentSocietyChallenge environment)
- **Python Version**: 3.11
- **Datasets**: Yelp (structured as `user.json`, `item.json`, `review.json`)
- **Architecture**: Modular agent design with Planning, Reasoning, Tool Use, and Memory modules

### Setup Checklist

1. **Setup Python Environment** - Install conda, create environment, install dependencies
2. **Download & Process Yelp Dataset** - Get raw data, process it, verify output files
3. **Configure Environment** - Set up `.env` with API keys
4. **Run Baseline** - Examine examples, update runner script, verify metrics

---

## 1. Python Environment Setup

### 1.1. Create and Activate the Environment

```bash
# Create conda environment with Python 3.11
conda create -n cs245_agentsociety python=3.11 -y

# Activate the environment
conda activate cs245_agentsociety

# Install dependencies
pip install -r requirements.txt

# Install the AgentSocietyChallenge framework
pip install -e ./AgentSocietyChallenge
```

### 1.2. Verify Installation

Run a quick test to confirm everything is installed correctly:

```bash
python -c "import websocietysimulator; print('websocietysimulator installed successfully')"
```

If you see the success message with no errors, your environment is ready!

**Next**: After completing all setup steps, read `simulation_explanation.md` for a detailed explanation of how the simulation system works.

---

## 2. Data Setup (Yelp Dataset)

The simulator expects processed data in this structure:

```
data/
├── raw/
│   └── yelp/          # Raw Yelp Open Dataset files
└── processed/
    └── yelp/          # Processed JSON files
        ├── user.json
        ├── item.json
        └── review.json
```

**Note**: The data directories contain `.gitkeep` files to preserve the folder structure in git (git doesn't track empty directories). These directories will exist automatically when you clone the repo, ready for you to add data files.

### 2.1. Download Yelp Open Dataset

Download the Yelp Open Dataset from the official source:
- **URL**: https://www.yelp.com/dataset
- download the json version 
- Place the downloaded files in `data/raw/yelp/`

You'll need files like:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

**Note**: Each team member should download the dataset individually. The raw data is too large to commit to git (~10GB).

### 2.2. Process the Raw Data

The raw Yelp data must be processed before use because:

1. **Field Naming**: The simulator expects `item_id` but Yelp uses `business_id`
2. **File Naming**: The simulator expects `item.json`, `user.json`, `review.json` with specific formats
3. **Data Filtering**: The full dataset is too large; we filter to the top 3 cities (Philadelphia, Tampa, Tucson)

Run the provided processing script:

```bash
python process_yelp_data.py --input_dir data/raw/yelp --output_dir data/processed/yelp
```

This will create three processed files in `data/processed/yelp/` ready for the simulator.

### 2.3. Verify Data Files

Confirm the processed files exist:

```bash
ls -lh data/processed/yelp/
# Should show: user.json, item.json, review.json
```

---

## 3. Configuration

### 3.1. Environment Variables (`.env` file)

Create a `.env` file for API keys and secrets:

```bash
# Copy the template
cp .env.example .env

# Then edit .env and add your actual API key
nano .env  # or use your preferred editor
```

Your `.env` file should look like:
```bash
# .env (do not commit to git!)
OPENAI_API_KEY=sk-proj-abc123...  # Your actual key here
DATA_DIR=data/processed/yelp
```

**Important**: 
- ⚠️ `.env` is in `.gitignore` - it will NOT be committed to git (keeps your API keys secret)
- ✅ `.env.example` IS committed - it's a template for your team
- 🔑 Never share your `.env` file or commit it to git!

### 3.2. Non-Secret Configuration (`config.yaml`)

The `config.yaml` file contains non-secret settings like model parameters and paths:

```yaml
model:
  name: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 500

simulation:
  num_tasks: 100
  device: cpu
```

**Key differences**:
- **`.env`**: Secrets (API keys) - NOT committed to git
- **`config.yaml`**: Settings (model names, parameters) - IS committed to git

You can modify `config.yaml` to experiment with different settings and share those configurations with your team.

---

## 4. Running the Baseline Agent

### 4.1. Examine Repository Examples

The AgentSocietyChallenge repository includes example agents in the `examples/` or `agents/` directory. Look for:
- User modeling agent examples
- Sample runner scripts
- Configuration files

### 4.2. Update Baseline Runner Script

The baseline runner template is at `final_proj_code/run_baseline.py`. Update it based on the repository examples:

**Key updates needed**:
1. Fix imports based on actual agent class in examples
2. Update agent initialization parameters
3. Verify simulator API matches repository

### 4.3. Run the Baseline

```bash
python final_proj_code/run_baseline.py
```

Expected output:
- Task execution progress
- Evaluation metrics (MAE, sentiment scores, topic similarity)
- Overall performance score

### 4.4. Troubleshooting

Common issues:
- **Import errors**: Ensure `pip install -e .` was run successfully
- **API key errors**: Verify `OPENAI_API_KEY` is set
- **Data errors**: Confirm processed data files exist in `data/processed/yelp/`
- **Memory issues**: Reduce `num_tasks` or use a smaller data subset

---

## Understanding the Simulation

**Before moving forward**, read `simulation_explanation.md` to understand:
- How the simulation system works
- What inputs and outputs the agent receives
- How evaluation metrics are computed
- Why this challenge is difficult

This will help you design better agents!

---

## 5. Next Steps & Team Milestones

### 5.1. Week 1 Goals (Setup Phase)

By the end of Week 1, each teammate should:

✓ Have the AgentSocietyChallenge repository cloned  
✓ Have the `cs245_agentsociety` environment set up and verified  
✓ Have processed Yelp data available locally (or access to shared processed data)  
✓ Successfully run the baseline script and obtain evaluation metrics  
✓ Record baseline metrics in our shared document

**Baseline Metrics to Record**:
- Rating MAE (lower is better)
- Review Quality Score (higher is better)
- Overall Score
- Runtime per task
- Any errors or issues encountered

### 5.2. Week 2-3: Custom Agent Development

After the setup phase, we'll focus on:

1. **Design Custom Agent Architecture**
   - Implement `CS245UserModelingAgent` in `final_proj_code/agents/`
   - Add modular components: Planning, Memory, Reasoning, Tool Use

2. **Enhance Agent Capabilities**
   - **Memory Module**: Store and retrieve user history, preferences
   - **Planning Module**: Multi-step reasoning for rating/review generation
   - **Context Retrieval**: Leverage similar past reviews
   - **Reflection**: Self-improvement based on feedback

3. **Experimentation**
   - Run ablation studies (with/without each module)
   - Compare against baseline
   - Optimize prompts and strategies

4. **Documentation & Report**
   - Track all experiments and results
   - Write final project report
   - Prepare presentation slides

### 5.3. Quick Reference Commands

```bash
# Activate environment
conda activate cs245_agentsociety

# Run baseline (with 10 tasks)
python final_proj_code/run_baseline.py --num_tasks 10

# Run custom agent (to be created)
python final_proj_code/run_custom_agent.py

# Process new data
python AgentSocietyChallenge/data_process.py --raw_dir data/raw/yelp --output_dir data/processed/yelp
```

---

## 6. Resources

- **AgentSocietyChallenge**: https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/
- **Yelp Dataset**: https://www.yelp.com/dataset
- **GCP Credits**: Use provided link for GPU access
- **Team Document**: [Add link to shared Google Doc/Notion]