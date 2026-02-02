# Experiment Results

This folder contains the evaluation results for the baseline and improved prompting with Chain-of-Thought (CoT) agents.

## Files

### Baseline Agent
- `baseline_50.json` - Evaluation metrics for baseline agent (50 tasks)
- `baseline_50_history.json` - Detailed task-by-task results for baseline

### Chain-of-Thought Agent (v2)
- `cot_50_v2.json` - Evaluation metrics for CoT agent (50 tasks)
- `cot_50_v2_history.json` - Detailed task-by-task results for CoT

## Results Summary

| Metric | Baseline | CoT v2 | Improvement |
|--------|----------|--------|-------------|
| Rating MAE | 0.834 | 0.850 | -1.9% |
| Review Quality | 0.842 | 0.870 | +3.3% |
| Overall Quality | 0.838 | 0.860 | +2.6% |

## How to Run

### Baseline Agent
```bash
python final_proj_code/run_baseline.py --num_tasks 50 --output final_proj_code/experiments/baseline_50.json
```

### CoT Agent
```bash
python final_proj_code/run_cot_agent.py --num_tasks 50 --output final_proj_code/experiments/cot_50_v2.json
```

## CoT Implementation Details

The Chain-of-Thought agent uses a 3-step reasoning framework:

1. **Analyze Rating Behavior** - Identify if user is generous, balanced, or critical reviewer
2. **Assess Business Reputation** - Evaluate business rating and review sentiment
3. **Determine Rating & Write Review** - Use explicit logic to match reviewer type with business reputation

Key improvement: Explicit rating guidance with 9 scenarios covering different combinations of reviewer types and business ratings.

