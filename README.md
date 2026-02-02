# AgentSociety: User Behavior Simulation Agent

**Team Project:** Evan, Rita, Abby, Kelly

## Project Overview
This project was developed for the **AgentSociety Challenge**, a framework for building Large Language Model (LLM) agents that simulate realistic societal behaviors.

Our objective was to build a **User Simulation Agent** capable of generating reviews and ratings that accurately reflect a specific user's historical preferences and writing style. By implementing advanced prompting and retrieval strategies, we significantly outperformed the competition baseline.


**Key Results:**
* **Review Generation Quality:** **+5.06%** improvement over baseline.
* **Preference Estimation:** **+4.37%** improvement over baseline.
* **Overall Quality:** **+4.72%** improvement over baseline.

## System Architecture

Our agent improves upon standard zero-shot generation using a modular architecture located in `src/`:

### 1. Chain-of-Thought (CoT) Reasoning
*File: `src/run_cot_agent.py`*
We implemented a structured **3-step reasoning process** to guide the LLM:
1.  **User Analysis:** Classifies the user (e.g., "Generous" vs. "Critical") based on historical rating distributions.
2.  **Item Assessment:** Evaluates the target business's reputation from public metrics.
3.  **Synthesis:** Determines the rating by interacting the user's tendency with the business's quality.

### 2. Iterative Retrieval (RAG)
*File: `src/retrieve_expanded_context.py`*
To capture broader context, we implemented **Query Expansion**. Instead of using a single raw query, the agent uses an LLM to generate multiple semantically related search queries. This allows the memory module to retrieve past reviews that share *thematic* similarities (e.g., "service speed") rather than just keyword overlaps.

### 3. Smart Context Selection
*File: `src/smart_review_selection.py`*
We addressed the context window limitation for users with extensive histories (100+ reviews) using a **Weighted Average Embedding** strategy.
* We calculate a user's "persona embedding" weighted by rating intensity (giving higher weight to 5-star and 1-star reviews).
* We filter and retain only the top-15 most relevant reviews for the prompt context.

### 4. Style Enforcement
*File: `src/user_pattern_analysis.py`*
We developed a standalone module to analyze raw user history and extract a "User Persona" without biasing the rating logic:
* **Verbosity:** Enforces sentence length consistency (e.g., "Write short, 1-2 sentence reviews").
* **Vocabulary:** Identifies and reuses the user's recurring signature phrases.

## Performance Metrics

We evaluated the agent using the AgentSociety Challenge benchmark metrics on the Yelp dataset.

| Metric | Baseline Score | Our Agent (CoT + SCS) | Improvement |
| :--- | :--- | :--- | :--- |
| **Review Generation** | 0.830 | **0.872** | **+5.06%** |
| **Preference Estimation** | 0.822 | **0.858** | **+4.37%** |
| **Overall Quality** | 0.826 | **0.865** | **+4.72%** |

## Repository Structure

* **`src/`**: Core agent logic.
    * `run_cot_agent.py`: Main entry point containing the `CoTAgent` class.
    * `user_pattern_analysis.py`: Logic for extracting writing style constraints.
    * `smart_review_selection.py`: Weighted embedding logic for context management.
    * `retrieve_expanded_context.py`: RAG logic for query expansion.
    * `prompt_utils.py`: Chain-of-Thought prompt templates.
* **`websocietysimulator/`**: The simulation environment framework (Required dependency).
* **`scripts/`**: Utilities.
    * `process_yelp_data.py`: Pipeline to clean and format raw Yelp Open Dataset files.
* **`docs/`**: Documentation.
    * `simulation_details.md`: Technical breakdown of the simulator inputs/outputs.
    * `presentation.pdf`: Presentation slides covering methodology and ablation studies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aseseri/agent-society-user-simulation.git
    cd agent-society-user-simulation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY=sk-...
    ```

## Usage

### 1. Data Setup
Download the [Yelp Open Dataset](https://www.yelp.com/dataset) and run the processing script:
```bash
python scripts/process_yelp_data.py --input_dir data/raw --output_dir data/processed

```

### 2. Run Simulation

To run the CoT Agent on the user simulation task:

```bash
python src/run_cot_agent.py --num_tasks 10 --output results/cot_run.json

```

---

*Based on the [AgentSociety Challenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge) framework.*
