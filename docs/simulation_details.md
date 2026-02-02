# How the AgentSociety Simulation Works

This document explains the complete simulation system for Track 1 (User Behavior Modeling) of the AgentSociety Challenge.

---

## Overview

The simulation is a **systematic evaluation framework** that tests how well an agent can predict user behavior (ratings and reviews). The agent is given a user-business pair and must predict what rating and review the user would give.

---

## The Three Key Inputs

### 1. Processed Data (`data/processed/yelp/`)

Three files that the simulator loads into memory:

#### `user.json` - One line per user

```json
{
  "user_id": "qVc8ODYU5SZjKXVBgXdI7w",
  "name": "Walker",
  "review_count": 585,
  "average_stars": 3.91,
  "useful": 7217,
  "funny": 1259,
  "cool": 5994
}
```

#### `item.json` - One line per business

```json
{
  "item_id": "tUFrWirKiKi_TAnsVWINQQ",
  "name": "Target",
  "city": "Tucson",
  "state": "AZ",
  "stars": 3.5,
  "review_count": 22,
  "categories": "Department Stores, Shopping, ..."
}
```

#### `review.json` - One line per review (connections between users and items)

```json
{
  "review_id": "...",
  "user_id": "...",
  "item_id": "...",
  "stars": 4.0,
  "text": "Great place! Love the service...",
  "date": "2023-01-15"
}
```

**InteractionTool APIs:**

The simulator loads these into `InteractionTool`, which provides APIs like:
- `get_user(user_id)` - Returns user profile
- `get_item(item_id)` - Returns business details  
- `get_reviews(user_id=...)` - Returns all reviews by a user
- `get_reviews(item_id=...)` - Returns all reviews for a business

---

### 2. Tasks (`AgentSocietyChallenge/example/track1/yelp/tasks/`)

Each task is a simple JSON file (e.g., `task_0.json`):

```json
{
  "type": "user_behavior_simulation",
  "user_id": "wAo7casDFsbUR4O8Vb3u8A",
  "item_id": "cVoyA9wrdF5E8OroBahxCg"
}
```

**What this means**: "Simulate what would happen if this user reviewed this business"

The task provides:
- Which user to simulate
- Which business they're reviewing
- **Nothing else!** The agent must figure out the rating and review text

---

### 3. Ground Truth (`AgentSocietyChallenge/example/track1/yelp/groundtruth/`)

Each ground truth file (e.g., `groundtruth_0.json`) contains the **actual** rating and review:

```json
{
  "stars": 2.0,
  "review": "I had mixed feelings about my experience with Mad Batter Bakery. While the cupcakes and desserts were indeed delicious..."
}
```

This is what **really happened** - the actual review the user wrote. The simulator compares the agent's output to this to compute accuracy.

---

## The Simulation Flow

Here's what happens when you run `simulator.run_simulation(number_of_tasks=5)`:

### For Each Task:

#### 1. Load Task
- Simulator reads `task_0.json`
- Extracts `user_id` and `item_id`

#### 2. Initialize Agent
- Creates a new instance of `BaselineAgent`
- Sets `agent.task = {"user_id": "...", "item_id": "..."}`
- Sets `agent.interaction_tool = InteractionTool` (provides data access)

#### 3. Run Agent Workflow
The agent's `workflow()` method executes:
- Agent uses `self.task` to know which user/item
- Agent calls `self.interaction_tool.get_user(...)` to get user profile
- Agent calls `self.interaction_tool.get_item(...)` to get business details
- Agent calls `self.interaction_tool.get_reviews(...)` to get historical reviews
- Agent uses memory to find similar content
- Agent calls LLM to generate rating and review
- Returns `{"stars": 4.0, "review": "Great food and service!"}`

#### 4. Store Output
- Simulator saves the agent's prediction

#### 5. Repeat
- Process next task

---

## The Evaluation Phase

After all tasks complete, `simulator.evaluate()` runs:

### 1. Load Ground Truth
- Reads all `groundtruth_X.json` files

### 2. Compare Predictions
- For each task, compares agent output vs ground truth

### 3. Compute Metrics

#### Preference Estimation (Rating Accuracy)
- Computes MAE (Mean Absolute Error) between predicted and actual ratings
- Example: Predicted 4.0, actual 5.0 → error of 1.0
- Lower error = better score
- Baseline achieves: **0.96 (96% accuracy)**

#### Review Generation Quality
Compares review text using:
- **Emotional Tone**: Does it match the sentiment?
- **Topic Similarity**: Does it discuss similar aspects?
- **Semantic Similarity**: Overall content similarity using embeddings
- Baseline achieves: **0.86 (86% quality)**

#### Overall Quality
- Weighted combination of both metrics
- Baseline achieves: **0.91 (91% overall)**

---

## What the Simulator Provides

The simulator provides these attributes to the agent:

| Attribute | Description |
|-----------|-------------|
| `self.task` | Dict with `user_id` and `item_id` for current task |
| `self.interaction_tool` | API to access user, item, and review data |
| `self.llm` | The LLM client to use (OpenAI, etc.) |
| `self.memory` | Memory module for similarity search |

---

## What the Agent Must Figure Out

The agent must determine:
- What information to gather
- How to process it
- What rating to predict (1.0 - 5.0)
- What review text to generate (2-4 sentences)


