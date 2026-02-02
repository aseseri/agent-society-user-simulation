"""
This script runs the baseline agent on the Yelp dataset to establish
performance benchmarks for rating prediction and review generation.

Usage:
    python final_proj_code/run_baseline.py --num_tasks 10 --output results/baseline_run1.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

# ============================================================================
# feature flags, use these to control which improvements are enabled
# ============================================================================
USE_COT_PROMPTING = True  # Chain-of-Thought prompting
USE_ITERATIVE_RETRIEVAL = True  # Iterative Retrieval via query expansion
USE_SELECTIVE_MEMORY = True  # Selective review filtering before memory storage
USE_STYLE_ENFORCEMENT = True  # User pattern analysis for style enforcement

# prompt improvements
from final_proj_code.prompt_utils import build_cot_prompt, build_baseline_prompt
# iterative retrieval
from final_proj_code.retrieve_expanded_context import retrieve_expanded_context
# selective memory storage
from final_proj_code.smart_review_selection import select_top_k_reviews
# style enforcement
from final_proj_code.user_pattern_analysis import analyze_user_pattern

class PlanningBaseline(PlanningBase):
    """
    Simple baseline planning module that creates a fixed 2-step plan.
    
    This is intentionally simple - more advanced agents could create dynamic,
    context-aware plans based on the task. The baseline just follows the same
    steps for every task: get user info, then get business info.
    """
    
    def __init__(self, llm):
        """
        Initialize the planning module.
        
        Args:
            llm: The language model client (OpenAILLM) used for planning.
        """
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        """
        Create a simple 2-step plan for gathering information.
        
        Args:
            task_description (dict): Contains 'user_id' and 'item_id' from simulator.
            
        Returns:
            list: A plan with two steps - one for user info, one for business info.
        """
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """
    Simple baseline reasoning module that uses LLM to generate ratings and reviews.
    
    This module takes a fully-formed prompt (with user profile, business details,
    and example reviews) and sends it to the LLM to get a rating and review text.
    """
    
    def __init__(self, profile_type_prompt, llm):
        """
        Initialize the reasoning module.
        
        Args:
            profile_type_prompt (str): Optional profile-based prompt prefix (unused in baseline).
            llm: The language model client (OpenAILLM) used for reasoning.
        """
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """
        Use LLM to generate reasoning (rating and review).
        
        Args:
            task_description (str): The fully-formed prompt containing user profile,
                                   business details, and instructions.
                                   
        Returns:
            str: LLM response containing "stars: X.X" and "review: ..." lines.
        """
        prompt = '{task_description}'.format(task_description=task_description)
        messages = [{"role": "user", "content": prompt}]
        
        # Increase max_tokens for Chain-of-Thought reasoning if enabled
        max_tokens = 2000 if USE_COT_PROMPTING else 1000
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,      # Deterministic (no randomness)
            max_tokens=max_tokens
        )
        return reasoning_result


class BaselineAgent(SimulationAgent):
    """
    Baseline agent for user modeling.
    
    This agent simulates user behavior by:
    1. Planning: Creates a 2-step plan to gather user and business info
    2. Tool Use: Retrieves data from the simulator's InteractionTool
    3. Memory: Uses embeddings to find similar reviews for context
    4. Reasoning: Calls LLM to generate rating and review text
    
    Improvements are controlled via global feature flags at the top of this file.
    """
    
    def __init__(self, llm):
        """
        Initialize the baseline agent with modular components.
        
        Args:
            llm: The language model client (OpenAILLM) used by all modules.
        """
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)  
        
    def workflow(self):
        """
        Simulate user behavior: generate rating and review.
        
        This is the main execution flow that runs for each task. It follows these steps:
        1. Create a plan for information gathering
        2. Execute the plan to retrieve user and business data
        3. Use memory to find similar reviews for context
        4. Build a comprehensive prompt with all gathered information
        5. Call LLM to generate rating and review
        6. Parse and return the result
        
        Simulator-provided attributes:
            self.task (dict): Contains 'user_id' and 'item_id' for current task
            self.interaction_tool: Provides access to user, item, and review data
        
        Returns:
            dict: {"stars": float, "review": str} - The predicted rating and review text
        """
        try:
            # Step 1: Create a 2-step plan (get user info, get business info)
            plan = self.planning(task_description=self.task)  # self.task provided by simulator

            # Step 2: Execute the plan - gather user and business information
            user_stats_text = ""  # Style enforcement text
            for sub_task in plan:
                if 'user' in sub_task['description']:
                    # Get user profile (review history, average rating, etc.)
                    # self.interaction_tool is provided by the simulator
                    user_data = self.interaction_tool.get_user(user_id=self.task['user_id'])
                    # Analyze user pattern for style enforcement (before converting to string)
                    if USE_STYLE_ENFORCEMENT:
                        user_stats_text = analyze_user_pattern(user_data)
                    user = str(user_data)
                elif 'business' in sub_task['description']:
                    # Get business details (name, location, categories, etc.)
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            
            # Step 3: Use memory to find relevant context
            # Store reviews for this business in memory (for similarity search)
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            # Selective memory storage - filter reviews before storing
            if USE_SELECTIVE_MEMORY:
                selected_reviews = select_top_k_reviews(
                    business_reviews=reviews_item,
                    user_reviews=reviews_user,
                    llm=self.llm,
                    top_k=15
                )
                for review_text in selected_reviews:
                    if review_text and review_text.strip():
                        self.memory(f'review: {review_text}')
            else:
                for review in reviews_item:
                    self.memory(f'review: {review["text"]}')
            
            # Step 4: Get similar review from memory
            review_similar = ""
            
            # Use iterative retrieval for better context, or default single retrieval
            if USE_ITERATIVE_RETRIEVAL:
                review_similar = retrieve_expanded_context(
                    llm=self.llm,
                    user_reviews=reviews_user,
                    memory=self.memory,
                    num_queries=3, 
                    num_retrieved_per_query=2
                )
            else:
                review_similar = self.memory(f'{reviews_user[0]["text"]}')  # Default fallback
            
            # Step 5: Build prompt using appropriate strategy
            # Check global feature flags for improvements
            if USE_COT_PROMPTING:
                task_description = build_cot_prompt(user, business, review_similar)
            else:
                task_description = build_baseline_prompt(user, business, review_similar)
            
            # Append style enforcement instructions to prompt
            if USE_STYLE_ENFORCEMENT and user_stats_text:
                task_description = task_description + "\n" + user_stats_text
            
            # Step 6: Call LLM to generate rating and review
            result = self.reasoning(task_description)
            
            # Step 7: Parse the LLM response to extract rating and review text
            try:
                # Extract lines containing "stars:" and "review:"
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error parsing LLM response:', result)
                return {"stars": 3.0, "review": ""}  # Return default values on error

            # Parse the rating and review text
            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            # Truncate review if too long (simulator constraint)
            if len(review_text) > 512:
                review_text = review_text[:512]
                
            # Return the final result - this will be compared against ground truth
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            # If any error occurs in the workflow, return default values
            print(f"Error in workflow: {e}")
            return {"stars": 3.0, "review": ""}  # Default neutral rating and empty review


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline agent for Track 1")
    parser.add_argument('--data_dir', type=str, default='data/processed/yelp',
                       help='Path to processed data directory')
    parser.add_argument('--task_dir', type=str, default='AgentSocietyChallenge/example/track1/yelp/tasks',
                       help='Path to task directory')
    parser.add_argument('--groundtruth_dir', type=str, default='AgentSocietyChallenge/example/track1/yelp/groundtruth',
                       help='Path to groundtruth directory')
    parser.add_argument('--num_tasks', type=int, default=10,
                       help='Number of tasks to run (None for all)')
    parser.add_argument('--output', type=str, default='results/baseline_evaluation.json',
                       help='Output file for evaluation results')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='OpenAI model to use')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    return parser.parse_args()


def main():
    """
    Main execution function for running the baseline agent.
    
    This function:
    1. Loads configuration and API keys
    2. Initializes the Simulator with processed Yelp data
    3. Sets up tasks and ground truth data for evaluation
    4. Configures the BaselineAgent with OpenAI LLM
    5. Runs the simulation on the specified number of tasks
    6. Evaluates results and saves metrics to file
    
    The simulation runs each task through the agent's workflow, which generates
    a rating and review. Results are compared against ground truth to compute
    quality metrics.
    """
    args = parse_args()
    
    # Check for API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please set your API key in .env file or export OPENAI_API_KEY=your-key")
        sys.exit(1)
    
    print("=" * 60)
    print("BASELINE AGENT - TRACK 1 (USER MODELING)")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Task directory: {args.task_dir}")
    print(f"Model: {args.model}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    # add to this for debugging
    print("Active Improvements:")
    print(f"  - CoT Prompting: {'ON' if USE_COT_PROMPTING else 'OFF'}")
    print(f"  - Iterative Retrieval: {'ON' if USE_ITERATIVE_RETRIEVAL else 'OFF'}")
    print(f"  - Selective Memory: {'ON' if USE_SELECTIVE_MEMORY else 'OFF'}")
    print(f"  - Style Enforcement: {'ON' if USE_STYLE_ENFORCEMENT else 'OFF'}")
    print("=" * 60)
    
    # Initialize the simulator with processed data
    # The simulator loads user.json, item.json, and review.json from data_dir
    simulator = Simulator(
        data_dir=args.data_dir,  # Path to processed Yelp data
        device=args.device,       # 'cpu' or 'cuda'
        cache=True                # Cache data for faster repeated access
    )
    
    # Set tasks and ground truth data for evaluation
    # Tasks contain (user_id, item_id) pairs
    # Ground truth contains actual ratings and reviews for comparison
    simulator.set_task_and_groundtruth(
        task_dir=args.task_dir,
        groundtruth_dir=args.groundtruth_dir
    )
    
    # Configure the agent class and LLM to use
    # The simulator will instantiate BaselineAgent with the provided LLM
    simulator.set_agent(BaselineAgent)
    simulator.set_llm(OpenAILLM(api_key=api_key, model=args.model))
    
    # Run the simulation - each task goes through the agent's workflow
    print("\nRunning simulation...")
    outputs = simulator.run_simulation(
        number_of_tasks=args.num_tasks,     # How many tasks to run (None = all)
        enable_threading=(args.workers > 1), # Parallel execution if workers > 1
        max_workers=args.workers             # Number of parallel workers
    )
    
    # Evaluate agent performance by comparing outputs to ground truth
    print("\nEvaluating results...")
    evaluation_results = simulator.evaluate()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(evaluation_results, indent=2))
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    
    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
    history_path = output_path.parent / f"{output_path.stem}_history.json"
    with open(history_path, 'w') as f:
        json.dump(evaluation_history, f, indent=4)
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()

