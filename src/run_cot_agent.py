"""
This script runs an enhanced agent with Chain-of-Thought (CoT) prompting
on the Yelp dataset for improved rating prediction and review generation.

CoT Enhancement: Prompts the LLM to think step-by-step through the reasoning
process before generating the final rating and review.

Usage:
    python final_proj_code/run_cot_agent.py --num_tasks 10 --output results/cot_run1.json
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


class ReasoningCoT(ReasoningBase):
    """
    Chain-of-Thought (CoT) reasoning module that prompts LLM to think step-by-step.
    
    Enhancement over baseline: Uses explicit step-by-step reasoning prompts to guide
    the LLM through analyzing user patterns, understanding the business, predicting
    ratings, and then generating the review.
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
        
        # Call OpenAI API with deterministic settings
        # Increased max_tokens for Chain-of-Thought reasoning steps
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,      # Deterministic (no randomness)
            max_tokens=1500       # Increased to accommodate CoT reasoning
        )
        return reasoning_result


class CoTAgent(SimulationAgent):
    """
    Chain-of-Thought (CoT) enhanced agent for user modeling.
    
    This agent simulates user behavior by:
    1. Planning: Creates a 2-step plan to gather user and business info
    2. Tool Use: Retrieves data from the simulator's InteractionTool
    3. Memory: Uses embeddings to find similar reviews for context
    4. Reasoning: Uses CoT prompting to guide LLM through step-by-step analysis
    
    Enhancement over baseline: The reasoning module now explicitly prompts the LLM
    to think through user patterns, business analysis, and rating prediction before
    generating the final review.
    """
    
    def __init__(self, llm):
        """
        Initialize the CoT agent with modular components.
        
        Args:
            llm: The language model client (OpenAILLM) used by all modules.
        """
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningCoT(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)  # Memory module for finding similar content
        
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
            for sub_task in plan:
                if 'user' in sub_task['description']:
                    # Get user profile (review history, average rating, etc.)
                    # self.interaction_tool is provided by the simulator
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    # Get business details (name, location, categories, etc.)
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            
            # Step 3: Use memory to find relevant context
            # Store all reviews for this business in memory (for similarity search)
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                self.memory(f'review: {review["text"]}')
            
            # Get the user's review history and find the most similar review
            # This helps the agent understand the typical review style for this business
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            
            # Step 4: Build a Chain-of-Thought prompt with simplified steps and explicit rating guidance
            # CoT Enhancement: 3 focused steps + concrete rating examples
            task_description = f'''
You are a real human user on Yelp. Here is your profile and history: {user}

You need to write a review for this business: {business}

Here are examples of others' reviews for this business: {review_similar}

Let's think step-by-step:

STEP 1 - Analyze Your Rating Behavior:
Look at your average rating in your profile. This tells you if you're:
- Generous reviewer (avg 4.0+ stars) → you tend to give 4.0 or 5.0 stars
- Balanced reviewer (avg 3.0-4.0 stars) → you give varied ratings based on experience  
- Critical reviewer (avg below 3.0 stars) → you tend to give 1.0-3.0 stars

STEP 2 - Assess This Business's Reputation:
Look at the business's average rating and read the example reviews:
- High-rated (4.0+ stars): Most reviews are positive, praise quality/service
- Mid-rated (3.0-4.0 stars): Mixed reviews, some issues mentioned
- Low-rated (below 3.0 stars): Mostly negative, complaints about quality/service

STEP 3 - Determine Your Rating Using This Logic:
Match YOUR tendency with the business's reputation:

If you're a GENEROUS reviewer:
- High-rated business (4.0+) → Give 5.0 stars (you love good places)
- Mid-rated business (3.0-4.0) → Give 4.0 stars (still positive despite flaws)
- Low-rated business (below 3.0) → Give 3.0 stars (disappointed but not harsh)

If you're a BALANCED reviewer:
- High-rated business (4.0+) → Give 4.0 or 5.0 stars (deserves the reputation)
- Mid-rated business (3.0-4.0) → Give 3.0 or 4.0 stars (matches experience)
- Low-rated business (below 3.0) → Give 2.0 or 3.0 stars (fair criticism)

If you're a CRITICAL reviewer:
- High-rated business (4.0+) → Give 3.0 or 4.0 stars (good but not perfect)
- Mid-rated business (3.0-4.0) → Give 2.0 or 3.0 stars (you see the flaws)
- Low-rated business (below 3.0) → Give 1.0 or 2.0 stars (unacceptable)

Then write a 2-4 sentence review that matches your rating:
- High rating (4.0-5.0) → Positive review praising good aspects
- Medium rating (3.0) → Balanced review with pros and cons
- Low rating (1.0-2.0) → Critical review mentioning problems

Requirements:
- Star rating MUST be exactly one of: 1.0, 2.0, 3.0, 4.0, 5.0
- Review text should match your rating sentiment
- Write in your typical style (length, tone, details)

Format your final response as:
stars: [your rating]
review: [your review]
'''
            # Step 5: Call LLM to generate rating and review
            result = self.reasoning(task_description)
            
            # Step 6: Parse the LLM response to extract rating and review text
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
    parser.add_argument('--output', type=str, default='results/cot_evaluation.json',
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
    Main execution function for running the Chain-of-Thought (CoT) agent.
    
    This function:
    1. Loads configuration and API keys
    2. Initializes the Simulator with processed Yelp data
    3. Sets up tasks and ground truth data for evaluation
    4. Configures the CoTAgent with OpenAI LLM and CoT prompting
    5. Runs the simulation on the specified number of tasks
    6. Evaluates results and saves metrics to file
    
    The CoT agent uses step-by-step reasoning prompts to guide the LLM through
    analyzing user patterns, understanding the business, predicting ratings, and
    generating reviews. Results are compared against ground truth to compute
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
    print("CHAIN-OF-THOUGHT (CoT) AGENT - TRACK 1 (USER MODELING)")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Task directory: {args.task_dir}")
    print(f"Model: {args.model}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"Workers: {args.workers}")
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
    # The simulator will instantiate CoTAgent with the provided LLM
    simulator.set_agent(CoTAgent)
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

