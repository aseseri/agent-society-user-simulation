"""
Prompt Engineering Utilities for User Modeling Agent

This module contains reusable prompt templates and builders for improving
the baseline agent's performance through Chain-of-Thought (CoT) reasoning.

Author: Evan Li
"""


def build_cot_prompt(user: str, business: str, review_similar: str) -> str:
    """
    Build a Chain-of-Thought (CoT) prompt for user review generation.
    
    This prompt guides the LLM through step-by-step reasoning to improve
    both rating accuracy and review quality.
    
    Args:
        user (str): User profile information (as string)
        business (str): Business information (as string)
        review_similar (str): Similar reviews for context (as string)
    
    Returns:
        str: Formatted CoT prompt
    """
    prompt = f'''
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
    return prompt


def build_baseline_prompt(user: str, business: str, review_similar: str) -> str:
    """
    Build the original baseline prompt for comparison.
    
    Args:
        user (str): User profile information (as string)
        business (str): Business information (as string)
        review_similar (str): Similar reviews for context (as string)
    
    Returns:
        str: Formatted baseline prompt
    """
    prompt = f'''
        You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

        You need to write a review for this business: {business}

        Others have reviewed this business before: {review_similar}

        Please analyze the following aspects carefully:
        1. Based on your user profile and review style, what rating would you give this business?
        2. Given the business details and your past experiences, what specific aspects would you comment on?

        Requirements:
        - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
        - Review text should be 2-4 sentences, focusing on your personal experience
        - Maintain consistency with your historical review style and rating patterns

        Format your response exactly as follows:
        stars: [your rating]
        review: [your review]
        '''
    return prompt

