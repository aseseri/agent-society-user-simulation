import re

def retrieve_expanded_context(llm, user_reviews, memory, num_queries=3, num_retrieved_per_query=2):
    """
    Implements Iterative Retrieval via query expansion. Iterative Retrieval is a Retrieval-Augmented Generation (RAG) strategy
    that enhances the memory retrieval process by expanding the initial query into multiple semantically related queries. This
    approach aims to capture a broader context related to the user's review style and content.

    Args:
        llm (LLMBase): The LLM client (OpenAILLM) to use for query expansion
        user_reviews (list): List of the user's past reviews
        memory (MemoryDILU): Memory module for finding similar content
        num_queries (int): Number of expanded queries to generate
        num_retrieved_per_query (int): Number of reviews to retrieve per expanded query
        
    Returns:
        str: Aggregated retrieved context from similar reviews
    """
    if not user_reviews:
        return "No similar reviews found."

    # Step 1. Construct the initial query (based on the user's most recent review)
    initial_query = user_reviews[0]["text"]

    # Step 2. LLM prompt for query expansion
    prompt = f"""
    You are a query expansion engine. Your task is to generate {num_queries} semantically similar or related search queries based on the user's review style and content.

    Original User Review Text: {initial_query}

    Generate {num_queries} queries that would find other reviews written in a similar style or focusing on similar business aspects.
    List each new query on a separate line, preceded by a number (e.g., "1. Query Text").
    """
    llm_response = llm(
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.1,    # generate diverse queries
        max_tokens=1000     # maximum response length
    )

    # Step 3. Parse LLM response to get expanded queries
    expanded_queries = []
    for line in llm_response.split('\n'):
        # Filter: check if the line matches the numbered query format (e.g., "1. Query Text")
        if re.match(r'^\s*\d+\.\s*', line):
            # Extract and clean:
            # - split the line only at the first period,
            # - take the text that follows
            # - remove surrounding whitespace
            query_text = line.split('.', 1)[1].strip()
            # Add query to list of valid queries
            expanded_queries.append(query_text)
            # Stop if we have enough queries
            if len(expanded_queries) == num_queries:
                break
    
    # Fallback: ensure at least the initial query is used
    if not expanded_queries:
        expanded_queries = [initial_query]

    # Step 4. Execute secondary retrieval and aggregate results
    results = set()
    for query in expanded_queries:
        for _ in range(num_retrieved_per_query): 
            similar_review = memory(query)
            if similar_review and similar_review.strip() and similar_review not in results:
                results.add(similar_review)

    # Step 5. Aggregate and return
    if not results:
        return memory(initial_query)  # Fallback: initial query retrieval

    return '\n---\n'.join(list(results))