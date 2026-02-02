import numpy as np


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def calculate_average_embedding(user_reviews, llm, top_k=10):
    """
    Calculate weighted average embedding of user reviews.
    Uses rating as weight to emphasize higher-rated reviews.
    """
    embedding_model = llm.get_embedding_model()
    if embedding_model is None:
        return None
    
    # Extract user review texts and ratings
    user_review_data = []
    for review in user_reviews[:top_k]:
        if isinstance(review, dict):
            text = review.get("text", "")
            stars = review.get("stars", 3.0)  # Default to 3.0 if missing
        else:
            text = str(review)
            stars = 3.0
        if text and text.strip():
            user_review_data.append((text, stars))
    
    if not user_review_data:
        return None
    
    # Calculate embeddings for each user review
    user_embeddings = []
    weights = []
    for review_text, stars in user_review_data:
        try:
            emb = embedding_model.embed_query(review_text)
            user_embeddings.append(emb)
            # Use rating as weight (normalize to 0.5-1.5 range for stability)
            weights.append(0.5 + (stars - 1.0) / 4.0)  # 1.0->0.5, 5.0->1.5
        except Exception:
            continue
    
    if not user_embeddings:
        return None
    
    # Calculate weighted average embedding
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    query_embedding = np.average(user_embeddings, axis=0, weights=weights).tolist()
    return query_embedding


def select_top_k_reviews(business_reviews, user_reviews, llm, top_k=15):
    """
    How it works:
    1. Calculate weighted average embedding of user's reviews (weighted by rating)
    2. Calculate text similarity (cosine similarity) for each business review
    3. Calculate rating similarity (how close business review rating is to user's avg rating)
    4. Combine both scores to select top-k most relevant reviews
    """
    # If reviews are few, return all
    if len(business_reviews) <= top_k:
        return [r.get("text", "") for r in business_reviews if r.get("text", "")]
    
    # If no user reviews, return first top_k
    if not user_reviews:
        return [r.get("text", "") for r in business_reviews[:top_k] if r.get("text", "")]
    
    embedding_model = llm.get_embedding_model()
    if embedding_model is None:
        return [r.get("text", "") for r in business_reviews[:top_k] if r.get("text", "")]
    
    # Calculate user's average rating for rating similarity
    user_ratings = []
    for review in user_reviews[:10]:  # Use first 10 reviews for rating average
        if isinstance(review, dict):
            stars = review.get("stars", 3.0)
        else:
            stars = 3.0
        user_ratings.append(stars)
    user_avg_rating = np.mean(user_ratings) if user_ratings else 3.0
    query_embedding = calculate_average_embedding(user_reviews, llm, top_k=10)
    if query_embedding is None:
        return [r.get("text", "") for r in business_reviews[:top_k] if r.get("text", "")]
    
    # Calculate the combined scores for each business review
    review_scores = []
    for review in business_reviews:
        if isinstance(review, dict):
            review_text = review.get("text", "")
            review_rating = review.get("stars", 3.0)
        else:
            review_text = str(review)
            review_rating = 3.0
        
        if not review_text:
            continue
        
        try:
            # Text similarity (cosine similarity)
            review_embedding = embedding_model.embed_query(review_text)
            text_sim = cosine_similarity(query_embedding, review_embedding)
            
            # Rating similarity (how close to user's average rating)
            # Normalize: 1.0 difference = 0.2 penalty, max difference 4.0 = 0.8 so that there still exist some similarity 
            rating_diff = abs(review_rating - user_avg_rating)
            rating_sim = 1.0 - min(rating_diff / 4.0, 0.8)  # Range: 0.2-1.0
            
            # Combined score: 70% text similarity + 30% rating similarity
            combined_score = 0.7 * text_sim + 0.3 * rating_sim
            
            review_scores.append((review_text, combined_score, text_sim, rating_sim))
        except Exception:
            continue
    
    # Sort by combined score and select top-k
    review_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [text for text, _, _, _ in review_scores[:top_k]]
    
    if not selected:
        return [r.get("text", "") for r in business_reviews[:top_k] if r.get("text", "")]
    
    return selected

