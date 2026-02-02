"""
Yelp Data Processing Script for AgentSociety Challenge

Purpose:
    This script transforms raw Yelp Open Dataset files into the format required by the
    websocietysimulator framework. It performs two essential operations:

    1. Field Renaming: Converts 'business_id' to 'item_id' to match simulator API
    2. File Naming: Renames files to simulator conventions (item.json, user.json, review.json)

    Optional: Data Filtering: Reduces dataset size by filtering to top 3 cities (Philadelphia, Tampa, Tucson)

Input:
    - yelp_academic_dataset_business.json
    - yelp_academic_dataset_user.json
    - yelp_academic_dataset_review.json

Output:
    - item.json (businesses with 'item_id' field)
    - user.json (users who reviewed filtered businesses)
    - review.json (reviews with 'item_id' field)

Usage:
    python process_yelp_data.py --input_dir data/raw/yelp --output_dir data/processed/yelp
"""

import json
import logging
import pandas as pd
from tqdm import tqdm
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_FILES = [
    'yelp_academic_dataset_business.json',
    'yelp_academic_dataset_user.json',
    'yelp_academic_dataset_review.json'
]

DEFAULT_CITIES = ['Philadelphia', 'Tampa', 'Tucson']


def load_data(file_path):
    """Load JSON Lines data into a Pandas DataFrame with progress bar."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in tqdm(lines, desc=f"Loading {os.path.basename(file_path)}", unit=" lines"):
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping invalid JSON line: {e}")
    return pd.DataFrame(data)


def save_jsonl(dataframe, output_file):
    """Save a Pandas DataFrame to a JSON Lines file."""
    logging.info(f"Saving {output_file}...")
    dataframe.to_json(output_file, orient='records', lines=True)
    logging.info(f"Saved {len(dataframe)} records to {output_file}")


def filter_data(top_cities, business_df, user_df, review_df):
    """Filter data for the specified cities."""
    logging.info(f"Filtering data for cities: {top_cities}")
    
    filtered_businesses = business_df[business_df['city'].isin(top_cities)].copy()
    logging.info(f"  {len(filtered_businesses)} businesses in target cities")
    
    filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])].copy()
    logging.info(f"  {len(filtered_reviews)} reviews for these businesses")
    
    filtered_users = user_df[user_df['user_id'].isin(filtered_reviews['user_id'])].copy()
    logging.info(f"  {len(filtered_users)} users who wrote these reviews")
    
    return filtered_businesses, filtered_reviews, filtered_users


def prepare_for_simulator(filtered_businesses, filtered_reviews, filtered_users, output_dir):
    """Transform data to match simulator's expected format."""
    logging.info("Preparing data for simulator...")
    
    # Rename business_id to item_id for businesses
    items = filtered_businesses.copy()
    items = items.rename(columns={'business_id': 'item_id'})
    items['source'] = 'yelp'
    items['type'] = 'business'
    
    # Rename business_id to item_id for reviews
    reviews = filtered_reviews.copy()
    reviews = reviews.rename(columns={'business_id': 'item_id'})
    if 'review_id' not in reviews.columns:
        logging.warning("review_id not found, using index as review_id")
        reviews['review_id'] = reviews.index.astype(str)
    reviews['source'] = 'yelp'
    
    # Add source field to users
    users = filtered_users.copy()
    users['source'] = 'yelp'
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(items, os.path.join(output_dir, 'item.json'))
    save_jsonl(reviews, os.path.join(output_dir, 'review.json'))
    save_jsonl(users, os.path.join(output_dir, 'user.json'))


def check_required_files(input_dir):
    """Check if all required Yelp files exist."""
    missing_files = []
    for file in REQUIRED_FILES:
        if not os.path.exists(os.path.join(input_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        logging.error("Missing required files:")
        for file in missing_files:
            logging.error(f"   - {file}")
        return False
    
    logging.info("All required files found")
    return True


def main():
    parser = argparse.ArgumentParser(description="Process Yelp dataset for AgentSociety simulator")
    parser.add_argument('--input_dir', required=True, help="Path to raw Yelp data directory")
    parser.add_argument('--output_dir', required=True, help="Path to output directory for processed data")
    parser.add_argument('--cities', nargs='+', default=DEFAULT_CITIES,
                       help=f"Cities to filter (default: {' '.join(DEFAULT_CITIES)})")
    args = parser.parse_args()

    if not check_required_files(args.input_dir):
        return

    logging.info("=" * 60)
    logging.info("LOADING RAW DATA")
    logging.info("=" * 60)
    
    business_file = os.path.join(args.input_dir, 'yelp_academic_dataset_business.json')
    user_file = os.path.join(args.input_dir, 'yelp_academic_dataset_user.json')
    review_file = os.path.join(args.input_dir, 'yelp_academic_dataset_review.json')
    
    business_df = load_data(business_file)
    user_df = load_data(user_file)
    review_df = load_data(review_file)
    
    logging.info(f"\nLoaded: {len(business_df)} businesses, {len(user_df)} users, {len(review_df)} reviews")

    logging.info("\n" + "=" * 60)
    logging.info("FILTERING DATA")
    logging.info("=" * 60)
    
    filtered_businesses, filtered_reviews, filtered_users = filter_data(
        args.cities, business_df, user_df, review_df
    )

    logging.info("\n" + "=" * 60)
    logging.info("SAVING PROCESSED DATA")
    logging.info("=" * 60)
    
    prepare_for_simulator(filtered_businesses, filtered_reviews, filtered_users, args.output_dir)

    logging.info("\n" + "=" * 60)
    logging.info("DATA PROCESSING COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Processed files saved to: {args.output_dir}")
    logging.info(f"\nNext steps:")
    logging.info(f"  1. Verify files: ls -lh {args.output_dir}/")
    logging.info(f"  2. Update config to use data_dir='{args.output_dir}'")
    logging.info(f"  3. Run the baseline agent")


if __name__ == '__main__':
    main()

