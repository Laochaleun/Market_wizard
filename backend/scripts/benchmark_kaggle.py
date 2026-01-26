
import polars as pl
import asyncio
import os
import sys
from pathlib import Path

# Add backend to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.ssr_engine import SSREngine, SSRResult
from app.i18n import Language
import numpy as np

async def main():
    # Path to dataset
    dataset_path = "/Users/pawel/Documents/!Praca/Subverse/Human purchase intent/Womens Clothing E-Commerce Reviews.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please download it from https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews")
        print("and place it in the current directory.")
        return

    print("Loading dataset...")
    df = pl.read_csv(dataset_path)
    
    # Filter for valid reviews
    df = df.filter(pl.col("Review Text").is_not_null())
    
    # Take a sample to avoid extremely long runtimes during testing
    sample_size = 100
    print(f"Processing sample of {sample_size} reviews...")
    df = df.sample(n=sample_size, seed=42)
    
    # Temperatures to test
    temperatures = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    print(f"Running optimization on sample of {sample_size} reviews...")
    
    best_temp = None
    best_mae = float('inf')
    
    print(f"{'Temp':<6} | {'MAE':<8} | {'RMSE':<8} | {'Corr':<8} | {'Mean Pred':<10}")
    print("-" * 55)
    
    # Cache embeddings to avoid re-embedding for every temperature
    # We'll embed once using the first engine, then reuse
    text_cache = []
    embedding_cache = []
    
    # Pre-fetch data and embeddings
    print("Pre-computing embeddings...")
    temp_engine = SSREngine(language=Language.EN) # Helper for embedding
    
    filtered_rows = []
    for row in df.iter_rows(named=True):
        filtered_rows.append(row)
        
    # Batch processing could be better but simple loop is fine for 100 items
    for row in filtered_rows:
        text = row["Review Text"]
        text_cache.append(text)
        # Directly use client to get embedding
        emb = temp_engine.embedding_client.embed([text])[0]
        embedding_cache.append(emb)

    for temp in temperatures:
        # Initialize engine with current temperature
        engine = SSREngine(language=Language.EN, temperature=temp)
        
        results = []
        
        for i, row in enumerate(filtered_rows):
            actual_rating = row["Rating"]
            emb = embedding_cache[i]
            
            # Manually trigger the PMF calculation part of rate_response to skip re-embedding
            # This requires accessing private methods which is hacky but fast for benchmark
            # Or we just use the public API method but patch the embedding client? 
            # Let's just run it standard way, 100 items is fast enough to repeat.
            # Re-running standard rate_response for simplicity and correctness.
            
            try:
                ssr_result = engine.rate_response(text_cache[i])
                
                results.append({
                    "actual_rating": actual_rating,
                    "predicted_score": ssr_result.expected_score,
                    "diff": ssr_result.expected_score - actual_rating,
                })
            except Exception as e:
                pass
                
        # Metrics
        results_df = pl.DataFrame(results)
        diffs = results_df["diff"].to_numpy()
        actuals = results_df["actual_rating"].to_numpy()
        predicteds = results_df["predicted_score"].to_numpy()
        
        mae = np.mean(np.abs(diffs))
        rmse = np.sqrt(np.mean(diffs**2))
        correlation = np.corrcoef(actuals, predicteds)[0, 1]
        mean_pred = np.mean(predicteds)
        
        print(f"{temp:<6.2f} | {mae:<8.4f} | {rmse:<8.4f} | {correlation:<8.4f} | {mean_pred:<10.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_temp = temp

    print("-" * 55)
    print(f"Best Temperature (by MAE): {best_temp}")

if __name__ == "__main__":
    asyncio.run(main())
