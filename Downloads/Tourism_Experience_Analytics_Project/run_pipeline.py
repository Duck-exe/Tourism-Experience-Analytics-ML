"""Run the complete reproducible build: cleaning -> EDA -> predictive models -> recommender."""
from src.data_pipeline import run_pipeline
from src.eda import run_eda
from src.train_models import train_all
from src.recommender import train_recommender

if __name__ == "__main__":
    run_pipeline()
    run_eda()
    train_all()
    train_recommender()
    print("\nAll project artifacts were generated successfully.")
