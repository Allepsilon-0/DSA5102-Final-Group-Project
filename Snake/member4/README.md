# Member 4 — final PPO models and results

Open Member4_PPO_Final.ipynb for the English report, complete implementation and executed outputs. No retraining is needed to read it. Start Jupyter in this directory when executing cells so the relative data paths resolve.

Install requirements.txt to execute the code. Use `python run_model.py evaluate --episodes 200 --seed 2950000` to replay the representative model's test games, or `python run_model.py demo` to generate a gameplay GIF. The default model is the validation-selected seed-7, clip-0.2 checkpoint, not the best test model. Use --checkpoint to select any other included checkpoint.

results/final contains the final protocol, environment versions, aggregate scores, all 15 selected best.pt checkpoints, their validation histories and individual test outcomes. Figures are the full validation curves and seed-level comparison. Original environment rules, rewards, observations and actions are retained. This package excludes development searches, old baseline models and intermediate training-stage weights.

Each final model was trained for 500,000 interactions. The formal experiment totals 7,500,000 training interactions and 3,000 test games. Checkpoints are chosen by validation score, then success and reward, not by test performance. All three clipping groups have identical budgets and settings except the clipping value.
