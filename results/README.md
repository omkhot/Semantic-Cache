# results/

Evaluation output lands here (threshold_sweep.csv, summary.json, comparison_*.md,
workload.json/.md, PNG plots). Generate it with:

    python -m evaluation.run_eval --dataset qqp --embedder minilm --verifier rules
    python -m evaluation.simulate --embedder minilm --threshold 0.85

Only put numbers on your resume that appear in these files after YOU ran them.
