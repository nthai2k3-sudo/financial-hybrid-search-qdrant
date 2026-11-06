# Re-run

Run from the repository root with the project virtual environment:

```powershell
$env:PYTHONPATH="experiments\dim_384_vs_768\scripts;src"
python experiments\dim_384_vs_768\scripts\snapshot_384.py
python experiments\dim_384_vs_768\scripts\build_768_collection.py
python experiments\dim_384_vs_768\scripts\verify_integrity.py
python experiments\dim_384_vs_768\scripts\run_eval.py
python experiments\dim_384_vs_768\scripts\stats_compare.py
python experiments\dim_384_vs_768\scripts\latency_compare.py
python experiments\dim_384_vs_768\scripts\make_report.py
```

The scripts refuse to overwrite an existing 768 collection unless their own
checkpoint exists. They do not recreate or modify `financial_docs`.
