# run_project.ps1  –  Car Insurance GNN Fraud Detection
# Run from project root with conda env active:
#   conda activate paysim_gnn
#   .\run_project.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Car Insurance GNN Fraud Detection – Runner" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$csv = "data\car_insurance_fraud_dataset.csv"
if (-Not (Test-Path $csv)) {
    Write-Host "[ERROR] Dataset not found: $csv" -ForegroundColor Red; exit 1
}
Write-Host "[OK] Dataset found." -ForegroundColor Green

Write-Host "`n>>> Step 1/3: Building graph ..." -ForegroundColor Yellow
python graph_construction.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED." -ForegroundColor Red; exit 1 }
Write-Host "[OK] Graph saved." -ForegroundColor Green

Write-Host "`n>>> Step 2/3: XGBoost baseline ..." -ForegroundColor Yellow
python baseline_models.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED." -ForegroundColor Red; exit 1 }
Write-Host "[OK] XGBoost done." -ForegroundColor Green

Write-Host "`n>>> Step 3/3: Training GNN ..." -ForegroundColor Yellow
python train_gnn.py --epochs 30 --batch_size 512 --hidden_dim 128 --lr 0.0003 --num_neighbors 20 15
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED." -ForegroundColor Red; exit 1 }
Write-Host "[OK] GNN done." -ForegroundColor Green

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host " All steps complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Outputs:"
Write-Host "  data\insurance_graph.pt          - PyG graph"
Write-Host "  data\best_model.pt               - Best GNN checkpoint"
Write-Host "  data\xgb_model.json              - XGBoost model"
Write-Host "  data\xgb_feature_importance.png  - Feature importance plot"
