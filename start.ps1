# Real-Time Fraud Detection Engine — Startup Script
Write-Host "Starting Fraud Detection Engine..." -ForegroundColor Green

# Step 1 — Start Docker infrastructure
Write-Host "`n[1/5] Starting Docker containers..." -ForegroundColor Cyan
docker compose up -d
Start-Sleep -Seconds 5

# Step 2 — Create Kafka topics
Write-Host "`n[2/5] Creating Kafka topics..." -ForegroundColor Cyan
docker exec fraud-engine-redpanda-1 rpk topic create transactions --brokers localhost:9092 --partitions 6 --replicas 1 2>$null
docker exec fraud-engine-redpanda-1 rpk topic create fraud-alerts --brokers localhost:9092 --partitions 2 --replicas 1 2>$null
Write-Host "Topics ready." -ForegroundColor Green

# Step 3 — Start inference server
Write-Host "`n[3/5] Starting ML inference server on :8888..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\projects\fraud-engine; python inference_server.py"
Start-Sleep -Seconds 3

# Step 4 — Start detection service
Write-Host "`n[4/5] Starting detection service on :2112..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\projects\fraud-engine; go run detection\main.go"
Start-Sleep -Seconds 3

# Step 5 — Start API server
Write-Host "`n[5/5] Starting REST API on :8090..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\projects\fraud-engine; go run api\main.go"
Start-Sleep -Seconds 2

Write-Host "`n✅ All services started!" -ForegroundColor Green
Write-Host "`nAvailable endpoints:" -ForegroundColor Yellow
Write-Host "  Redpanda Console:  http://localhost:8080" -ForegroundColor White
Write-Host "  Grafana Dashboard: http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  Fraud API:         http://localhost:8090/evaluate" -ForegroundColor White
Write-Host "  Metrics:           http://localhost:2112/metrics" -ForegroundColor White
Write-Host "`nTo start the producer run:" -ForegroundColor Yellow
Write-Host "  go run producer\main.go" -ForegroundColor White
Write-Host "`nTo run load test:" -ForegroundColor Yellow
Write-Host "  python loadtest.py" -ForegroundColor White