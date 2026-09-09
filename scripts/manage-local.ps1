param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("ready", "start", "stop", "status", "diagnose")]
    [string]$Command
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$backendPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$frontendRoot = Join-Path $repositoryRoot "frontend"
$viteScript = Join-Path $frontendRoot "node_modules\vite\bin\vite.js"
$workerPath = Join-Path $repositoryRoot ".vendor\Eagle\Embodied\locateanything_worker.py"
$runtimeRoot = Join-Path $repositoryRoot ".tmp\verifyvision-runtime"
$statePath = Join-Path $runtimeRoot "state.json"
$backendPort = 8000
$frontendPort = 5173

function Set-LocalEnvironment {
    $cacheRoot = Join-Path $repositoryRoot ".cache"
    if (-not $env:npm_config_cache) { $env:npm_config_cache = Join-Path $cacheRoot "npm" }
    if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $repositoryRoot "model-cache\huggingface" }
    if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $repositoryRoot "model-cache\torch" }
    if (-not $env:XDG_CACHE_HOME) { $env:XDG_CACHE_HOME = $cacheRoot }
    if (-not $env:VERIFYVISION_DATA_DIR) {
        $env:VERIFYVISION_DATA_DIR = Join-Path $repositoryRoot "backend\data\projects"
    }
    if (-not $env:VERIFYVISION_LOCATEANYTHING_MODEL) {
        $env:VERIFYVISION_LOCATEANYTHING_MODEL = "nvidia/LocateAnything-3B"
    }
    if (-not $env:VERIFYVISION_LOCATEANYTHING_DEVICE) {
        $env:VERIFYVISION_LOCATEANYTHING_DEVICE = "auto"
    }
    if (-not $env:VERIFYVISION_LOCATEANYTHING_DTYPE) {
        $env:VERIFYVISION_LOCATEANYTHING_DTYPE = "auto"
    }
    if (-not $env:VERIFYVISION_LOCATEANYTHING_WORKER_PATH) {
        $env:VERIFYVISION_LOCATEANYTHING_WORKER_PATH = $workerPath
    }
    $env:TEMP = Join-Path $repositoryRoot ".tmp"
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Force -Path @(
        $env:npm_config_cache,
        $env:HF_HOME,
        $env:TORCH_HOME,
        $env:XDG_CACHE_HOME,
        $env:VERIFYVISION_DATA_DIR,
        $env:TEMP,
        $runtimeRoot
    ) | Out-Null
}

function Get-NodeExecutable {
    $localNode = Get-ChildItem -LiteralPath (Join-Path $repositoryRoot ".tools\NodeJS") `
        -Filter node.exe -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($localNode) { return $localNode.FullName }
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($node) { return $node.Source }
    return $null
}

function Test-SetupReady([switch]$CheckPackages) {
    if (-not (Test-Path -LiteralPath $backendPython -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $viteScript -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $workerPath -PathType Leaf)) { return $false }
    if (-not (Get-NodeExecutable)) { return $false }
    if ($CheckPackages) {
        & $backendPython -c "import fastapi, torch, transformers, ultralytics, uvicorn" *> $null
        return $LASTEXITCODE -eq 0
    }
    return $true
}

function Test-LocalPortInUse([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync("127.0.0.1", $Port)
        if (-not $connection.Wait(100)) { return $false }
        return $client.Connected
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Get-RuntimeState {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) { return $null }
    try { return Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json }
    catch { return $null }
}

function Test-TrackedProcess($Descriptor) {
    if (-not $Descriptor) { return $false }
    $process = Get-Process -Id ([int]$Descriptor.pid) -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    try {
        $actualStartTicks = $process.StartTime.ToUniversalTime().Ticks
        $actualPath = [IO.Path]::GetFullPath($process.Path)
        $expectedPath = [IO.Path]::GetFullPath([string]$Descriptor.executable)
        return $actualStartTicks -eq [int64]$Descriptor.start_time_utc_ticks -and
            $actualPath.Equals($expectedPath, [StringComparison]::OrdinalIgnoreCase)
    }
    catch { return $false }
}

function Remove-StaleState {
    $state = Get-RuntimeState
    if (-not $state) {
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
        return
    }
    if (-not (Test-TrackedProcess $state.backend) -and -not (Test-TrackedProcess $state.frontend)) {
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    }
}

function Stop-TrackedService($Descriptor) {
    if (-not (Test-TrackedProcess $Descriptor)) { return $false }
    Stop-Process -Id ([int]$Descriptor.pid) -Force
    try { Wait-Process -Id ([int]$Descriptor.pid) -Timeout 10 -ErrorAction SilentlyContinue }
    catch { }
    return $true
}

function Wait-ForBackend([System.Diagnostics.Process]$Process) {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($Process.HasExited) { throw "Backend exited during startup. Run .\verifyvision.ps1 diagnose for details." }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$backendPort/api/health" -TimeoutSec 1
            if ($health.status -eq "ok") { return $health }
        }
        catch { }
        Start-Sleep -Milliseconds 250
    }
    throw "Backend did not become healthy. Run .\verifyvision.ps1 diagnose for details."
}

function Wait-ForFrontend([System.Diagnostics.Process]$Process) {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($Process.HasExited) { throw "Frontend exited during startup. Run .\verifyvision.ps1 diagnose for details." }
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$frontendPort" -TimeoutSec 1 -UseBasicParsing
            if ($response.StatusCode -eq 200) { return }
        }
        catch { }
        Start-Sleep -Milliseconds 250
    }
    throw "Frontend did not become ready. Run .\verifyvision.ps1 diagnose for details."
}

function Start-VerifyVision {
    if (-not (Test-SetupReady)) {
        throw "VerifyVision is not set up yet. Run .\verifyvision.ps1 setup or .\verifyvision.ps1 all."
    }

    Remove-StaleState
    $state = Get-RuntimeState
    if ($state -and (Test-TrackedProcess $state.backend) -and (Test-TrackedProcess $state.frontend)) {
        Write-Host "VerifyVision is already running."
        Write-Host "Frontend: http://localhost:$frontendPort"
        Write-Host "Backend:  http://localhost:$backendPort"
        return
    }
    if ($state) {
        Stop-TrackedService $state.frontend | Out-Null
        Stop-TrackedService $state.backend | Out-Null
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    }
    if (Test-LocalPortInUse $backendPort) {
        throw "Port $backendPort is already in use by a process not managed by VerifyVision."
    }
    if (Test-LocalPortInUse $frontendPort) {
        throw "Port $frontendPort is already in use by a process not managed by VerifyVision."
    }
    $node = Get-NodeExecutable

    $backend = $null
    $frontend = $null
    try {
        $backendArguments = "-m uvicorn app.main:app --app-dir `"$($repositoryRoot)\backend`" --host 127.0.0.1 --port $backendPort"
        $backend = Start-Process -FilePath $backendPython -ArgumentList $backendArguments `
            -WorkingDirectory $repositoryRoot -WindowStyle Hidden -PassThru

        $frontendArguments = "`"$viteScript`" --host 127.0.0.1 --port $frontendPort --strictPort"
        $frontend = Start-Process -FilePath $node -ArgumentList $frontendArguments `
            -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru

        $health = Wait-ForBackend $backend
        if ($health.inference_provider -ne "locateanything" -or -not $health.inference_available) {
            throw "LocateAnything CUDA runtime unavailable: $($health.inference_detail)"
        }
        Wait-ForFrontend $frontend

        $backend.Refresh()
        $frontend.Refresh()
        $runtimeState = [ordered]@{
            schema_version = 1
            repository_root = $repositoryRoot
            started_at = (Get-Date).ToUniversalTime().ToString("o")
            backend = [ordered]@{
                pid = $backend.Id
                executable = $backend.Path
                start_time_utc_ticks = $backend.StartTime.ToUniversalTime().Ticks
            }
            frontend = [ordered]@{
                pid = $frontend.Id
                executable = $frontend.Path
                start_time_utc_ticks = $frontend.StartTime.ToUniversalTime().Ticks
            }
        }
        $runtimeState | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding utf8

        $backend.Dispose()
        $frontend.Dispose()
        $backend = $null
        $frontend = $null

        Write-Host ""
        Write-Host "VerifyVision"
        Write-Host ""
        Write-Host "Frontend: http://localhost:$frontendPort"
        Write-Host "Backend:  http://localhost:$backendPort"
        Write-Host ""
        Write-Host "LocateAnything-3B"
        Write-Host "CUDA: $($health.inference_gpu) · $($health.inference_selected_dtype)"
    }
    catch {
        if ($frontend -and -not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force }
        if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
        throw
    }
}

function Stop-VerifyVision {
    Remove-StaleState
    $state = Get-RuntimeState
    if (-not $state) {
        Write-Host "VerifyVision is already stopped."
        return
    }
    $frontendStopped = Stop-TrackedService $state.frontend
    $backendStopped = Stop-TrackedService $state.backend
    Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
    if (-not $frontendStopped -and -not $backendStopped) {
        Write-Host "Removed stale launcher state."
        return
    }
    Write-Host "VerifyVision stopped."
}

function Show-Status {
    Remove-StaleState
    $state = Get-RuntimeState
    $frontendTracked = $state -and (Test-TrackedProcess $state.frontend)
    $backendTracked = $state -and (Test-TrackedProcess $state.backend)
    $frontendStatus = if ($frontendTracked) { "Running" } elseif (Test-LocalPortInUse $frontendPort) { "Occupied" } else { "Stopped" }
    $backendStatus = if ($backendTracked) { "Running" } elseif (Test-LocalPortInUse $backendPort) { "Occupied" } else { "Stopped" }

    $health = $null
    if ($backendTracked) {
        try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$backendPort/api/health" -TimeoutSec 2 }
        catch { }
    }
    $gpu = $null
    $cuda = "Unavailable"
    if ($health -and $health.inference_available) {
        $cuda = "Available"
        $gpu = $health.inference_gpu
    }
    else {
        $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            $gpu = (& $nvidiaSmi.Source --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
            if ($gpu) { $cuda = "Available" }
        }
    }

    Write-Host "VerifyVision Status"
    Write-Host ""
    Write-Host ("Frontend   {0,-10} http://localhost:{1}" -f $frontendStatus, $frontendPort)
    Write-Host ("Backend    {0,-10} http://localhost:{1}" -f $backendStatus, $backendPort)
    Write-Host ("CUDA       {0}" -f $cuda)
    if ($gpu) { Write-Host ("GPU        {0}" -f $gpu.Trim()) }
    Write-Host "Model      LocateAnything-3B"
}

Set-LocalEnvironment
switch ($Command) {
    "ready" { if (Test-SetupReady -CheckPackages) { exit 0 } else { exit 2 } }
    "start" { Start-VerifyVision }
    "stop" { Stop-VerifyVision }
    "status" { Show-Status }
    "diagnose" {
        if (-not (Test-SetupReady)) {
            throw "VerifyVision is not set up yet. Run .\verifyvision.ps1 setup."
        }
        & $backendPython (Join-Path $PSScriptRoot "diagnose_locateanything.py")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host ""
        Show-Status
    }
}
