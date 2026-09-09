param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$environmentPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$localPython312 = Join-Path $repositoryRoot ".tools\Python312\python.exe"
$localNodeRoot = Join-Path $repositoryRoot ".tools\NodeJS"
$eagleRoot = Join-Path $repositoryRoot ".vendor\Eagle"
$eagleRevision = "783f656d127ee498137b5ff52603ce36c292d317"
$cacheRoot = Join-Path $repositoryRoot ".cache"
$temporaryRoot = Join-Path $repositoryRoot ".tmp"

if (-not $env:PIP_CACHE_DIR) { $env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip" }
if (-not $env:npm_config_cache) { $env:npm_config_cache = Join-Path $cacheRoot "npm" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $repositoryRoot "model-cache\huggingface" }
if (-not $env:TORCH_HOME) { $env:TORCH_HOME = Join-Path $repositoryRoot "model-cache\torch" }
if (-not $env:XDG_CACHE_HOME) { $env:XDG_CACHE_HOME = $cacheRoot }
if (-not $env:VERIFYVISION_DATA_DIR) {
    $env:VERIFYVISION_DATA_DIR = Join-Path $repositoryRoot "backend\data\projects"
}
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
New-Item -ItemType Directory -Force -Path @(
    $env:PIP_CACHE_DIR, $env:npm_config_cache, $env:HF_HOME, $env:TORCH_HOME,
    $env:XDG_CACHE_HOME, $env:VERIFYVISION_DATA_DIR, $temporaryRoot
) | Out-Null

if (Test-Path -LiteralPath $localPython312) {
    $python312 = $localPython312
}
else {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pythonLauncher) {
        throw "Python 3.12 (64-bit) is required. Install it or place it under .tools\Python312."
    }
    $python312 = & $pythonLauncher.Source -3.12 -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0 -or -not $python312) {
        throw "Python 3.12 (64-bit) is required."
    }
}

$localNpm = Get-ChildItem -LiteralPath $localNodeRoot -Filter npm.cmd -File -Recurse `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if ($localNpm) {
    $npmCommand = $localNpm.FullName
    $env:Path = "$($localNpm.DirectoryName);$env:Path"
}
else {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { throw "Node.js 20+ and npm are required." }
    $npmCommand = $npm.Source
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    Write-Host "Creating Python 3.12 environment on $([IO.Path]::GetPathRoot($repositoryRoot))..."
    & $python312 -m venv (Join-Path $repositoryRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python could not create .venv." }
}
$environmentVersion = & $environmentPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($environmentVersion -ne "3.12") {
    throw ".venv uses Python $environmentVersion; recreate it with Python 3.12."
}

Write-Host "Installing pinned CUDA PyTorch..."
& $environmentPython -m pip install torch==2.12.1 torchvision==0.27.1 --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "The CUDA-enabled PyTorch wheels could not be installed." }
& $environmentPython -m pip install -r (Join-Path $repositoryRoot "backend\requirements-local.lock")
if ($LASTEXITCODE -ne 0) { throw "Pinned backend dependencies could not be installed." }
& $environmentPython -m pip install --no-deps -e (Join-Path $repositoryRoot "backend")
if ($LASTEXITCODE -ne 0) { throw "VerifyVision backend could not be registered in .venv." }

Write-Host "Installing frontend packages..."
& $npmCommand --prefix (Join-Path $repositoryRoot "frontend") ci
if ($LASTEXITCODE -ne 0) { throw "Locked frontend dependencies could not be installed." }

if (-not (Test-Path -LiteralPath (Join-Path $eagleRoot ".git"))) {
    New-Item -ItemType Directory -Path $eagleRoot -Force | Out-Null
    & git -C $eagleRoot init
    & git -C $eagleRoot remote add origin https://github.com/NVlabs/Eagle.git
}
Write-Host "Fetching the pinned NVIDIA Eagle worker..."
& git -C $eagleRoot fetch --depth 1 origin $eagleRevision
if ($LASTEXITCODE -ne 0) { throw "The pinned NVIDIA Eagle revision could not be downloaded." }
& git -C $eagleRoot checkout --detach $eagleRevision
if ($LASTEXITCODE -ne 0) { throw "The NVIDIA Eagle worker could not be checked out." }
$env:VERIFYVISION_LOCATEANYTHING_WORKER_PATH = Join-Path $eagleRoot "Embodied\locateanything_worker.py"

& $environmentPython (Join-Path $PSScriptRoot "diagnose_locateanything.py")
if ($LASTEXITCODE -ne 0) {
    throw "Setup completed, but the NVIDIA CUDA runtime is not usable. Review the diagnostic output."
}
Write-Host "VerifyVision is ready. Start it with .\verifyvision.ps1 start"
