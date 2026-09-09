param(
    [ValidateSet("Review", "Apply")]
    [string]$Action = "Review",
    [string[]]$PythonPackage = @(),
    [string[]]$FrontendPackage = @(),
    [switch]$IncludeCriticalRuntime,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$frontend = Join-Path $repositoryRoot "frontend"
$criticalRuntime = @(
    "torch", "torchvision", "transformers", "ultralytics", "peft", "numpy",
    "opencv-python-headless", "lmdb"
)
$localNpm = Get-ChildItem -LiteralPath (Join-Path $repositoryRoot ".tools\NodeJS") `
    -Filter npm.cmd -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($localNpm) {
    $npmCommand = $localNpm.FullName
    $env:Path = "$($localNpm.DirectoryName);$env:Path"
}
else {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { throw "Node.js/npm is required." }
    $npmCommand = $npm.Source
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing managed .venv. Run .\verifyvision.ps1 setup first."
}

& $python (Join-Path $PSScriptRoot "check_dependency_updates.py")
if ($LASTEXITCODE -ne 0) { throw "Dependency update discovery failed." }
if ($Action -eq "Review") {
    exit 0
}

if ($PythonPackage.Count -eq 0 -and $FrontendPackage.Count -eq 0) {
    throw "Apply requires at least one explicit -PythonPackage or -FrontendPackage value."
}
foreach ($package in @($PythonPackage + $FrontendPackage)) {
    if ($package -notmatch '^(?:@[-a-zA-Z0-9_.]+/)?[-a-zA-Z0-9_.]+$') {
        throw "Invalid package name: $package"
    }
}
$requestedCritical = @($PythonPackage | Where-Object { $criticalRuntime -contains $_.ToLowerInvariant() })
if ($requestedCritical.Count -gt 0 -and -not $IncludeCriticalRuntime) {
    throw "Critical ML/runtime updates require -IncludeCriticalRuntime: $($requestedCritical -join ', ')"
}

Write-Host "Requested Python packages: $($PythonPackage -join ', ')"
Write-Host "Requested frontend packages: $($FrontendPackage -join ', ')"
$confirmation = Read-Host "Type UPDATE to back up manifests, apply these changes, and run health checks"
if ($confirmation -cne "UPDATE") {
    throw "Dependency update cancelled; nothing was changed."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupRoot = Join-Path $repositoryRoot ".dependency-backups\$timestamp"
$backupDirectories = @(
    (Join-Path $backupRoot "backend")
    (Join-Path $backupRoot "frontend")
    (Join-Path $backupRoot "scripts")
)
New-Item -ItemType Directory -Force -Path $backupDirectories | Out-Null
Copy-Item -LiteralPath (Join-Path $repositoryRoot "backend\pyproject.toml") -Destination (Join-Path $backupRoot "backend\pyproject.toml")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "backend\requirements-local.lock") `
    -Destination (Join-Path $backupRoot "backend\requirements-local.lock")
Copy-Item -LiteralPath (Join-Path $frontend "package.json") -Destination (Join-Path $backupRoot "frontend\package.json")
Copy-Item -LiteralPath (Join-Path $frontend "package-lock.json") -Destination (Join-Path $backupRoot "frontend\package-lock.json")
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "setup-local.ps1") -Destination (Join-Path $backupRoot "scripts\setup-local.ps1")
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "setup-local.sh") -Destination (Join-Path $backupRoot "scripts\setup-local.sh")

try {
    if ($PythonPackage.Count -gt 0) {
        $pipArguments = @("-m", "pip", "install", "--upgrade") + $PythonPackage
        if ($requestedCritical -contains "torch" -or $requestedCritical -contains "torchvision") {
            $pipArguments += @("--extra-index-url", $TorchIndexUrl)
        }
        & $python @pipArguments
        if ($LASTEXITCODE -ne 0) { throw "Python dependency update failed." }

        foreach ($package in $PythonPackage) {
            $distribution = $package.ToLowerInvariant()
            $installed = & $python -c "import importlib.metadata as m,sys; print(m.version(sys.argv[1]).split('+')[0])" $distribution
            if ($LASTEXITCODE -ne 0 -or -not $installed) {
                throw "Could not resolve the installed version of $package."
            }
            foreach ($relativePath in @(
                "backend\pyproject.toml", "backend\requirements-local.lock",
                "scripts\setup-local.ps1", "scripts\setup-local.sh"
            )) {
                $path = Join-Path $repositoryRoot $relativePath
                $content = Get-Content -Raw -LiteralPath $path
                $escaped = [regex]::Escape($distribution)
                $content = [regex]::Replace(
                    $content,
                    "(?i)(?<=$escaped==)[0-9][0-9A-Za-z.+-]*",
                    $installed
                )
                Set-Content -LiteralPath $path -Value $content -Encoding utf8NoBOM
            }
        }
        & $python -m pip install --no-deps -e (Join-Path $repositoryRoot "backend")
        if ($LASTEXITCODE -ne 0) { throw "Backend manifest refresh failed." }
    }

    if ($FrontendPackage.Count -gt 0) {
        $npmPackages = @($FrontendPackage | ForEach-Object { "$_@latest" })
        & $npmCommand --prefix $frontend install --save-exact @npmPackages
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency update failed." }
    }

    & $python (Join-Path $PSScriptRoot "diagnose_locateanything.py")
    if ($LASTEXITCODE -ne 0) { throw "CUDA/LocateAnything diagnostic failed after update." }
    & $python -m pytest (Join-Path $repositoryRoot "backend\tests") -q
    if ($LASTEXITCODE -ne 0) { throw "Backend tests failed after update." }
    & $npmCommand --prefix $frontend run test
    if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed after update." }
    & $npmCommand --prefix $frontend run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed after update." }
}
catch {
    Write-Error "Update failed. Restore files from $backupRoot, then run .\verifyvision.ps1 setup."
    throw
}

Write-Host "Update complete. Checks passed. Backup: $backupRoot"
