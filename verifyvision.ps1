param(
    [Parameter(Position = 0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manager = Join-Path $repositoryRoot "scripts\manage-local.ps1"
$setupScript = Join-Path $repositoryRoot "scripts\setup-local.ps1"
$updateScript = Join-Path $repositoryRoot "scripts\update-dependencies.ps1"
$versionFile = Join-Path $repositoryRoot "backend\app\VERSION"
$normalizedCommand = $Command.Trim().ToLowerInvariant()

function Show-Help {
    Write-Host "VerifyVision"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\verifyvision.ps1 <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  all        Set up and start VerifyVision"
    Write-Host "  setup      Install pinned dependencies"
    Write-Host "  start      Start the frontend and backend"
    Write-Host "  stop       Stop services started by this launcher"
    Write-Host "  status     Show service and CUDA status"
    Write-Host "  diagnose   Check the environment and CUDA"
    Write-Host "  update     Check for dependency updates"
    Write-Host "  version    Show the VerifyVision version"
    Write-Host "  help       Show this help"
}

function Invoke-Managed([string]$ManagedCommand) {
    if ($env:VERIFYVISION_LAUNCHER_TEST_MODE -eq "1") {
        Write-Output "dispatch:${ManagedCommand}:scripts/manage-local.ps1"
        return
    }
    & $manager $ManagedCommand
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($normalizedCommand) {
    "" { Show-Help }
    "help" { Show-Help }
    "version" {
        $version = (Get-Content -Raw -LiteralPath $versionFile).Trim()
        Write-Host "VerifyVision v$version"
    }
    "setup" {
        if ($env:VERIFYVISION_LAUNCHER_TEST_MODE -eq "1") {
            Write-Output "dispatch:setup:scripts/setup-local.ps1"
            break
        }
        & $setupScript
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "start" { Invoke-Managed "start" }
    "stop" { Invoke-Managed "stop" }
    "status" { Invoke-Managed "status" }
    "diagnose" { Invoke-Managed "diagnose" }
    "update" {
        if ($env:VERIFYVISION_LAUNCHER_TEST_MODE -eq "1") {
            Write-Output "dispatch:update:scripts/update-dependencies.ps1"
            break
        }
        & $updateScript -Action Review
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "all" {
        if ($env:VERIFYVISION_LAUNCHER_TEST_MODE -eq "1") {
            Write-Output "dispatch:all:setup-if-needed,start"
            break
        }
        & $manager ready
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Preparing VerifyVision for first use..."
            & $setupScript
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        & $manager start
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    default {
        [Console]::Error.WriteLine("Unknown command: $Command")
        Show-Help
        exit 2
    }
}
