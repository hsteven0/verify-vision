$ErrorActionPreference = "Stop"
$launcherRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $launcherRoot)
$source = Join-Path $launcherRoot "Launcher.cs"
$icon = Join-Path $launcherRoot "VerifyVision.ico"
$output = Join-Path $repositoryRoot "VerifyVision.exe"
$version = (Get-Content -LiteralPath (Join-Path $repositoryRoot "backend\app\VERSION") -Raw).Trim()
$assemblyInfo = Join-Path ([IO.Path]::GetTempPath()) "VerifyVision-$([guid]::NewGuid()).cs"
$compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"

if ($version -notmatch '^\d+\.\d+\.\d+$') {
    throw "Invalid VerifyVision version: $version"
}
if (-not (Test-Path -LiteralPath $compiler)) {
    $compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
}
if (-not (Test-Path -LiteralPath $compiler)) {
    throw "The Windows C# compiler was not found."
}

@"
using System.Reflection;
[assembly: AssemblyTitle("VerifyVision")]
[assembly: AssemblyDescription("VerifyVision")]
[assembly: AssemblyProduct("VerifyVision")]
[assembly: AssemblyVersion("$version.0")]
[assembly: AssemblyFileVersion("$version.0")]
[assembly: AssemblyInformationalVersion("$version")]
"@ | Set-Content -LiteralPath $assemblyInfo -Encoding UTF8

Remove-Item -LiteralPath $output -Force -ErrorAction SilentlyContinue
try {
    & $compiler /nologo /target:winexe "/out:$output" "/win32icon:$icon" `
        /reference:System.dll /reference:System.Windows.Forms.dll $source $assemblyInfo
    if ($LASTEXITCODE -ne 0) {
        throw "Launcher build failed."
    }
}
finally {
    Remove-Item -LiteralPath $assemblyInfo -Force -ErrorAction SilentlyContinue
}

Write-Host "Built $output"
