<#
.SYNOPSIS
  Run a command with MSVC (cl.exe) and Windows SDK on PATH — required before nvcc can compile
  Gaussian Splatting submodules.

.DESCRIPTION
  Use this when a normal PowerShell / terminal yields:
    nvcc fatal : Cannot find compiler 'cl.exe' in PATH

  Starts cmd.exe with vcvars64.bat from the newest VS install that includes the MSVC toolset.

.EXAMPLE
  .\scripts\invoke_vs_build_tools.ps1 `
      "C:\Users\me\polygraph_worker\.venv\Scripts\python.exe" `
      -m pip install --no-build-isolation `
      "C:\polyGraphics\third_party\gaussian-splatting\submodules\diff-gaussian-rasterization"
#>

param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Argv
)

if ($Argv.Length -eq 0) {
    Write-Error "Missing command (pass args after script name)."
}

$ErrorActionPreference = "Stop"

$vswhere = Join-Path "${env:ProgramFiles(x86)}" "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    Write-Error "vswhere not found under Program Files (x86). Install Visual Studio 2022 or Build Tools with 'Desktop development with C++'."
}

$installationPath = (
    & $vswhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationpath 2>$null |
        Select-Object -First 1
).Trim()

if (-not $installationPath) {
    Write-Error "No Visual Studio instance with MSVC (VC.Tools.x86.x64). Install workload 'Desktop development with C++'."
}

$vcvars64 = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars64)) {
    Write-Error "Missing vcvars64.bat: $vcvars64"
}

function QuoteCmdArg([string]$s) {
    if ([string]::IsNullOrWhiteSpace($s)) { return '""' }
    if ($s -match '[\s\^"&|<>()%]') { '"' + $s.Replace('"', '""') + '"' }
    elseif ($s -match '^-') { $s }
    else { '"' + $s + '"' }
}

$line = ($Argv | ForEach-Object { QuoteCmdArg $_ }) -join ' '
$quotedVcvars = '"' + ($vcvars64.Replace('"', '""')) + '"'

$wrapper = Join-Path ([System.IO.Path]::GetTempPath()) ("polygraph-vsgs-" + [Guid]::NewGuid().ToString() + ".cmd")
try {
    @(
        "@echo off"
        "call $quotedVcvars"
        "if errorlevel 1 exit /b 1"
        $line
        "exit /b %ERRORLEVEL%"
    ) | Set-Content -Path $wrapper -Encoding OEM

    & cmd.exe /s /c "`"$wrapper`""
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Remove-Item -LiteralPath $wrapper -ErrorAction SilentlyContinue
}
