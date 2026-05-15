<#
.SYNOPSIS
  Run a command with MSVC (cl.exe) and Windows SDK on PATH — required before nvcc can compile
  Gaussian Splatting submodules.

.DESCRIPTION
  Use this when a normal PowerShell / terminal yields:
    nvcc fatal : Cannot find compiler 'cl.exe' in PATH

  Starts cmd.exe with vcvars64.bat from the newest VS install that includes the MSVC toolset.
  Pip and build tools write to stderr — this script temporarily sets $ErrorActionPreference to
  Continue for the cmd.exe child so Tee-Object pipelines do not fail with NativeCommandError.

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
    throw "Missing command (pass args after script name)."
}

$vswhere = Join-Path "${env:ProgramFiles(x86)}" "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere not found under Program Files (x86). Install Visual Studio 2022 or Build Tools with 'Desktop development with C++'."
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
    throw "No Visual Studio instance with MSVC (VC.Tools.x86.x64). Install workload 'Desktop development with C++'."
}

$vcvars64 = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars64)) {
    throw "Missing vcvars64.bat: $vcvars64"
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

    # Pip writes progress to stderr; $ErrorActionPreference = 'Stop' turns that into terminating errors.
    $priorEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & cmd.exe /s /c "`"$wrapper`""
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorEAP
    }
    if ($exitCode -ne 0) { exit $exitCode }
}
finally {
    Remove-Item -LiteralPath $wrapper -ErrorAction SilentlyContinue
}
