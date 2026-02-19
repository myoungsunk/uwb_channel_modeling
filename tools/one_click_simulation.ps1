# UWB Channel Modeling - one-click simulation/validation runner (PowerShell)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\tools\one_click_simulation.ps1
# Optional:
#   -OutReport artifacts/updated_report.md
#   -OutH5 artifacts/rt_sweep.h5
#   -SkipTests

param(
    [string]$OutReport = "artifacts/updated_report.md",
    [string]$OutH5 = "artifacts/rt_sweep.h5",
    [switch]$SkipTests,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

Write-Host "[0/9] Repo check" -ForegroundColor Cyan
$repo = (Get-Location).Path
Write-Host "Repo: $repo"

function Get-PythonCandidates {
    $candidates = New-Object System.Collections.ArrayList

    function Add-Candidate {
        param([object[]]$cand)
        if ($null -eq $cand -or $cand.Count -eq 0) { return }
        $key = ($cand -join " ").ToLowerInvariant()
        if (-not $script:SeenCandidates.ContainsKey($key)) {
            $script:SeenCandidates[$key] = $true
            [void]$candidates.Add($cand)
        }
    }

    $script:SeenCandidates = @{}

    # 0) Explicit user override
    if ($PythonExe -and (Test-Path $PythonExe)) {
        Add-Candidate @($PythonExe)
    }

    # 1) Launcher commands from PATH
    if (Get-Command py -ErrorAction SilentlyContinue) { Add-Candidate @("py", "-3") }
    if (Get-Command python -ErrorAction SilentlyContinue) { Add-Candidate @("python") }
    if (Get-Command python3 -ErrorAction SilentlyContinue) { Add-Candidate @("python3") }

    # 2) Well-known absolute locations on Windows
    $winPy = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
        "$env:ProgramFiles\Python313\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "$env:ProgramFiles\Python310\python.exe",
        "$env:ProgramFiles\Python39\python.exe",
        "$env:ProgramFiles(x86)\Python313\python.exe",
        "$env:ProgramFiles(x86)\Python312\python.exe",
        "$env:ProgramFiles(x86)\Python311\python.exe",
        "$env:ProgramFiles(x86)\Python310\python.exe",
        "$env:ProgramFiles(x86)\Python39\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe"
    )
    foreach ($path in $winPy) {
        if ($path -and (Test-Path $path)) {
            if ($path.ToLowerInvariant().EndsWith("\py.exe")) {
                Add-Candidate @($path, "-3")
            } else {
                Add-Candidate @($path)
            }
        }
    }

    # 3) Registry-based discovery (App Paths + PythonCore)
    $regPaths = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\python.exe",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\python.exe"
    )
    foreach ($rp in $regPaths) {
        try {
            $v = (Get-ItemProperty -Path $rp -ErrorAction Stop).'(default)'
            if ($v) { Add-Candidate @($v) }
        } catch {}
    }

    foreach ($base in @("HKLM:\SOFTWARE\Python\PythonCore", "HKCU:\SOFTWARE\Python\PythonCore")) {
        if (Test-Path $base) {
            Get-ChildItem $base -ErrorAction SilentlyContinue | ForEach-Object {
                $ip = Join-Path $_.PsPath "InstallPath"
                try {
                    $home = (Get-ItemProperty -Path $ip -ErrorAction Stop).'(default)'
                    if ($home) {
                        $exe = Join-Path $home "python.exe"
                        if (Test-Path $exe) { Add-Candidate @($exe) }
                    }
                } catch {}
            }
        }
    }

    # 4) Last fallback: Python launcher absolute path
    $pyLauncher = "$env:SystemRoot\py.exe"
    if (Test-Path $pyLauncher) { Add-Candidate @($pyLauncher, "-3") }

    return $candidates
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Launcher,
        [Parameter(Mandatory = $true)]
        [string[]]$CmdArgs
    )

    if ($Launcher.Count -eq 1) {
        & $Launcher[0] @CmdArgs 2>$null
    }
    else {
        & $Launcher[0] $Launcher[1] @CmdArgs 2>$null
    }
    return $LASTEXITCODE
}

function Test-Launcher {
    param([object[]]$Launcher)

    # Reject known embedded vendor runtimes that commonly fail venv/pip.
    $joined = ($Launcher -join " ").ToLowerInvariant()
    if ($joined.Contains("originlab") -or $joined.Contains("pydlls")) {
        return $false
    }

    $probe = @(
        "-c",
        "import sys, encodings, venv; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 2)"
    )
    $code = Invoke-Python -Launcher $Launcher -CmdArgs $probe
    return ($code -eq 0)
}

function Select-WorkingPython {
    $candidates = Get-PythonCandidates
    foreach ($candidate in $candidates) {
        try {
            if (Test-Launcher -Launcher $candidate) {
                return $candidate
            }
        } catch {
            continue
        }
    }

    $msg = @(
        "No usable Python launcher found.",
        "Tried: explicit -PythonExe, PATH commands (py/python/python3), common install paths, registry keys, and %SystemRoot%\\py.exe.",
        "Use: powershell -ExecutionPolicy Bypass -File .\\tools\\one_click_simulation.ps1 -PythonExe 'C:\\Path\\to\\python.exe'"
    ) -join " "
    throw $msg
}

$launcher = Select-WorkingPython
Write-Host ("Using launcher: " + ($launcher -join " ")) -ForegroundColor DarkCyan

Write-Host "[1/9] Setup venv" -ForegroundColor Yellow
$venvDir = Join-Path $repo ".venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"

if (!(Test-Path $venvPy)) {
    if (Test-Path $venvDir) {
        Write-Host " - Existing .venv looks incomplete. Recreating..." -ForegroundColor DarkYellow
        Remove-Item -Recurse -Force $venvDir
    }

    Write-Host " - Creating .venv via: -m venv .venv" -ForegroundColor DarkYellow
    $createCode = Invoke-Python -Launcher $launcher -CmdArgs @("-m", "venv", ".venv")
    if ($createCode -ne 0) {
        throw "venv creation failed (exit=$createCode)"
    }
}

if (!(Test-Path $venvPy)) {
    throw "venv python not found after creation: $venvPy"
}

Write-Host "[2/9] Upgrade packaging tools" -ForegroundColor Yellow
& $venvPy -m pip install --upgrade pip setuptools wheel

Write-Host "[3/9] Install dependencies" -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    & $venvPy -m pip install -r requirements.txt
} else {
    & $venvPy -m pip install numpy matplotlib scipy h5py pytest
}

Write-Host "[4/9] Baseline validation" -ForegroundColor Yellow
& $venvPy -m scripts.run_validation --out artifacts/baseline_report.md

if (-not $SkipTests) {
    Write-Host "[5/9] Run pytest" -ForegroundColor Yellow
    & $venvPy -m pytest -q
} else {
    Write-Host "[5/9] Skip pytest (-SkipTests)" -ForegroundColor DarkYellow
}

Write-Host "[6/9] Updated validation" -ForegroundColor Yellow
& $venvPy -m scripts.run_validation --out $OutReport

Write-Host "[7/9] Check HDF5 output" -ForegroundColor Yellow
if (Test-Path $OutH5) {
    Write-Host " - H5 exists: $OutH5" -ForegroundColor Green
} else {
    Write-Host " - H5 not found at $OutH5 (runner default is artifacts/rt_sweep.h5)." -ForegroundColor DarkYellow
}

Write-Host "[8/9] Summary" -ForegroundColor Green
Write-Host " - baseline: artifacts/baseline_report.md"
Write-Host " - updated : $OutReport"
Write-Host " - plots   : artifacts/plots"

Write-Host "[9/9] Updated report preview" -ForegroundColor Cyan
if (Test-Path $OutReport) {
    Get-Content $OutReport -TotalCount 60
} else {
    Write-Host "updated report not found: $OutReport" -ForegroundColor Red
}
