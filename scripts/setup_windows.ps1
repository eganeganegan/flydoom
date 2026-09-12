[CmdletBinding()]
param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [switch]$InstallVizDoom,
    [switch]$SkipTests,
    [ValidateRange(16, 100000000)]
    [int]$TrainingSteps = 128
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepositoryRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 -m venv .venv
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([version]$PythonVersion -lt [version]"3.12") {
        throw "FlyDoom requires Python 3.12 or newer; found $PythonVersion."
    }
    & python -m venv .venv
} else {
    throw "Python was not found. Install Python 3.12 from https://www.python.org/downloads/windows/."
}

$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment creation failed: $Python was not created."
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-Python -m pip install --upgrade pip
if ($Device -eq "cuda") {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "CUDA mode requires an NVIDIA driver and nvidia-smi on PATH."
    }
    Invoke-Python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
}

$Extras = if ($InstallVizDoom) { ".[dev,all]" } else { ".[dev]" }
Invoke-Python -m pip install -e $Extras

if ($Device -eq "cuda") {
    Invoke-Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; print(torch.cuda.get_device_name(0))"
}

if (-not $SkipTests) {
    Invoke-Python -m pytest
}

Invoke-Python scripts/train.py `
    experiment=flydoom_basic `
    data=mock_connectome `
    model=connectome_rate `
    seed=42 `
    device=$Device `
    data.node_count=100 `
    data.max_neurons=100 `
    training.total_steps=$TrainingSteps `
    training.rollout_steps=32 `
    training.epochs=1 `
    training.batch_size=16 `
    experiment.evaluation_episodes=1 `
    env.max_episode_steps=32

Write-Host "FlyDoom Windows smoke run completed. Artifacts are under outputs\."
