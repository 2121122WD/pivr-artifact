param(
  [int[]]$Seeds = @(0,1,2),
  [ValidateSet('screen','representative','custom')]
  [string]$Mode = 'screen',
  [string]$Out = 'PiVR/experiments/results/rq3_ablation_result111.csv',
  [switch]$ClearExisting,
  [switch]$RunAnalysis,
  [string]$Python = 'python',

  # RQ5-style sensitivity runs should keep each parameter sweep local and one-at-a-time.
  [string[]]$BackdoorDatasets = @('MNIST','Fashion-MNIST','GTSRB','CIFAR-10'),
  [string[]]$SafetySubnetworks = @('N2,9','N3,3','N1,9'),
  # Format: dataset:attribute
  [string[]]$FairnessSettings = @('bank:age','census:gender','census:race','census:age','credit:age','credit:gender'),

  # Representative settings for final multi-seed run.
  [string]$BackdoorRepresentative = 'GTSRB',
  [string]$SafetyRepresentative = 'N1,9',
  [string]$FairnessRepresentative = 'bank:age',

  # RQ3 variants aligned with the current paper design.
  [string[]]$Ablations = @('full','no_localization','no_verification','no_pathway_constraint'),
# ,'no_localization','no_verification','no_pathway_constraint'

  # Dataset-specific k values for the current analysis.
  [hashtable]$BackdoorKMap = @{ 'GTSRB' = 12; 'MNIST' = 6; 'Fashion-MNIST' = 4; 'CIFAR-10' = 1 },
  [hashtable]$FairnessKMap = @{ 'bank:age' = 1; 'census:age' = 6; 'census:gender' = 4; 'census:race' = 4; 'credit:age' = 4; 'credit:gender' = 9 },
  [hashtable]$SafetyKMap = @{ 'N1,9' = 1 },

  # Task switches.
  [switch]$SkipBackdoor,
  [switch]$SkipSafety,
  [switch]$SkipFairness
)

$ErrorActionPreference = 'Stop'

# This script can be run from anywhere inside the repo, or from the repo root.
#
# Output behavior:
#   By default, this script APPENDS new rows to -Out.
#   It never removes the existing RQ3 result CSV unless -ClearExisting is explicitly set.
#
# Modes:
#   screen         : run all candidate settings, usually with -Seeds 0.
#   representative: run one selected setting per task, usually with -Seeds 0,1,2.
#   custom         : run the user-provided BackdoorDatasets/SafetySubnetworks/FairnessSettings lists.

function Resolve-PythonCommand {
  param([string]$RequestedPython)
  if ($RequestedPython -ne 'python') { return $RequestedPython }
  if ($env:CONDA_PREFIX) {
    $Candidate = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path $Candidate) { return $Candidate }
  }
  return (Get-Command python).Source
}

$Python = Resolve-PythonCommand $Python
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location $RepoRoot

$ResolvedOut = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $RepoRoot $Out }
$ResultsDir = Split-Path $ResolvedOut -Parent
if ($ResultsDir -and !(Test-Path $ResultsDir)) {
  New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
}

if ((Test-Path $ResolvedOut) -and $ClearExisting) {
  Write-Host "Clearing existing RQ3 result file: $ResolvedOut"
  Remove-Item $ResolvedOut
} elseif (Test-Path $ResolvedOut) {
  Write-Host "Appending to existing RQ3 result file: $ResolvedOut"
} else {
  Write-Host "Creating new RQ3 result file: $ResolvedOut"
}

function Invoke-PythonChecked {
  param(
    [Parameter(Mandatory=$true)][string[]]$Args
  )

  & $Python @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $($LASTEXITCODE): $Python $($Args -join ' ')"
  }
}

function Resolve-ScriptPath {
  param([Parameter(Mandatory=$true)][string]$RelativePath)
  if ([System.IO.Path]::IsPathRooted($RelativePath)) { return $RelativePath }
  $repoRoot = $RepoRoot
  $candidates = @(
    (Join-Path $PSScriptRoot $RelativePath),
    (Join-Path (Split-Path $PSScriptRoot -Parent) $RelativePath),
    (Join-Path $repoRoot $RelativePath)
  )
  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) { return $candidate }
  }
  return (Join-Path $repoRoot $RelativePath)
}

function Parse-FairnessSetting {
  param([Parameter(Mandatory=$true)][string]$Setting)
  $parts = $Setting.Split(':')
  if ($parts.Count -ne 2) {
    throw "Invalid fairness setting '$Setting'. Expected format dataset:attribute, e.g., bank:age."
  }
  return @{ Dataset = $parts[0]; Attribute = $parts[1] }
}

if ($Mode -eq 'representative') {
  $BackdoorDatasets = @($BackdoorRepresentative)
  $SafetySubnetworks = @($SafetyRepresentative)
  $FairnessSettings = @($FairnessRepresentative)
}

function Get-KForBackdoor {
  param([string]$Dataset)
  if ($BackdoorKMap.ContainsKey($Dataset)) { return $BackdoorKMap[$Dataset] }
  return $null
}

function Get-KForSafety {
  param([string]$Subnetwork)
  if ($SafetyKMap.ContainsKey($Subnetwork)) { return $SafetyKMap[$Subnetwork] }
  return $null
}

function Get-KForFairness {
  param([string]$Setting)
  if ($FairnessKMap.ContainsKey($Setting)) { return $FairnessKMap[$Setting] }
  return $null
}

Write-Host "Using Python command: $Python"
Write-Host "RQ3 output: $ResolvedOut"
Write-Host "Mode: $Mode"
Write-Host "Seeds: $($Seeds -join ', ')"
Write-Host "Ablations: $($Ablations -join ', ')"
Write-Host "ClearExisting: $ClearExisting"
if (-not $SkipBackdoor) { Write-Host "Backdoor datasets: $($BackdoorDatasets -join ', ')" }
if (-not $SkipSafety) { Write-Host "Safety subnetworks: $($SafetySubnetworks -join ', ')" }
if (-not $SkipFairness) { Write-Host "Fairness settings: $($FairnessSettings -join ', ')" }

foreach ($seed in $Seeds) {
  foreach ($ablation in $Ablations) {
    if (-not $SkipBackdoor) {
      foreach ($dataset in $BackdoorDatasets) {
        $k = Get-KForBackdoor $dataset
        Write-Host "Running Backdoor dataset=$dataset ablation=$ablation seed=$seed k=$k"
        $args = @(
          (Resolve-ScriptPath 'PiVR/experiments/exp_backdoor_removal_multi.py'),
          '--dataset', $dataset,
          '--ablation', $ablation,
          '--seed', "$seed",
          '--rq3_output', $ResolvedOut
        )
        if ($null -ne $k) { $args += @('--k', "$k") }
        Invoke-PythonChecked $args
      }
    }

    if (-not $SkipSafety) {
      foreach ($subnetwork in $SafetySubnetworks) {
        $k = Get-KForSafety $subnetwork
        Write-Host "Running Safety ACAS subnetwork=$subnetwork ablation=$ablation seed=$seed k=$k"
        $args = @(
          (Resolve-ScriptPath 'PiVR/experiments/exp_safety_acas.py'),
          '--subnetwork', $subnetwork,
          '--ablation', $ablation,
          '--seed', "$seed",
          '--rq3_output', $Out
        )
        if ($null -ne $k) { $args += @('--k', "$k") }
        Invoke-PythonChecked $args
      }
    }

    if (-not $SkipFairness) {
      foreach ($setting in $FairnessSettings) {
        $fs = Parse-FairnessSetting $setting
        $k = Get-KForFairness $setting
        Write-Host "Running Fairness dataset=$($fs.Dataset) attribute=$($fs.Attribute) ablation=$ablation seed=$seed k=$k"
        $args = @(
          (Resolve-ScriptPath 'Socrates/source/run_fairness_pivr_benchmark.py'),
          '--dataset', $fs.Dataset,
          '--attribute', $fs.Attribute,
          '--ablation', $ablation,
          '--seed', "$seed",
          '--rq3_output', $Out
        )
        if ($null -ne $k) { $args += @('--k', "$k") }
        Invoke-PythonChecked $args
      }
    }
  }
}

Write-Host "All RQ3 ablation runs completed. Appended output: $Out"

if ($RunAnalysis) {
  Write-Host "Running RQ3 analysis. This regenerates derived summary/table/figure files but does not modify the raw result CSV."
  Invoke-PythonChecked @(
    (Resolve-ScriptPath 'PiVR/experiments/analyze_rq3_ablation.py'),
    '--input', $ResolvedOut
  )
}
