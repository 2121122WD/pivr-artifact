param(
    [string]$RepoRoot = "",
    [int[]]$Seeds = @(0,1,2),
    [switch]$RunAnalysis,
    [switch]$ClearExisting,
    [switch]$SkipBackdoor,
    [switch]$SkipSafety,
    [switch]$SkipFairness,

    # Default design: use representative settings across all tasks.
    [switch]$AllSettings,
    [string[]]$BackdoorDatasets = @(),
    [string[]]$SafetySubnetworks = @(),
    [string[]]$FairnessSettings = @(),

    # Use the best localization budgets selected by the RQ4 k-sweep.
    # Change this if the target scripts expose the localization budget with a different argument name.
    [string]$LocalizationKArgName = "--k",

    # Repair-stage sweep: analyze a single balance parameter lambda.
    # The target scripts should implement lambda_clean = lambda and lambda_task = 1 - lambda.
    [double[]]$BackdoorLambdaClean  = @(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    [double[]]$SafetyLambdaClean    = @(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    [double[]]$FairnessLambdaClean  = @(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
)

$ErrorActionPreference = 'Continue'

function Resolve-RepoRoot {
    param([string]$GivenRoot)

    if ($GivenRoot -ne "") {
        $candidate = (Resolve-Path $GivenRoot).Path
        if ((Test-Path (Join-Path $candidate "PiVR\experiments")) -and
            (Test-Path (Join-Path $candidate "Socrates\source\run_fairness_pivr_benchmark.py"))) {
            return $candidate
        }
        throw "Invalid -RepoRoot: $candidate. It must contain PiVR\experiments_birdnn and Socrates\source\run_fairness_pivr_benchmark.py"
    }

    $cur = (Resolve-Path $PSScriptRoot).Path
    while ($true) {
        if ((Test-Path (Join-Path $cur "PiVR\experiments)) -and
            (Test-Path (Join-Path $cur "Socrates\source\run_fairness_pivr_benchmark.py"))) {
            return $cur
        }
        $parent = Split-Path $cur -Parent
        if ($parent -eq $cur -or $parent -eq "") { break }
        $cur = $parent
    }

    throw "Cannot auto-detect repository root from $PSScriptRoot. Please run with -RepoRoot ""D:\paper_methods\PiVR_with_baseline\CCBR"""
}

$ROOT = Resolve-RepoRoot $RepoRoot
$OUT = Join-Path $ROOT "PiVR\experiments_birdnn\results\rq4_repair_stage_stability_result.csv"
$ANALYZER = Join-Path $ROOT "PiVR\experiments_birdnn\analyze_rq5_repair_sensitivity_plot.py"

Write-Host "[INFO] Repo root: $ROOT"
Write-Host "[INFO] Fairness script: $(Join-Path $ROOT "Socrates\source\run_fairness_pivr_benchmark.py")"

if ($ClearExisting -and (Test-Path $OUT)) { Remove-Item $OUT -Force }

function Invoke-SafeRun {
    param([string]$Command)
    Write-Host "[RUN] $Command"
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) { Write-Warning "Command failed with exit code $LASTEXITCODE" }
    } catch {
        Write-Warning $_.Exception.Message
    }
}

# Representative design.  It intentionally includes one saturated/easy subject and
# one harder subject when possible.  This makes the RQ4 story stronger than using
# only MNIST / ACAS N2,9 / Bank-age.
if ($BackdoorDatasets.Count -eq 0) {
    if ($AllSettings) { $BackdoorDatasets = @("MNIST","Fashion-MNIST","GTSRB","CIFAR-10") }
    else { $BackdoorDatasets = @("MNIST","Fashion-MNIST","GTSRB","CIFAR-10") }
}
if ($SafetySubnetworks.Count -eq 0) {
    if ($AllSettings) { $SafetySubnetworks = @("N2,9","N3,3","N1,9") }
    else { $SafetySubnetworks = @("N2,9","N3,3","N1,9") }
}
if ($FairnessSettings.Count -eq 0) {
    if ($AllSettings) { $FairnessSettings = @("bank:age","census:age","census:gender","census:race","credit:age","credit:gender") }
    else { $FairnessSettings = @("bank:age","census:age","census:gender","census:race","credit:age","credit:gender") }
}

$backdoorParams = @(
    @{ Name='lambda_clean'; Values=$BackdoorLambdaClean }
)
$safetyParams = @(
    @{ Name='lambda_clean'; Values=$SafetyLambdaClean }
)
$fairnessParams = @(
    @{ Name='lambda_clean'; Values=$FairnessLambdaClean }
)

# Best localization budgets from the RQ4 k-selection experiment.
# Backdoor and safety use the latest sweep after 2026-06-26 14:32:07; fairness uses the previous completed sweep.
$BackdoorBestK = @{
    "MNIST" = 6
    "Fashion-MNIST" = 4
    "GTSRB" = 12
    "CIFAR-10" = 1
}
$SafetyBestK = @{
    "N2,9" = 2
    "N3,3" = 6
    "N1,9" = 4
}
$FairnessBestK = @{
    "bank:age" = 1
    "census:age" = 4
    "census:gender" = 4
    "census:race" = 4
    "credit:age" = 3
    "credit:gender" = 9
}

function Get-BestKArg {
    param([int]$K)
    if ($LocalizationKArgName -eq "") { return "" }
    return "$LocalizationKArgName $K"
}

Write-Host "[INFO] Output: $OUT"
Write-Host "[INFO] Seeds: $($Seeds -join ',')"
Write-Host "[INFO] Backdoor datasets: $($BackdoorDatasets -join ',')"
Write-Host "[INFO] Safety subnetworks: $($SafetySubnetworks -join ',')"
Write-Host "[INFO] Fairness settings: $($FairnessSettings -join ',')"
Write-Host "[INFO] Localization k argument: $LocalizationKArgName"
Write-Host "[INFO] Repair sweep: lambda_clean = lambda, lambda_task = 1 - lambda"

foreach ($seed in $Seeds) {
    if (-not $SkipBackdoor) {
        foreach ($dataset in $BackdoorDatasets) {
            foreach ($p in $backdoorParams) {
                foreach ($v in $p.Values) {
                    if (-not $BackdoorBestK.ContainsKey($dataset)) { Write-Warning "No selected k for backdoor dataset: $dataset"; continue }
                    $locK = $BackdoorBestK[$dataset]
                    $kArg = Get-BestKArg $locK
                    $cmd = "python `"$ROOT\PiVR\experiments_birdnn\exp_backdoor_removal_multi.py`" --dataset `"$dataset`" --algorithm pivr --seed $seed $kArg --rq5_repair_sensitivity --rq5_param $($p.Name) --rq5_value $v --rq5_output `"$OUT`""
                    Invoke-SafeRun $cmd
                }
            }
        }
    }
    if (-not $SkipSafety) {
        foreach ($sub in $SafetySubnetworks) {
            foreach ($p in $safetyParams) {
                foreach ($v in $p.Values) {
                    if (-not $SafetyBestK.ContainsKey($sub)) { Write-Warning "No selected k for safety subnetwork: $sub"; continue }
                    $locK = $SafetyBestK[$sub]
                    $kArg = Get-BestKArg $locK
                    $cmd = "python `"$ROOT\PiVR\experiments_birdnn\exp_safety_acas.py`" --subnetwork `"$sub`" --seed $seed $kArg --rq5_repair_sensitivity --rq5_param $($p.Name) --rq5_value $v --rq5_output `"$OUT`""
                    Invoke-SafeRun $cmd
                }
            }
        }
    }
    if (-not $SkipFairness) {
        foreach ($setting in $FairnessSettings) {
            $parts = $setting.Split(':')
            if ($parts.Count -ne 2) { Write-Warning "Skipping invalid fairness setting: $setting"; continue }
            $dataset = $parts[0]; $attr = $parts[1]
            foreach ($p in $fairnessParams) {
                foreach ($v in $p.Values) {
                    $fairKey = "$dataset`:$attr"
                    if (-not $FairnessBestK.ContainsKey($fairKey)) { Write-Warning "No selected k for fairness setting: $fairKey"; continue }
                    $locK = $FairnessBestK[$fairKey]
                    $kArg = Get-BestKArg $locK
                    $cmd = "python `"$ROOT\Socrates\source\run_fairness_pivr_benchmark.py`" --dataset $dataset --attribute $attr --seed $seed $kArg --rq5_repair_sensitivity --rq5_param $($p.Name) --rq5_value $v --rq5_output `"$OUT`""
                    Invoke-SafeRun $cmd
                }
            }
        }
    }
}

function Strip-DrawdownColumn {
    param([string]$CsvPath)
    if (-not (Test-Path $CsvPath)) { return }
    $rows = Import-Csv -Path $CsvPath
    if (-not $rows) { return }
    $cols = @($rows[0].PSObject.Properties.Name | Where-Object { $_ -ne 'drawdown' })
    $rows |
        Select-Object $cols |
        Export-Csv -Path $CsvPath -NoTypeInformation -Encoding UTF8
}

Strip-DrawdownColumn $OUT

if ($RunAnalysis) {
    python $ANALYZER --input $OUT --out-dir (Split-Path $OUT -Parent) --fig-dir "$ROOT\ACM_Conference_Proceedings_Primary_Article_Template\figures"
}
