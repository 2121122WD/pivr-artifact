<#
RQ4 localization-stage stability runner for PiVR.

This script implements an SMiR-style localization analysis and varies only the
localization granularity parameter k. Other localization hyperparameters are
fixed to their default values in the paper.

Typical usage:
  .\run_rq4_localization_stability.ps1 -ClearExisting
  .\run_rq4_localization_stability.ps1 -ClearExisting -AllSettings
  .\run_rq4_localization_stability.ps1 -DryRun

Notes:
  1. Run this script from the repository root, or pass -RepoRoot explicitly.
  2. Adjust target script paths if your project layout differs.
  3. The script is for stability analysis, not hyperparameter search.
#>

param(
    # Repository root. Default: current directory.
    [string]$RepoRoot = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),

    # Python executable. Use python from the active environment by default.
    [string]$PythonExe = "python",

    # Runner script.
    [string]$RunnerScript = "PiVR\experiments\run_param_sensitivity.py",

    # Target experiment entrypoints.
    [string]$BackdoorScript = "PiVR\experiments\exp_backdoor_removal_multi.py",
    [string]$SafetyScript   = "PiVR\experiments\exp_safety_acas.py",
    [string]$FairnessScript = "PiVR\experiments\run_fairness_pivr_benchmark.py",

    # Grid files.
    [string]$BackdoorGrid = "PiVR\experiments\grid_backdoor.json",
    [string]$SafetyGrid   = "PiVR\experiments\grid_safety.json",
    [string]$FairnessGrid = "PiVR\experiments\grid_fairness.json",

    # Output directory.
    [string]$ResultsDir = "results\rq4_localization_stability",

    # Run controls.
    [switch]$RunBackdoor = $true,
    [switch]$RunSafety   = $true,
    [switch]$RunFairness = $true,
    [switch]$AllSettings,
    [switch]$ClearExisting,
    [switch]$DryRun,

    # Seeds. Use one seed for quick stability inspection, or multiple for a stronger study.
    [int[]]$Seeds = @(0,1,2),

    # Representative settings. These include both easy and harder settings.
    [string[]]$BackdoorDatasets = @("MNIST","Fashion-MNIST","GTSRB","CIFAR-10"),
#     [string[]]$SafetySubnetworks = @("N2,9","N3,3","N1,9"),
    [string[]]$FairnessSettings = @("bank:age","census:age","census:gender","census:race","credit:age","credit:gender")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Join-RepoPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $RepoRoot $Path
}

function Ensure-File([string]$Path, [string]$Name) {
    $Full = Join-RepoPath $Path
    if (!(Test-Path $Full)) {
        throw "$Name not found: $Full"
    }
    return $Full
}

function Invoke-Sweep {
    param(
        [string]$Task,
        [string]$Script,
        [string]$Grid,
        [string]$BaseArgs,
        [string]$Output
    )

    $Runner = Ensure-File $RunnerScript "Runner script"
    $TargetScript = Ensure-File $Script "Target script for $Task"
    $GridPath = Ensure-File $Grid "Grid file for $Task"
    $OutputPath = Join-RepoPath $Output

    $OutputParent = Split-Path $OutputPath -Parent
    if (!(Test-Path $OutputParent)) {
        New-Item -ItemType Directory -Force -Path $OutputParent | Out-Null
    }

    $Args = @(
        $Runner,
        "--task", $Task,
        "--script", $TargetScript,
        "--grid", $GridPath,
        "--base-args", $BaseArgs,
        "--output", $OutputPath,
        "--top-k", "5"
    )

    if ($DryRun) {
        $Args += "--dry-run"
    }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Running localization-stage stability sweep" -ForegroundColor Cyan
    Write-Host "Task       : $Task" -ForegroundColor Cyan
    Write-Host "Base args  : $BaseArgs" -ForegroundColor Cyan
    Write-Host "Output     : $OutputPath" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "$PythonExe $($Args -join ' ')"

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Sweep failed for task=$Task, output=$OutputPath"
    }
}

function Safe-Name([string]$Name) {
    return ($Name -replace '[,/:\\ ]+', '_' -replace '[^A-Za-z0-9_.-]', '_')
}

Push-Location $RepoRoot
try {
    $ResolvedResultsDir = Join-RepoPath $ResultsDir
    if ($ClearExisting -and (Test-Path $ResolvedResultsDir)) {
        Write-Host "Removing existing results directory: $ResolvedResultsDir" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $ResolvedResultsDir
    }
    if (!(Test-Path $ResolvedResultsDir)) {
        New-Item -ItemType Directory -Force -Path $ResolvedResultsDir | Out-Null
    }

    # If requested, run all subjects used in the main evaluation.
    if ($AllSettings) {
        $BackdoorDatasets = @("GTSRB", "MNIST", "Fashion-MNIST", "CIFAR-10")
        $SafetySubnetworks = @("N2,9", "N3,3", "N1,9")
        $FairnessSettings = @(
            "census:race",
            "census:age",
            "census:gender",
            "bank:age",
            "credit:age",
            "credit:gender"
        )
    }

    foreach ($Seed in $Seeds) {
        if ($RunBackdoor) {
            foreach ($Dataset in $BackdoorDatasets) {
                $Name = Safe-Name "backdoor_${Dataset}_seed${Seed}"
                $Output = Join-Path $ResultsDir "$Name.csv"
                $BaseArgs = "--dataset $Dataset --algorithm pivr --seed $Seed"
                Invoke-Sweep -Task "backdoor" -Script $BackdoorScript -Grid $BackdoorGrid -BaseArgs $BaseArgs -Output $Output
            }
        }

        if ($RunSafety) {
            foreach ($Subnet in $SafetySubnetworks) {
                $Name = Safe-Name "safety_${Subnet}_seed${Seed}"
                $Output = Join-Path $ResultsDir "$Name.csv"
                $BaseArgs = "--subnetwork $Subnet --seed $Seed"
                Invoke-Sweep -Task "safety" -Script $SafetyScript -Grid $SafetyGrid -BaseArgs $BaseArgs -Output $Output
            }
        }

        if ($RunFairness) {
            foreach ($Setting in $FairnessSettings) {
                $Parts = $Setting.Split(":")
                if ($Parts.Count -ne 2) {
                    throw "Invalid fairness setting '$Setting'. Expected format dataset:attribute, e.g., credit:age"
                }
                $Dataset = $Parts[0]
                $Attr = $Parts[1]
                $Name = Safe-Name "fairness_${Dataset}_${Attr}_seed${Seed}"
                $Output = Join-Path $ResultsDir "$Name.csv"
                $BaseArgs = "--dataset $Dataset --attribute $Attr --seed $Seed"
                Invoke-Sweep -Task "fairness" -Script $FairnessScript -Grid $FairnessGrid -BaseArgs $BaseArgs -Output $Output
            }
        }
    }

    $CombinedOutput = Join-Path $ResolvedResultsDir "rq3_ablation_result.csv"
    $CsvFiles = Get-ChildItem -Path $ResolvedResultsDir -Filter *.csv -File | Where-Object { $_.Name -ne "rq3_ablation_result.csv" } | Sort-Object Name
    if ($CsvFiles.Count -gt 0) {
        $AllRows = @()
        foreach ($CsvFile in $CsvFiles) {
            $Imported = Import-Csv -Path $CsvFile.FullName
            foreach ($Row in $Imported) {
                $KValue = $null
                if ($Row.PSObject.Properties.Match('k').Count -and $Row.k -ne '') {
                    $KValue = $Row.k
                } elseif ($Row.PSObject.Properties.Match('layer_k_ratio_cap').Count -and $Row.layer_k_ratio_cap -ne '') {
                    $KValue = $Row.layer_k_ratio_cap
                }
                if ($null -ne $KValue) {
                    $Row | Add-Member -NotePropertyName k -NotePropertyValue $KValue -Force
                }
                $AllRows += $Row
            }
        }
        $OutputColumns = @(
            'timestamp','task','script','returncode','stdout_tail','stderr_tail',
            'seed','dataset','attribute','subnetwork','algorithm','sbfl_strategy','alpha','k','layer_k_ratio_cap',
            'repair_metric_name','metric_before','metric_after','acc_before','acc_after','drawdown',
            'modified_params','modified_params_ratio','loc_time','repair_time','total_time'
        )
        $AllRows |
            Select-Object $OutputColumns |
            Export-Csv -Path $CombinedOutput -NoTypeInformation -Encoding UTF8
        Write-Host "Combined CSV written to: $CombinedOutput" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "All requested localization-stage stability sweeps completed." -ForegroundColor Green
    Write-Host "Results directory: $ResolvedResultsDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "Suggested paper wording:" -ForegroundColor DarkCyan
    Write-Host "  RQ4 is a parameter stability analysis rather than a hyperparameter search."
    Write-Host "  We vary localization-stage parameters around the default configuration and report the resulting range of repair metric, accuracy, and drawdown."
}
finally {
    Pop-Location
}
