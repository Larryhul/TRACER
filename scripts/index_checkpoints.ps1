param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$projectRootPath = (Resolve-Path -LiteralPath $ProjectRoot).Path
$experimentRoot = Join-Path $projectRootPath 'experiments'
$outputPath = Join-Path $experimentRoot 'CHECKPOINTS.csv'
$latestOutputPath = Join-Path $experimentRoot 'LATEST_CHECKPOINTS.csv'

$rows = Get-ChildItem -LiteralPath $experimentRoot -File -Recurse -ErrorAction Stop |
    Where-Object { $_.Extension -in @('.pt', '.pth', '.ckpt') } |
    Sort-Object FullName |
    ForEach-Object {
        $relativeNativePath = $_.FullName.Substring($projectRootPath.Length + 1)
        $relativePath = $relativeNativePath.Replace('\', '/')
        $relativeDirectory = [IO.Path]::GetDirectoryName($relativeNativePath).Replace('\', '/')
        $epoch = $null
        if ($_.BaseName -match '(?:^|_)epoch(?<epoch>\d+)(?:_|$)') {
            $epoch = [int]$Matches.epoch
        }

        [pscustomobject]@{
            path          = $relativePath
            directory     = $relativeDirectory
            filename      = $_.Name
            series        = ($_.BaseName -replace '_epoch\d+$', '')
            epoch         = $epoch
            size_mb       = [math]::Round($_.Length / 1MB, 3)
            modified_time = $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
            sha256        = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

$rows | Export-Csv -LiteralPath $outputPath -NoTypeInformation -Encoding UTF8
$activeRows = $rows | Where-Object { $_.path -notlike 'experiments/archive/*' }
$latestRows = $activeRows |
    Group-Object { "{0}|{1}" -f $_.directory, $_.series } |
    ForEach-Object {
        $_.Group | Sort-Object `
            @{ Expression = { if ($null -eq $_.epoch) { -1 } else { [int]$_.epoch } }; Descending = $true }, `
            @{ Expression = { $_.modified_time }; Descending = $true } |
            Select-Object -First 1
    } |
    Sort-Object directory, series
$latestRows | Export-Csv -LiteralPath $latestOutputPath -NoTypeInformation -Encoding UTF8

Write-Output ("Indexed {0} checkpoints in {1}" -f $rows.Count, $outputPath)
Write-Output ("Selected {0} latest checkpoints in {1}" -f $latestRows.Count, $latestOutputPath)
