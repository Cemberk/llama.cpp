# Gathers what the Vulkan backend needs to build on Windows without the
# LunarG SDK installer, which needs administrator rights:
#   - Vulkan-Headers and SPIRV-Headers, installed to one prefix, so the
#     include folder holds vulkan/ and spirv/ as the SDK's does
#   - vulkan-1.lib, generated from the vulkan-1.dll the GPU driver installed
#   - glslc, from Google's prebuilt shaderc
# and prints the cmake command that uses them. Everything goes under -Root.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows-vulkan-noadmin.ps1
#
# Needs git, CMake and the Visual Studio Build Tools (C++).

param(
    [string]$Root = "$env:USERPROFILE\vulkan-local"
)

$ErrorActionPreference = 'Stop'
# Invoke-WebRequest is many times slower while it draws a progress bar.
$ProgressPreference = 'SilentlyContinue'

$install = Join-Path $Root 'install'
New-Item -ItemType Directory -Force -Path $Root, (Join-Path $install 'lib') | Out-Null

$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) { $cmake = 'C:\Program Files\CMake\bin\cmake.exe' }
if (-not (Test-Path $cmake)) { throw 'CMake was not found: winget install Kitware.CMake' }

# 1. Headers, installed to one prefix.
foreach ($repo in 'Vulkan-Headers', 'SPIRV-Headers') {
    $src = Join-Path $Root $repo
    if (-not (Test-Path $src)) {
        git clone --depth 1 "https://github.com/KhronosGroup/$repo.git" $src
        if ($LASTEXITCODE -ne 0) { throw "git clone $repo failed" }
    }
    & $cmake -S $src -B (Join-Path $src 'build') "-DCMAKE_INSTALL_PREFIX=$install" -DSPIRV_HEADERS_ENABLE_TESTS=OFF | Out-Null
    & $cmake --install (Join-Path $src 'build') | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "installing $repo failed" }
}

# 2. vulkan-1.lib from the driver's loader.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'The Visual Studio Build Tools (C++) were not found.' }
$msvc = Get-ChildItem (Join-Path $vs 'VC\Tools\MSVC') | Sort-Object Name | Select-Object -Last 1
$bin = Join-Path $msvc.FullName 'bin\Hostx64\x64'
$dll = Join-Path $env:SystemRoot 'System32\vulkan-1.dll'
if (-not (Test-Path $dll)) { throw 'vulkan-1.dll was not found: install or update the GPU driver.' }
$exports = & (Join-Path $bin 'dumpbin.exe') /NOLOGO /EXPORTS $dll
$names = $exports | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)') { $Matches[1] }
}
$def = Join-Path $Root 'vulkan-1.def'
(@('LIBRARY vulkan-1', 'EXPORTS') + ($names | ForEach-Object { "  $_" })) | Set-Content -Encoding ascii $def
Push-Location (Join-Path $install 'lib')
& (Join-Path $bin 'lib.exe') /NOLOGO "/DEF:$def" /MACHINE:X64 | Out-Null
Pop-Location
if (-not (Test-Path (Join-Path $install 'lib\vulkan-1.lib'))) { throw 'vulkan-1.lib was not created' }
Write-Host "vulkan-1.lib: $($names.Count) functions"

# 3. glslc from the newest shaderc build in Google's bucket.
$glslc = Join-Path $Root 'shaderc\install\bin\glslc.exe'
if (-not (Test-Path $glslc)) {
    $bucket = 'https://storage.googleapis.com/shaderc'
    $prefix = 'artifacts/prod/graphics_shader_compiler/shaderc/windows-vs2022-amd64-release/continuous/'
    $keys = @()
    $marker = ''
    do {
        [xml]$list = (Invoke-WebRequest -UseBasicParsing "$bucket`?prefix=$prefix&marker=$marker").Content
        $keys += $list.ListBucketResult.Contents | ForEach-Object { $_.Key } | Where-Object { $_ -like '*/install.zip' }
        $marker = $list.ListBucketResult.NextMarker
        if (-not $marker -and $list.ListBucketResult.IsTruncated -eq 'true') {
            $marker = ($list.ListBucketResult.Contents | Select-Object -Last 1).Key
        }
    } while ($list.ListBucketResult.IsTruncated -eq 'true')
    $newest = $keys | Sort-Object { [int](($_ -replace [regex]::Escape($prefix), '') -split '/')[0] } | Select-Object -Last 1
    if (-not $newest) { throw 'No shaderc build was found in the bucket.' }
    $zip = Join-Path $Root 'shaderc.zip'
    Invoke-WebRequest -UseBasicParsing "$bucket/$newest" -OutFile $zip
    Expand-Archive -Force $zip (Join-Path $Root 'shaderc')
}
& $glslc --version | Select-Object -First 1

$inc = ($install -replace '\\', '/') + '/include'
$lib = ($install -replace '\\', '/') + '/lib/vulkan-1.lib'
$spirv = ($install -replace '\\', '/') + '/share/cmake/SPIRV-Headers'
$glslcFwd = $glslc -replace '\\', '/'
Write-Host ''
Write-Host 'Configure the Vulkan build with:'
Write-Host "cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF -DGGML_VULKAN=ON -DVulkan_INCLUDE_DIR=$inc -DVulkan_LIBRARY=$lib -DVulkan_GLSLC_EXECUTABLE=$glslcFwd -DSPIRV-Headers_DIR=$spirv"
Write-Host 'cmake --build build --config Release --target llama-server -j 4'
