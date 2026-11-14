# Offline source-contract negative tests for the *actual* serving/server.cpp.
# Runs on Windows PowerShell 5.1 with no Python, compiler, sockets or GPU.
# Structural checks DO NOT replace dynamic Linux fake PP2 HTTP/TSan tests.
$ErrorActionPreference = 'Stop'
$sourcePath = Join-Path (Split-Path -Parent $PSScriptRoot) 'server.cpp'
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "missing serving/server.cpp: $sourcePath"
}
$source = Get-Content -Raw -LiteralPath $sourcePath
$checks = 0
function Assert-Contract([bool]$condition, [string]$label) {
    if (-not $condition) { throw "FAIL: $label" }
    $script:checks++
    Write-Output "PASS: $label"
}
function Has-Atomic-Tokenizer([string]$text) {
    return $text.Contains('std::atomic<bool> incremental_{false};') -and
        $text.Contains('incremental_.load(std::memory_order_relaxed)') -and
        $text.Contains('incremental_.store(false, std::memory_order_relaxed)') -and
        $text.Contains('incremental_.store(ready.value("incremental", false),')
}
function Has-Terminal-Wakeup([string]$text) {
    return $text.Contains('void fail_terminal_transport_noexcept(') -and
        $text.Contains('job->terminal_transport_failed = true;') -and
        $text.Contains('job->done.store(true);') -and
        $text.Contains('job->cv.notify_all();') -and
        $text.Contains('void complete(const std::shared_ptr<Job>& job,') -and
        $text.Contains('noexcept {') -and
        $text.Contains('fail_terminal_transport_noexcept(job);')
}
function Has-Atomic-Terminal-Frames([string]$text) {
    $start = $text.IndexOf('void complete_impl(')
    $end = $text.IndexOf('bool admit(', $start)
    if ($start -lt 0 -or $end -lt $start) { return $false }
    $body = $text.Substring($start, $end - $start)
    $build = $body.IndexOf('std::string terminal_sse;')
    $commit = $body.IndexOf('job->stream_chunks.push_back(std::move(terminal_sse));')
    $done = $body.IndexOf('job->done.store(true);')
    $lastAppend = $body.LastIndexOf('job->stream_chunks.push_back(')
    return ($build -ge 0 -and $commit -gt $build -and
        $done -gt $commit -and $lastAppend -eq $commit -and
        $body.Contains('terminal_sse += "data: [DONE]\n\n";'))
}
function Has-Allocation-Free-Sse-Fallback([string]$text) {
    return $text.Contains('constexpr char kTerminalTransportFailureSse[]') -and
        $text.Contains('job->terminal_transport_failed') -and
        $text.Contains('sink.write(kTerminalTransportFailureSse,') -and
        $text.Contains('sizeof(kTerminalTransportFailureSse) - 1') -and
        $text.Contains('job->stream_chunks.clear();') -and
        $text.Contains('job->stream_buffered_bytes = 0;')
}

Assert-Contract (Has-Atomic-Tokenizer $source) 'tokenizer restart uses atomic capability load/store'
Assert-Contract (Has-Terminal-Wakeup $source) 'noexcept callback always sets done and notifies'
Assert-Contract (Has-Atomic-Terminal-Frames $source) 'finish+usage+DONE staged before one queue commit'
Assert-Contract (Has-Allocation-Free-Sse-Fallback $source) 'static SSE error+DONE after terminal allocation failure'
Assert-Contract ($source.Contains('HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE')) 'CPU-only partial-success failpoint present'
Assert-Contract ($source.Contains('HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE')) 'CPU-only terminal queue failpoint present'
Assert-Contract ($source.Contains('HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE')) 'CPU-only callback-event failpoint preserved'
Assert-Contract ($source.Contains('HERMES_TEST_FAIL_COMPLETE_ONCE')) 'CPU-only JSON completion failpoint preserved'
Assert-Contract ($source.Contains('options.on_terminal_delivery_failure =')) 'scheduler emergency callback hook preserved'

# CPU-only in-memory source mutations: the exact checker must reject each
# forbidden regression without editing production or creating test binaries.
$mutated = $source.Replace('std::atomic<bool> incremental_{false};',
                            'bool incremental_{false};')
Assert-Contract (-not (Has-Atomic-Tokenizer $mutated)) 'NEGATIVE plain-bool tokenizer regression refused'
$mutated = $source.Replace('job->terminal_transport_failed = true;',
                            'job->terminal_transport_failed = false;')
Assert-Contract (-not (Has-Terminal-Wakeup $mutated)) 'NEGATIVE missing emergency flag refused'
$mutated = $source.Replace('job->stream_chunks.push_back(std::move(terminal_sse));',
                            'job->stream_chunks.push_back(std::string("unfinished"));')
Assert-Contract (-not (Has-Atomic-Terminal-Frames $mutated)) 'NEGATIVE broken terminal commit refused'
$mutated = $source.Replace('sink.write(kTerminalTransportFailureSse,',
                            'sink.write(nullptr,')
Assert-Contract (-not (Has-Allocation-Free-Sse-Fallback $mutated)) 'NEGATIVE missing emergency SSE refused'
Write-Output "PASS SOURCE_CONTRACT_ONLY checks=$checks (no dynamic HTTP/TSan coverage)"
