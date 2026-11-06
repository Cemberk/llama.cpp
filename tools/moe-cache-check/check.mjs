#!/usr/bin/env node
// Checks the MoE cache on a real MoE model: each configuration generates the
// same tokens at temperature 0 from the same prompt with llama-server, and
// the outputs are compared with the run that keeps every expert on the GPU.
//
//   node tools/moe-cache-check/check.mjs --server build/bin/llama-server --model olmoe.gguf
//   node tools/moe-cache-check/check.mjs --server ... --model ... --probs
//
// Expected: the cache configurations (slru, lfu, next) give the same text as
// the all-GPU run, and with --probs the same log-probabilities to 4 decimals;
// hybrid mode computes the misses on the CPU, so it differs from the all-GPU
// run only as the all-CPU run does (rounding). A second run with the same
// profile should report more hits. A small MoE model fits any machine:
// OLMoE-1B-7B (allenai/OLMoE-1B-7B-0924-Instruct-GGUF, q4_k_m, 4.2 GB).
//
// Options:
//   --server PATH    llama-server to test (required)
//   --model PATH     a MoE GGUF (required)
//   --cache-mib N    MoE cache size (default 256: smaller than one token's experts on OLMoE)
//   --out DIR        where the trace and profile go (default: a new temporary folder)
//   --probs          compare log-probabilities of the first 8 tokens instead of texts
//   --tokens N       tokens to generate (default 96)

import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

function option(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
const server = option('server');
const model = option('model');
if (!server || !model) {
  console.error('usage: check.mjs --server PATH --model PATH [--cache-mib N] [--out DIR] [--probs] [--tokens N]');
  process.exit(2);
}
const cacheMib = option('cache-mib', '256');
const outDir = option('out') ?? mkdtempSync(path.join(os.tmpdir(), 'moe-cache-check-'));
const probsMode = process.argv.includes('--probs');
const nPredict = probsMode ? 8 : Number(option('tokens', '96'));
const PROMPT =
  'Write a Python function that parses a CSV file of orders, groups them by customer, ' +
  'and returns the three customers with the highest total. Explain each step.';

function freePort() {
  return new Promise((resolve) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

async function waitHealthy(port, child) {
  for (let i = 0; i < 1200; i++) {
    if (child.exitCode !== null) throw new Error(`llama-server exited with ${child.exitCode}`);
    try {
      if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) return;
    } catch {
      // Not up yet.
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('llama-server did not answer in 10 minutes');
}

function cacheMetrics(text) {
  const m = {};
  for (const line of text.split('\n')) {
    const match = /^llamacpp:(moe_cache_[a-z_]+)\s+(\S+)$/.exec(line.trim());
    if (match) m[match[1]] = Number(match[2]);
  }
  return m;
}

const complete = (port, body) =>
  fetch(`http://127.0.0.1:${port}/completion`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ temperature: 0, top_k: 1, cache_prompt: false, ...body }),
  }).then((r) => r.json());

async function run(name, extra, { waitForProfileSave = false } = {}) {
  const port = await freePort();
  const args = [
    '-m', model, '--host', '127.0.0.1', '--port', String(port),
    '-c', '4096', '--parallel', '1', '--metrics', '--no-webui', '-ngl', '99',
    '--temp', '0', '--seed', '1', ...extra,
  ];
  const child = spawn(server, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let log = '';
  child.stdout.on('data', (d) => (log += d));
  child.stderr.on('data', (d) => (log += d));
  try {
    await waitHealthy(port, child);
    const body = await complete(port, { prompt: PROMPT, n_predict: nPredict, ...(probsMode ? { n_probs: 5 } : {}) });
    if (waitForProfileSave) {
      // The profile is saved every 30 s; a short request after that triggers the save.
      await new Promise((r) => setTimeout(r, 31_000));
      await complete(port, { prompt: 'Hi', n_predict: 4 });
    }
    const metrics = cacheMetrics(await (await fetch(`http://127.0.0.1:${port}/metrics`)).text());
    const probs = (body.completion_probabilities ?? []).map((step) =>
      Object.fromEntries((step.top_logprobs ?? []).map((p) => [p.token, p.logprob])),
    );
    return { name, content: body.content ?? '', gen_tps: body.timings?.predicted_per_second, metrics, probs };
  } catch (e) {
    return { name, error: String(e), log: log.slice(-2000) };
  } finally {
    child.kill();
    await new Promise((r) => child.once('exit', r));
  }
}

const commonPrefix = (a, b) => {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
};

/** Largest difference in log-probability of the candidates two runs share, per step. */
function probDiffs(a, b) {
  return a.probs.map((step, i) => {
    const other = b.probs[i] ?? {};
    let worst = 0;
    for (const [tok, lp] of Object.entries(step)) if (tok in other) worst = Math.max(worst, Math.abs(lp - other[tok]));
    return worst;
  });
}

const profile = path.join(outDir, 'profile.txt');
const trace = path.join(outDir, 'trace.txt');
for (const f of [profile, trace]) if (existsSync(f)) rmSync(f);
const cpuExperts = ['--n-cpu-moe', '999'];
const cache = [...cpuExperts, '--moe-cache-mib', cacheMib];
const configs = probsMode
  ? [
      ['gpu', []],
      ['cpu-experts', cpuExperts],
      ['cache-next', [...cache, '--moe-cache-policy', 'next']],
      ['hybrid', [...cache, '--moe-cache-hybrid']],
    ]
  : [
      ['gpu', []],
      ['cpu-experts', cpuExperts],
      ['cache-slru', [...cache, '--moe-cache-policy', 'slru', '--moe-cache-trace', trace]],
      ['cache-lfu', [...cache, '--moe-cache-policy', 'lfu']],
      ['cache-next', [...cache, '--moe-cache-policy', 'next']],
      ['hybrid', [...cache, '--moe-cache-hybrid']],
      ['profile-1', [...cache, '--moe-cache-hybrid', '--moe-cache-profile', profile], { waitForProfileSave: true }],
      ['profile-2', [...cache, '--moe-cache-hybrid', '--moe-cache-profile', profile]],
    ];

console.log(`server ${server}\nmodel ${model}\ncache ${cacheMib} MiB, output in ${outDir}\n`);
const results = [];
let failed = false;
for (const [name, extra, opts] of configs) {
  const result = await run(name, extra, opts);
  results.push(result);
  const ref = results[0];
  if (result.error) {
    failed = true;
    console.log(`${name.padEnd(12)} | ERROR ${result.error}\n${result.log}`);
    continue;
  }
  const m = result.metrics;
  const hits = m.moe_cache_hits_total;
  const misses = m.moe_cache_misses_total;
  const parts = [name.padEnd(12)];
  if (probsMode) {
    parts.push(`max |logprob - gpu| per step: ${probDiffs(result, ref).map((d) => d.toFixed(4)).join(' ')}`);
  } else {
    const same = result.content === ref.content;
    parts.push(same ? 'same text as gpu' : `differs from gpu after ${commonPrefix(result.content, ref.content)} of ${ref.content.length} chars`);
    if (result.gen_tps) parts.push(`gen ${result.gen_tps.toFixed(1)} tok/s`);
  }
  if (hits !== undefined) {
    parts.push(`hits ${hits} misses ${misses} (${((100 * hits) / Math.max(1, hits + misses)).toFixed(1)}%)`);
  }
  console.log(parts.join(' | '));
  // The cache configurations must match the all-GPU run exactly.
  if (name.startsWith('cache-')) {
    const exact = probsMode
      ? probDiffs(result, ref).every((d) => d < 1e-4)
      : result.content === ref.content;
    if (!exact) {
      failed = true;
      console.log(`  FAIL: ${name} should match the all-GPU run exactly`);
    }
  }
}

if (probsMode) {
  const cpu = results.find((r) => r.name === 'cpu-experts');
  const hybrid = results.find((r) => r.name === 'hybrid');
  if (cpu && hybrid && !cpu.error && !hybrid.error) {
    console.log(`hybrid       | max |logprob - cpu-experts| per step: ${probDiffs(hybrid, cpu).map((d) => d.toFixed(4)).join(' ')}`);
  }
} else {
  const p1 = results.find((r) => r.name === 'profile-1');
  const p2 = results.find((r) => r.name === 'profile-2');
  const rate = (r) => r.metrics.moe_cache_hits_total / Math.max(1, r.metrics.moe_cache_hits_total + r.metrics.moe_cache_misses_total);
  if (p1 && p2 && !p1.error && !p2.error) {
    console.log(`\nwarm start: hit rate ${(100 * rate(p1)).toFixed(1)}% -> ${(100 * rate(p2)).toFixed(1)}% with the saved profile`);
  }
  if (existsSync(profile)) console.log(`profile: ${readFileSync(profile, 'utf8').split('\n').length - 2} experts (${profile})`);
  else {
    failed = true;
    console.log('FAIL: the profile was not saved');
  }
  if (existsSync(trace)) {
    console.log(`trace: ${readFileSync(trace, 'utf8').split('\n').length - 1} lines (${trace})`);
    console.log(`  replay: python tools/moe-sim/moe_cache_sim.py "${trace}" --slots <slots of the cache>`);
  }
}
process.exit(failed ? 1 : 0);
