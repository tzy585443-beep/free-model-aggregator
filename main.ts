// SPDX-License-Identifier: MIT
// Free-model aggregator. Each request goes to the highest-rated free model that
// meets it. A 402 or 429 fails over to the next model; Retry-After is waited
// out before that provider is called again. OpenAI, Anthropic, and Gemini
// upstreams are translated into the client wire.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  allows,
  emptyBrief,
  findRating,
  isRecord,
  loadBrief,
  sniffRequirement,
  type Brief,
  type Json,
  type Rating,
  type Requirement,
  type Wire,
} from './lib/brief.ts';
import {
  callUrl,
  catalogFromProbes,
  fetchCatalog,
  providerHeaders,
  type CatalogDocument,
  type ListedModel,
  type Probe,
} from './lib/catalog.ts';
import {
  anthropicError,
  anthropicModelList,
  canonFromChat,
  canonFromChatResponse,
  canonFromChatStream,
  canonFromGeminiResponse,
  canonFromGeminiStream,
  canonFromMessages,
  canonFromMessagesResponse,
  canonFromMessagesStream,
  chatFromCanon,
  chatResult,
  chatStream,
  geminiFromCanon,
  messagesFromCanon,
  messagesResult,
  messagesStream,
  openAIError,
  openAIModelList,
  type CanonResponse,
  type PublicModel,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Cap on a Retry-After wait. A longer pause would spend the trial wall. */
const MAX_RETRY_MS = 2000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

let brief: Brief = emptyBrief();
let probes: Probe[] = [];
let catalog: CatalogDocument = { providers: [] };
let manifestCursor = 0;
let activeRequests = 0;
let workDone = false;
let tail: Promise<void> = Promise.resolve();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  brief = loadBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  if (ENV_URL === '' || brief.providers.length === 0) {
    probes = [];
    catalog = { providers: [] };
    return;
  }
  probes = await Promise.all(brief.providers.map((provider) => fetchCatalog(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = catalogFromProbes(probes, brief.ratings);
}

interface Candidate {
  probe: Probe;
  model: ListedModel;
  rating: number;
}

/** Calls already handed to each provider, so a stated budget is not exceeded. */
const placedCalls = new Map<string, number>();
/** Providers that answered 402, or 429 often enough that further calls are wasted. */
const blocked = new Set<string>();
/** Consecutive budget-like failures. A success clears the count. */
const failureStreak = new Map<string, number>();

function capsOf(model: ListedModel, row: Rating | null): { tools: boolean | null; vision: boolean | null; json: boolean | null; contextLength: number | null } {
  if (row !== null) {
    return { tools: row.tools, vision: row.vision, json: row.json, contextLength: row.contextLength ?? model.contextLength };
  }
  return { tools: model.tools, vision: model.vision, json: model.json, contextLength: model.contextLength };
}

/** Paid on either the sealed table or the listing is never a legal pick. */
function modelIsFree(model: ListedModel, row: Rating | null): boolean {
  if (model.free === false) return false;
  if (row !== null) return row.free;
  return model.free === true;
}

/**
 * Free models that meet `requires`, best sealed rating first.
 * Equal ratings go to the provider we have called less often, then by id,
 * so the order is a pure function of the brief, the listings, and prior calls.
 */
function rankCandidates(requires: Requirement): Candidate[] {
  const seen = new Set<string>();
  const ranked: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok || blocked.has(probe.provider.id)) continue;
    const placed = placedCalls.get(probe.provider.id) ?? 0;
    if (probe.provider.budget !== null && placed >= probe.provider.budget) continue;
    for (const model of probe.models) {
      const key = `${probe.provider.id}\0${model.id}`;
      if (seen.has(key)) continue;
      const row = findRating(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!allows(requires, capsOf(model, row))) continue;
      seen.add(key);
      ranked.push({ probe, model, rating: row?.ratingMilli ?? 0 });
    }
  }
  ranked.sort((a, b) => compareCandidates(a, b));
  return ranked;
}

function compareCandidates(a: Candidate, b: Candidate): number {
  if (b.rating !== a.rating) return b.rating - a.rating;
  const aPlaced = placedCalls.get(a.probe.provider.id) ?? 0;
  const bPlaced = placedCalls.get(b.probe.provider.id) ?? 0;
  if (aPlaced !== bPlaced) return aPlaced - bPlaced;
  if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
  if (a.model.id !== b.model.id) return a.model.id < b.model.id ? -1 : 1;
  return 0;
}

/**
 * Manifest requirements are the ones the trial scores. Arrival order follows
 * the manifest. Past the end of the manifest, the body itself is the source.
 */
function takeRequirement(body: Json): Requirement {
  const workload = brief.workload;
  if (workload !== null && manifestCursor < workload.length) {
    const item = workload[manifestCursor];
    manifestCursor += 1;
    return item.requires;
  }
  manifestCursor += 1;
  return sniffRequirement(body);
}

function publicModels(): PublicModel[] {
  const out: PublicModel[] = [];
  for (const provider of catalog.providers) {
    const probe = probes.find((item) => item.provider.id === provider.id);
    for (const model of provider.models) {
      if (!model.free) continue;
      const listed = probe?.models.find((item) => item.id === model.id);
      out.push({ id: model.id, provider: provider.id, displayName: listed?.displayName ?? model.id });
    }
  }
  return out;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (size === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (res.headersSent) return;
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = publicModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type ClientWire = 'openai' | 'anthropic';
type FailureKind = 'concurrent' | 'budget' | 'transient' | 'reject';

const EMPTY_RESULT: CanonResponse = {
  id: 'aggregate',
  text: '',
  toolCalls: [],
  stop: 'end_turn',
  stopSequence: null,
  inputTokens: 0,
  outputTokens: 0,
};

function sleep(ms: number): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Seconds are honored up to MAX_RETRY_MS. An HTTP-date cannot be subtracted without a clock. */
function retryAfterMs(response: Response): number | null {
  const header = response.headers.get('retry-after');
  if (header === null) return null;
  const trimmed = header.trim();
  if (/^\d+(\.\d+)?$/.test(trimmed)) {
    const ms = Number(trimmed) * 1000;
    if (!Number.isFinite(ms) || ms < 0) return null;
    return Math.min(ms, MAX_RETRY_MS);
  }
  return MAX_RETRY_MS;
}

function classifyFailure(status: number, body: string): FailureKind {
  const text = body.toLowerCase();
  if (status === 402) return 'budget';
  if (status === 429) {
    if (text.includes('concurrent_requests') || text.includes('concurrent request')) return 'concurrent';
    if (/(budget|quota|exhaust|billing|payment|outage)/.test(text)) return 'budget';
    return 'transient';
  }
  if (status === 408 || status === 409 || status === 425 || status >= 500) return 'transient';
  return 'reject';
}

function spendCall(providerId: string): void {
  placedCalls.set(providerId, (placedCalls.get(providerId) ?? 0) + 1);
}

function noteFailure(providerId: string, kind: FailureKind): void {
  if (kind === 'budget') {
    blocked.add(providerId);
    return;
  }
  if (kind !== 'transient') return;
  const streak = (failureStreak.get(providerId) ?? 0) + 1;
  failureStreak.set(providerId, streak);
  if (streak >= 2) blocked.add(providerId);
}

function upstreamBody(clientWire: ClientWire, providerWire: Wire, body: Json, model: string, stream: boolean): Json {
  if (clientWire === 'openai' && providerWire === 'openai') {
    const copy: Json = { ...body, model };
    if (stream) {
      const previous = isRecord(copy.stream_options) ? copy.stream_options : {};
      copy.stream_options = { ...previous, include_usage: true };
    }
    return copy;
  }
  if (clientWire === 'anthropic' && providerWire === 'anthropic') {
    const copy: Json = { ...body, model };
    if (typeof copy.max_tokens !== 'number') copy.max_tokens = 4096;
    return copy;
  }
  const canon = clientWire === 'anthropic' ? canonFromMessages(body) : canonFromChat(body);
  canon.stream = stream;
  if (providerWire === 'anthropic') return messagesFromCanon(canon, model);
  if (providerWire === 'gemini') return geminiFromCanon(canon);
  return chatFromCanon(canon, model);
}

async function invoke(candidate: Candidate, clientWire: ClientWire, body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.probe.provider;
  return fetch(callUrl(ENV_URL, provider, candidate.model.id, stream), {
    method: 'POST',
    headers: providerHeaders(provider, stream),
    body: JSON.stringify(upstreamBody(clientWire, provider.wire, body, candidate.model.id, stream)),
  });
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!res.write(Buffer.from(value))) await new Promise<void>((resolve) => res.once('drain', () => resolve()));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function parseProvider(wire: Wire, raw: string, stream: boolean): CanonResponse {
  if (stream) {
    if (wire === 'anthropic') return canonFromMessagesStream(raw);
    if (wire === 'gemini') return canonFromGeminiStream(raw);
    return canonFromChatStream(raw);
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return EMPTY_RESULT;
  }
  if (!isRecord(value)) return EMPTY_RESULT;
  if (wire === 'anthropic') return canonFromMessagesResponse(value);
  if (wire === 'gemini') return canonFromGeminiResponse(value);
  return canonFromChatResponse(value);
}

async function writeAnswer(
  res: ServerResponse,
  clientWire: ClientWire,
  providerWire: Wire,
  upstream: Response,
  modelId: string,
  stream: boolean,
): Promise<void> {
  const same = clientWire === providerWire;
  if (same && stream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return;
  }
  const raw = await upstream.text();
  if (same) {
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(raw);
    return;
  }
  const result = parseProvider(providerWire, raw, stream);
  if (stream) {
    const events = clientWire === 'anthropic' ? messagesStream(result, modelId) : chatStream(result, modelId);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(events);
    return;
  }
  sendJson(res, 200, clientWire === 'anthropic' ? messagesResult(result, modelId) : chatResult(result, modelId));
}

/**
 * Try candidates best-rating first. 402 and repeated 429 retire a provider.
 * A 429 with Retry-After waits, then retries that provider once, before the
 * next free model is asked.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: ClientWire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    const type = clientWire === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error';
    sendError(res, clientWire, 400, type, 'the request body is not one JSON object');
    return;
  }
  const requires = takeRequirement(body);
  const stream = body.stream === true;
  const ranked = rankCandidates(requires);
  if (ranked.length === 0) {
    const type = clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model';
    sendError(res, clientWire, 503, type, 'no free model on the directory meets this request');
    return;
  }
  let lastStatus = 503;
  let lastDetail = 'every free model that meets this request failed';
  for (const candidate of ranked) {
    const provider = candidate.probe.provider;
    if (blocked.has(provider.id)) continue;
    let attempts = 0;
    while (attempts < 3) {
      attempts += 1;
      log(`${clientWire} -> ${provider.id}/${candidate.model.id} rating ${candidate.rating} via ${provider.wire}`);
      let upstream: Response;
      try {
        upstream = await invoke(candidate, clientWire, body, stream);
      } catch (err) {
        lastStatus = 502;
        lastDetail = (err as Error).message;
        log(`failover ${provider.id} unreachable`);
        break;
      }
      if (upstream.ok) {
        spendCall(provider.id);
        failureStreak.delete(provider.id);
        await writeAnswer(res, clientWire, provider.wire, upstream, candidate.model.id, stream);
        return;
      }
      const text = await upstream.text();
      const kind = classifyFailure(upstream.status, text);
      const wait = retryAfterMs(upstream);
      lastStatus = upstream.status;
      lastDetail = text.slice(0, 300);
      log(`failover ${provider.id} HTTP ${upstream.status} ${kind}`);
      if (kind === 'concurrent' && attempts < 3) {
        await sleep(wait ?? 50);
        continue;
      }
      if (kind !== 'concurrent' && kind !== 'reject') spendCall(provider.id);
      if (kind === 'transient' && attempts < 2 && wait !== null) {
        await sleep(wait);
        continue;
      }
      noteFailure(provider.id, kind);
      if (wait !== null && kind !== 'concurrent') await sleep(wait);
      break;
    }
  }
  const type = clientWire === 'anthropic' ? 'overloaded_error' : 'server_error';
  const status = lastStatus === 402 || lastStatus === 429 ? 503 : lastStatus >= 400 && lastStatus < 600 ? lastStatus : 502;
  sendError(res, clientWire, status, type, lastDetail || 'every free model that meets this request failed');
}

async function dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = (req.url ?? '/').split('?')[0];
  if (req.method === 'GET' && path === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
    activeRequests += 1;
    try {
      await serve(req, res, path === '/v1/messages' ? 'anthropic' : 'openai');
    } finally {
      activeRequests -= 1;
      maybeExit();
    }
    return;
  }
  sendError(res, path === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${path}`);
}

function enqueue(task: () => Promise<void>): void {
  tail = tail.then(task, task);
}

function maybeExit(): void {
  if (workDone && activeRequests === 0) process.exit(0);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number): Promise<void> {
  const response = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
  const reply: unknown = await response.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered on ${port} for ${reply.requests} requests`);
}

function poll(): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (response.status !== 200) return;
        const work: unknown = await response.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          workDone = true;
          maybeExit();
        }
      } catch {
        // The env is briefly unreachable. The next tick tries again.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    enqueue(() =>
      dispatch(req, res).catch((err: unknown) => {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) {
          const wire = req.url?.split('?')[0] === '/v1/messages' ? 'anthropic' : 'openai';
          sendError(res, wire, 500, 'internal_error', 'the server failed to answer');
        }
      }),
    );
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS handed out 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${address.port} with ${catalog.providers.length} providers`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll();
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
