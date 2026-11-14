// SPDX-License-Identifier: MIT
// Free-model aggregator. Round 1: register, list, and serve both client wires
// through OpenAI-wire providers. Routing is first-fit; failover and the other
// upstream wires come in later passes.

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
  canonFromChatResponse,
  canonFromChatStream,
  canonFromMessages,
  chatFromCanon,
  messagesResult,
  messagesStream,
  openAIError,
  openAIModelList,
  type PublicModel,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

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
  catalog = catalogFromProbes(probes);
}

interface Route {
  probe: Probe;
  model: ListedModel;
}

function capsOf(model: ListedModel, row: Rating | null): { tools: boolean | null; vision: boolean | null; json: boolean | null; contextLength: number | null } {
  if (row !== null) {
    return { tools: row.tools, vision: row.vision, json: row.json, contextLength: row.contextLength ?? model.contextLength };
  }
  return { tools: model.tools, vision: model.vision, json: model.json, contextLength: model.contextLength };
}

function modelIsFree(model: ListedModel, row: Rating | null): boolean {
  if (row !== null) return row.free;
  return model.free === true;
}

/**
 * First free model in brief order whose id sorts first on that provider.
 * This ignores sealed ratings on purpose; the next pass ranks by rating_milli.
 */
function selectRoute(requires: Requirement): Route | null {
  for (const probe of probes) {
    if (!probe.ok) continue;
    const models = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of models) {
      const row = findRating(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!allows(requires, capsOf(model, row))) continue;
      return { probe, model };
    }
  }
  return null;
}

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

async function callOpenAI(route: Route, body: Json, stream: boolean): Promise<Response> {
  const headers = providerHeaders(route.probe.provider, stream);
  return fetch(callUrl(ENV_URL, route.probe.provider, route.model.id, stream), {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
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

async function postChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const route = selectRoute(takeRequirement(body));
  if (route === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (route.probe.provider.wire !== 'openai') {
    return sendError(res, 'openai', 501, 'not_implemented', `the ${route.probe.provider.wire} upstream wire is not served yet`);
  }
  const stream = body.stream === true;
  const upstreamBody: Json = { ...body, model: route.model.id };
  if (stream) {
    const previous = isRecord(upstreamBody.stream_options) ? upstreamBody.stream_options : {};
    upstreamBody.stream_options = { ...previous, include_usage: true };
  }
  let upstream: Response;
  try {
    upstream = await callOpenAI(route, upstreamBody, stream);
  } catch (err) {
    return sendError(res, 'openai', 502, 'upstream_unreachable', (err as Error).message);
  }
  if (stream && upstream.ok && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(upstream.body, res);
    return;
  }
  const text = await upstream.text();
  res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
  res.end(text);
}

async function postMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const route = selectRoute(takeRequirement(body));
  if (route === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if (route.probe.provider.wire !== 'openai') {
    return sendError(res, 'anthropic', 501, 'api_error', `the ${route.probe.provider.wire} upstream wire is not served yet`);
  }
  const stream = body.stream === true;
  const upstreamBody = chatFromCanon(canonFromMessages(body), route.model.id);
  let upstream: Response;
  try {
    upstream = await callOpenAI(route, upstreamBody, stream);
  } catch (err) {
    return sendError(res, 'anthropic', 502, 'api_error', (err as Error).message);
  }
  if (!upstream.ok) {
    const text = await upstream.text();
    sendJson(res, upstream.status, anthropicError('api_error', `the provider answered HTTP ${upstream.status}: ${text.slice(0, 500)}`));
    return;
  }
  const reported = route.model.id;
  if (stream && upstream.body !== null) {
    const raw = await upstream.text();
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(messagesStream(canonFromChatStream(raw), reported));
    return;
  }
  const answer: unknown = await upstream.json();
  const canon = isRecord(answer)
    ? canonFromChatResponse(answer)
    : { id: 'aggregate', text: '', toolCalls: [], stop: 'end_turn' as const, stopSequence: null, inputTokens: 0, outputTokens: 0 };
  sendJson(res, 200, messagesResult(canon, reported));
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
      if (path === '/v1/chat/completions') await postChat(req, res);
      else await postMessages(req, res);
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
