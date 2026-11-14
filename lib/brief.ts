// SPDX-License-Identifier: MIT
// Brief parser for the free-model aggregator trial.
//
// MAKEFUN_PROMPT_FILE is prose wrapped around three JSON documents, found by
// their `schema` member. Nothing in here reads the clock or draws randomness.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export const isRecord = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Path under MAKEFUN_ENV_URL, or an absolute URL when the brief gives one. */
  base: string;
  keyEnv: string;
  key: string;
  /** Per-trial call budget, when the brief states one. Null when it does not. */
  budget: number | null;
}

export interface Rating {
  provider: string;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingTable {
  rows: Rating[];
}

export interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number | null;
}

export interface WorkItem {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingTable | null;
  workload: WorkItem[] | null;
  error: string | null;
}

export function emptyBrief(error: string | null = null): Brief {
  return { providers: [], ratings: null, workload: null, error };
}

/** Join a provider base onto the env origin. Absolute bases are kept as given. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return origin;
  const path = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return `${origin}${path.replace(/\/+$/, '')}`;
}

/** Headers that carry a provider key. `none` and an empty key send nothing. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (auth === 'none' || key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

export function findRating(table: RatingTable | null, providerId: string, modelId: string): Rating | null {
  if (table === null) return null;
  const bare = stripModelPrefix(modelId);
  for (const row of table.rows) {
    if (row.provider !== providerId) continue;
    if (row.model === modelId || stripModelPrefix(row.model) === bare) return row;
  }
  return null;
}

export function stripModelPrefix(modelId: string): string {
  return modelId.startsWith('models/') ? modelId.slice('models/'.length) : modelId;
}

/** A capability the request does not ask for is not required. Unknown never satisfies a demand. */
export function allows(
  requires: Requirement,
  caps: { tools: boolean | null; vision: boolean | null; json: boolean | null; contextLength: number | null },
): boolean {
  if (requires.tools && caps.tools !== true) return false;
  if (requires.vision && caps.vision !== true) return false;
  if (requires.json && caps.json !== true) return false;
  if (requires.minContext !== null && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/** Requirements inferred from a client body when the manifest has no row left. */
export function sniffRequirement(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    vision = vision || contentHasImage(message.content);
  }
  vision = vision || contentHasImage(body.system);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema' || isRecord(body.output_format),
    minContext: null,
  };
}

function contentHasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'image' || block.type === 'image_url' || block.type === 'input_image') return true;
  }
  return false;
}

export function loadBrief(path: string): Brief {
  if (path === '') return emptyBrief('MAKEFUN_PROMPT_FILE is not set');
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return emptyBrief(`cannot read ${path}: ${(err as Error).message}`);
  }
  const docs = documentsWithSchema(text);
  const directory = docs.find((doc) => doc.schema === PROVIDERS_SCHEMA && Array.isArray(doc.providers));
  if (directory === undefined || !Array.isArray(directory.providers)) {
    return emptyBrief(`the brief at ${path} holds no ${PROVIDERS_SCHEMA} document`);
  }
  const providers = directory.providers.flatMap((value) => {
    const provider = providerFrom(value);
    return provider === null ? [] : [provider];
  });
  const ratingsDoc = docs.find((doc) => doc.schema === RATINGS_SCHEMA && Array.isArray(doc.ratings));
  const ratings: RatingTable | null =
    ratingsDoc === undefined || !Array.isArray(ratingsDoc.ratings)
      ? null
      : {
          rows: ratingsDoc.ratings.flatMap((value) => {
            const row = ratingFrom(value);
            return row === null ? [] : [row];
          }),
        };
  const workloadDoc = docs.find((doc) => doc.schema === WORKLOAD_SCHEMA && Array.isArray(doc.manifest));
  const workload: WorkItem[] | null =
    workloadDoc === undefined || !Array.isArray(workloadDoc.manifest)
      ? null
      : workloadDoc.manifest.flatMap((value) => {
          const item = workItemFrom(value);
          return item === null ? [] : [item];
        });
  return { providers, ratings, workload, error: null };
}

function documentsWithSchema(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = matchingBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // A brace inside prose or a string. Keep scanning.
    }
  }
  return found;
}

function matchingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function wireFrom(value: unknown): Wire | null {
  const wire = asString(value).toLowerCase();
  if (wire === 'openai' || wire === 'anthropic' || wire === 'gemini') return wire;
  return null;
}

function authFrom(value: unknown): AuthScheme {
  const auth = asString(value).toLowerCase();
  if (auth === 'x-api-key' || auth === 'api-key' || auth === 'apikey') return 'x-api-key';
  if (auth === 'x-goog-api-key' || auth === 'google' || auth === 'goog') return 'x-goog-api-key';
  if (auth === 'none') return 'none';
  return 'bearer';
}

function budgetFrom(value: Json): number | null {
  for (const key of ['budget', 'call_budget', 'quota', 'max_calls', 'free_calls']) {
    const parsed = finiteOrNull(value[key]);
    if (parsed !== null && parsed >= 0) return parsed;
  }
  return null;
}

function providerFrom(value: unknown): ProviderEntry | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  const wire = wireFrom(value.wire);
  if (id === '' || wire === null) return null;
  const base =
    asString(value.path) || asString(value.base_path) || asString(value.base_url) || asString(value.base);
  return {
    id,
    name: asString(value.name) || id,
    wire,
    auth: authFrom(value.auth),
    base,
    keyEnv: asString(value.key_env),
    key: asString(value.key) || asString(value.api_key),
    budget: budgetFrom(value),
  };
}

function ratingFrom(value: unknown): Rating | null {
  if (!isRecord(value)) return null;
  const model = asString(value.model);
  const provider = asString(value.provider);
  if (model === '' || provider === '') return null;
  return {
    provider,
    model,
    ratingMilli: finiteOrNull(value.rating_milli) ?? 0,
    contextLength: finiteOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function requirementFrom(value: unknown): Requirement {
  const source = isRecord(value) ? value : {};
  const minContext = finiteOrNull(source.min_context) ?? finiteOrNull(source.min_context_length);
  return {
    tools: source.tools === true,
    vision: source.vision === true,
    json: source.json === true,
    minContext,
  };
}

function workItemFrom(value: unknown): WorkItem | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const weight = finiteOrNull(value.weight);
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: weight === null ? 1 : weight,
    requires: requirementFrom(value.requires),
  };
}
