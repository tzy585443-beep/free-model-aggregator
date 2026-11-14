// SPDX-License-Identifier: MIT
// Provider catalog probes and the document registered with the env.
//
// base_url is the brief's own path, never the path joined to MAKEFUN_ENV_URL.
// The env origin changes between a live pass and its replay, and the catalog
// is part of the state export the replay must reproduce.

import {
  authHeaders,
  isRecord,
  joinBase,
  stripModelPrefix,
  type Json,
  type ProviderEntry,
  type Wire,
} from './brief.ts';

export interface ListedModel {
  id: string;
  displayName: string;
  /** Null when the listing did not say whether the model is free. */
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogDocument {
  providers: CatalogProvider[];
}

export function providerHeaders(provider: ProviderEntry, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    accept: stream ? 'text/event-stream' : 'application/json',
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

export function callUrl(envUrl: string, provider: ProviderEntry, modelId: string, stream: boolean): string {
  const root = joinBase(envUrl, provider.base);
  if (provider.wire === 'anthropic') return `${root}/v1/messages`;
  if (provider.wire === 'gemini') {
    const id = encodeURIComponent(stripModelPrefix(modelId));
    const method = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    return `${root}/v1beta/models/${id}:${method}`;
  }
  return `${root}/v1/chat/completions`;
}

/** GET the provider's model list in its own wire. Failures stay out of the catalog. */
export async function fetchCatalog(envUrl: string, provider: ProviderEntry, timeoutMs: number): Promise<Probe> {
  if (envUrl === '') {
    return { provider, ok: false, status: 0, models: [], error: 'MAKEFUN_ENV_URL is not set' };
  }
  const root = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${root}/v1beta/models` : `${root}/v1/models`;
  try {
    const response = await fetch(url, {
      headers: providerHeaders(provider, false),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      return { provider, ok: false, status: response.status, models: [], error: `GET ${url} answered HTTP ${response.status}` };
    }
    const body: unknown = await response.json();
    return { provider, ok: true, status: response.status, models: modelsFromListing(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * Catalog built from listings alone. A listing that hides its price leaves
 * `free` false until a later pass fills the gap from the sealed quality table.
 */
export function catalogFromProbes(probes: readonly Probe[]): CatalogDocument {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const ordered = [...probe.models].sort((a, b) => compareId(a.id, b.id));
    for (const model of ordered) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      models.push({
        id: model.id,
        free: model.free === true,
        context_length: model.contextLength,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

function compareId(a: string, b: string): number {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

function modelsFromListing(wire: Wire, body: unknown): ListedModel[] {
  if (Array.isArray(body)) return body.flatMap((value) => listedOrEmpty(wire, value));
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((value) => listedOrEmpty(wire, value));
}

function listedOrEmpty(wire: Wire, value: unknown): ListedModel[] {
  const model = listedModel(wire, value);
  return model === null ? [] : [model];
}

function listedModel(wire: Wire, value: unknown): ListedModel | null {
  if (!isRecord(value)) return null;
  const rawId = wire === 'gemini' ? asString(value.name) || asString(value.id) : asString(value.id) || asString(value.name);
  const id = wire === 'gemini' ? stripModelPrefix(rawId) : rawId;
  if (id === '') return null;
  const caps = isRecord(value.capabilities) ? value.capabilities : null;
  return {
    id,
    displayName: asString(value.display_name) || asString(value.displayName) || id,
    free: freeFlag(value),
    contextLength:
      finite(value.context_length) ??
      finite(value.context_window) ??
      finite(value.inputTokenLimit) ??
      finite(value.max_input_tokens) ??
      finite(value.max_context),
    tools: flag(value.tools) ?? flag(caps?.tools) ?? feature(value, 'tools'),
    vision: flag(value.vision) ?? flag(caps?.vision) ?? feature(value, 'vision'),
    json: flag(value.json) ?? flag(caps?.json) ?? feature(value, 'json'),
  };
}

function freeFlag(value: Json): boolean | null {
  if (typeof value.free === 'boolean') return value.free;
  const tier = asString(value.tier).toLowerCase();
  if (tier === 'free') return true;
  if (tier === 'paid' || tier === 'premium') return false;
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const numbers = Object.values(pricing).flatMap((item) => {
    if (typeof item === 'number') return [item];
    if (typeof item === 'string' && item.trim() !== '') return [Number(item)];
    return [];
  });
  if (numbers.length === 0) return null;
  return numbers.every((item) => Number.isFinite(item) && item === 0);
}

function feature(value: Json, name: string): boolean | null {
  const supported = value.supported_features ?? value.features;
  if (!Array.isArray(supported)) return null;
  const hit = supported.some((item) => asString(item).toLowerCase() === name);
  return hit ? true : null;
}

function flag(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

export function isJson(value: unknown): value is Json {
  return isRecord(value);
}
