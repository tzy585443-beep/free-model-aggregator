// SPDX-License-Identifier: MIT
// Translations between the OpenAI Chat Completions wire, the Anthropic Messages
// wire, and the Gemini generateContent wire. Pure functions of their inputs:
// fixed timestamps, no clock, no randomness.

import { isRecord, type Json } from './brief.ts';

/** Fixed epoch second for synthesized OpenAI objects. */
const CREATED = 1767225600;
/** Fixed instant for synthesized Anthropic model listings. */
const CREATED_AT = '2026-01-01T00:00:00Z';

export type StopReason = 'end_turn' | 'max_tokens' | 'tool_use' | 'refusal' | 'stop_sequence';
export type ToolChoice = 'auto' | 'none' | 'required' | { name: string };

export interface ImagePart {
  mediaType: string;
  data: string | null;
  url: string | null;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: ImagePart[];
  toolCalls: ToolCall[];
  toolCallId: string | null;
  toolName: string | null;
}

export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stop: string[];
  tools: CanonTool[];
  toolChoice: ToolChoice | null;
  jsonMode: boolean;
  jsonSchema: Json | null;
  jsonSchemaName: string | null;
  stream: boolean;
}

export interface CanonResponse {
  id: string;
  text: string;
  toolCalls: ToolCall[];
  stop: StopReason;
  stopSequence: string | null;
  inputTokens: number;
  outputTokens: number;
}

export interface PublicModel {
  id: string;
  provider: string;
  displayName: string;
}

export function openAIModelList(models: readonly PublicModel[]): Json {
  return {
    object: 'list',
    data: models.map((model) => ({
      id: model.id,
      object: 'model',
      created: CREATED,
      owned_by: model.provider,
    })),
  };
}

export function anthropicModelList(models: readonly PublicModel[]): Json {
  return {
    data: models.map((model) => ({
      type: 'model',
      id: model.id,
      display_name: model.displayName,
      created_at: CREATED_AT,
    })),
    has_more: false,
  };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function canonFromChat(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const systemParts: string[] = [];
  const names = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system' || role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      const toolCallId = asString(raw.tool_call_id);
      messages.push({
        role: 'tool',
        text: textOf(raw.content),
        images: [],
        toolCalls: [],
        toolCallId,
        toolName: names.get(toolCallId) ?? null,
      });
      continue;
    }
    if (role === 'assistant') {
      const toolCalls = openAIToolCalls(raw.tool_calls);
      for (const call of toolCalls) names.set(call.id, call.name);
      const parts = partsOf(raw.content);
      messages.push({ role: 'assistant', text: parts.text, images: parts.images, toolCalls, toolCallId: null, toolName: null });
      continue;
    }
    const parts = partsOf(raw.content);
    messages.push({ role: 'user', text: parts.text, images: parts.images, toolCalls: [], toolCallId: null, toolName: null });
  }
  return requestFields(body, systemParts.join('\n'), messages);
}

export function canonFromMessages(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const names = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push(plain(assistant ? 'assistant' : 'user', content));
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: ToolCall[] = [];
      let text = '';
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') text = joinText(text, block.text);
        else if (block.type === 'tool_use') {
          const call = toolUseBlock(block);
          names.set(call.id, call.name);
          toolCalls.push(call);
        }
      }
      messages.push({ role: 'assistant', text, images: [], toolCalls, toolCallId: null, toolName: null });
      continue;
    }
    let text = '';
    let images: ImagePart[] = [];
    const flush = (): void => {
      if (text === '' && images.length === 0) return;
      messages.push({ role: 'user', text, images, toolCalls: [], toolCallId: null, toolName: null });
      text = '';
      images = [];
    };
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        flush();
        const toolCallId = asString(block.tool_use_id);
        messages.push({
          role: 'tool',
          text: toolResultText(block.content),
          images: [],
          toolCalls: [],
          toolCallId,
          toolName: names.get(toolCallId) ?? null,
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') text = joinText(text, block.text);
      const image = imageFromAnthropic(block);
      if (image !== null) images.push(image);
    }
    flush();
  }
  const system = textOf(body.system);
  const request = requestFields(body, system, messages);
  if (Array.isArray(body.tools)) request.tools = anthropicTools(body.tools);
  request.toolChoice = anthropicToolChoice(body.tool_choice);
  if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)) request.maxTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences)) request.stop = body.stop_sequences.filter((item): item is string => typeof item === 'string');
  return request;
}

export function chatFromCanon(request: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (request.system !== '') messages.push({ role: 'system', content: request.system });
  for (const message of request.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: message.toolCallId ?? '', content: message.text });
      continue;
    }
    if (message.role === 'assistant') {
      const toolCalls = message.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: call.arguments === '' ? '{}' : call.arguments },
      }));
      messages.push({
        role: 'assistant',
        content: message.text !== '' ? message.text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    messages.push({ role: 'user', content: openAIUserContent(message) });
  }
  const out: Json = { model, messages };
  if (request.maxTokens !== null) out.max_tokens = request.maxTokens;
  if (request.temperature !== null) out.temperature = request.temperature;
  if (request.topP !== null) out.top_p = request.topP;
  if (request.stop.length > 0) out.stop = request.stop;
  if (request.tools.length > 0) {
    out.tools = request.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
    if (request.toolChoice !== null) out.tool_choice = openAIToolChoice(request.toolChoice);
  }
  if (request.jsonSchema !== null) {
    out.response_format = {
      type: 'json_schema',
      json_schema: { name: request.jsonSchemaName ?? 'response', schema: request.jsonSchema },
    };
  } else if (request.jsonMode) out.response_format = { type: 'json_object' };
  if (request.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function messagesFromCanon(request: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of request.messages) {
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    const blocks = anthropicBlocks(message);
    const previous = messages.length > 0 ? messages[messages.length - 1] : null;
    if (previous !== null && previous.role === role && Array.isArray(previous.content)) {
      previous.content = [...previous.content, ...blocks];
      continue;
    }
    messages.push({ role, content: blocks });
  }
  if (messages.length === 0) messages.push({ role: 'user', content: [{ type: 'text', text: '' }] });
  const out: Json = { model, max_tokens: request.maxTokens ?? 4096, messages };
  if (request.system !== '') out.system = request.system;
  if (request.temperature !== null) out.temperature = request.temperature;
  if (request.topP !== null) out.top_p = request.topP;
  if (request.stop.length > 0) out.stop_sequences = request.stop;
  if (request.tools.length > 0) {
    out.tools = request.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
    if (request.toolChoice !== null) out.tool_choice = anthropicChoiceOut(request.toolChoice);
  }
  if (request.jsonSchema !== null) out.output_format = { type: 'json_schema', schema: request.jsonSchema };
  if (request.stream) out.stream = true;
  return out;
}

export function geminiFromCanon(request: CanonRequest): Json {
  const contents: Json[] = [];
  for (const message of request.messages) {
    const role = message.role === 'assistant' ? 'model' : 'user';
    const parts = geminiParts(message);
    const previous = contents.length > 0 ? contents[contents.length - 1] : null;
    if (previous !== null && previous.role === role && Array.isArray(previous.parts)) {
      previous.parts = [...previous.parts, ...parts];
      continue;
    }
    contents.push({ role, parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: request.system || '' }] });
  const generationConfig: Json = {};
  if (request.maxTokens !== null) generationConfig.maxOutputTokens = request.maxTokens;
  if (request.temperature !== null) generationConfig.temperature = request.temperature;
  if (request.topP !== null) generationConfig.topP = request.topP;
  if (request.stop.length > 0) generationConfig.stopSequences = request.stop;
  if (request.jsonMode || request.jsonSchema !== null) generationConfig.responseMimeType = 'application/json';
  if (request.jsonSchema !== null) generationConfig.responseSchema = request.jsonSchema;
  const out: Json = { contents };
  if (request.system !== '') out.systemInstruction = { parts: [{ text: request.system }] };
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  if (request.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: request.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
    out.toolConfig = { functionCallingConfig: geminiToolConfig(request.toolChoice) };
  }
  return out;
}

export function canonFromChatResponse(body: Json): CanonResponse {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const parts = partsOf(message.content);
  const refusal = asString(message.refusal);
  const toolCalls = openAIToolCalls(message.tool_calls);
  let stop = stopFromFinish(choice.finish_reason);
  if (toolCalls.length > 0 && stop === 'end_turn') stop = 'tool_use';
  if (refusal !== '' && parts.text === '') stop = 'refusal';
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: asString(body.id) || 'aggregate',
    text: refusal !== '' && parts.text === '' ? refusal : parts.text,
    toolCalls,
    stop,
    stopSequence: null,
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
  };
}

export function canonFromMessagesResponse(body: Json): CanonResponse {
  let text = '';
  const toolCalls: ToolCall[] = [];
  const content = Array.isArray(body.content) ? body.content : [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text = joinText(text, block.text);
    else if (block.type === 'tool_use') toolCalls.push(toolUseBlock(block));
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop = stopFromAnthropic(body.stop_reason, toolCalls.length > 0);
  return {
    id: asString(body.id) || 'aggregate',
    text,
    toolCalls,
    stop,
    stopSequence: typeof body.stop_sequence === 'string' ? body.stop_sequence : null,
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
  };
}

export function canonFromGeminiResponse(body: Json): CanonResponse {
  return mergeGemini([body], asString(body.responseId) || asString(body.modelVersion) || 'gemini');
}

export function canonFromChatStream(raw: string): CanonResponse {
  let id = '';
  let text = '';
  const tools = new Map<number, ToolCall>();
  let stop: StopReason = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  for (const frame of framesOf(raw)) {
    const payload = frame.data.trim();
    if (payload === '' || payload === '[DONE]') continue;
    const value = parseJson(payload);
    if (!isRecord(value)) continue;
    if (typeof value.id === 'string' && value.id !== '') id = value.id;
    if (isRecord(value.usage)) {
      inputTokens = num(value.usage.prompt_tokens);
      outputTokens = num(value.usage.completion_tokens);
    }
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    else text += textOf(delta.content);
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : tools.size;
      const current = tools.get(index) ?? { id: '', name: '', arguments: '' };
      if (typeof call.id === 'string' && call.id !== '') current.id = call.id;
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') current.name = fn.name;
      if (typeof fn.arguments === 'string') current.arguments += fn.arguments;
      tools.set(index, current);
    }
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) stop = stopFromFinish(choice.finish_reason);
  }
  const toolCalls = [...tools.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, call], index) => ({
      id: call.id !== '' ? call.id : `call_${index}`,
      name: call.name,
      arguments: call.arguments === '' ? '{}' : call.arguments,
    }));
  if (toolCalls.length > 0 && stop === 'end_turn') stop = 'tool_use';
  return { id: id || 'aggregate', text, toolCalls, stop, stopSequence: null, inputTokens, outputTokens };
}

export function canonFromMessagesStream(raw: string): CanonResponse {
  let id = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopReason = 'end_turn';
  let stopSequence: string | null = null;
  const blocks = new Map<number, { kind: 'text' | 'tool'; text: string; id: string; name: string }>();
  for (const frame of framesOf(raw)) {
    const payload = frame.data.trim();
    if (payload === '' || payload === '[DONE]') continue;
    const value = parseJson(payload);
    if (!isRecord(value)) continue;
    const type = asString(value.type) || frame.event || '';
    if (type === 'message_start' && isRecord(value.message)) {
      id = asString(value.message.id) || id;
      const usage = isRecord(value.message.usage) ? value.message.usage : {};
      inputTokens = num(usage.input_tokens) || inputTokens;
    } else if (type === 'content_block_start') {
      const index = num(value.index);
      const block = isRecord(value.content_block) ? value.content_block : {};
      if (block.type === 'tool_use') {
        blocks.set(index, { kind: 'tool', text: '', id: asString(block.id) || `toolu_${index}`, name: asString(block.name) });
      } else {
        blocks.set(index, { kind: 'text', text: asString(block.text), id: '', name: '' });
      }
    } else if (type === 'content_block_delta') {
      const index = num(value.index);
      const delta = isRecord(value.delta) ? value.delta : {};
      const block = blocks.get(index) ?? { kind: delta.type === 'input_json_delta' ? 'tool' : 'text', text: '', id: `toolu_${index}`, name: '' };
      if (typeof delta.text === 'string') block.text += delta.text;
      if (typeof delta.partial_json === 'string') block.text += delta.partial_json;
      blocks.set(index, block);
    } else if (type === 'message_delta') {
      const delta = isRecord(value.delta) ? value.delta : {};
      if (delta.stop_reason !== undefined) stop = stopFromAnthropic(delta.stop_reason, false);
      if (typeof delta.stop_sequence === 'string') stopSequence = delta.stop_sequence;
      const usage = isRecord(value.usage) ? value.usage : {};
      if (typeof usage.output_tokens === 'number') outputTokens = num(usage.output_tokens);
      if (typeof usage.input_tokens === 'number') inputTokens = num(usage.input_tokens);
    }
  }
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const [, block] of [...blocks.entries()].sort((a, b) => a[0] - b[0])) {
    if (block.kind === 'text') text = joinText(text, block.text);
    else toolCalls.push({ id: block.id, name: block.name, arguments: block.text === '' ? '{}' : block.text });
  }
  if (toolCalls.length > 0 && stop === 'end_turn') stop = 'tool_use';
  return { id: id || 'aggregate', text, toolCalls, stop, stopSequence, inputTokens, outputTokens };
}

export function canonFromGeminiStream(raw: string): CanonResponse {
  const trimmed = raw.trim();
  if ((trimmed.startsWith('{') || trimmed.startsWith('[')) && !trimmed.startsWith('data:')) {
    const value = parseJson(trimmed);
    if (Array.isArray(value)) return mergeGemini(value, 'gemini');
    if (isRecord(value)) return canonFromGeminiResponse(value);
  }
  const chunks: unknown[] = [];
  const payloads: string[] = [];
  for (const frame of framesOf(raw)) if (frame.data.trim() !== '') payloads.push(frame.data);
  if (payloads.length === 0) {
    for (const line of raw.replace(/\r\n/g, '\n').split('\n')) {
      const text = line.trim();
      if (text.startsWith('data:')) payloads.push(text.slice('data:'.length).trim());
    }
  }
  for (const payload of payloads) {
    const value = parseJson(payload);
    if (value !== null) {
      chunks.push(value);
      continue;
    }
    for (const line of payload.split('\n')) {
      const nested = parseJson(line);
      if (nested !== null) chunks.push(nested);
    }
  }
  return mergeGemini(chunks, 'gemini');
}

export function chatResult(result: CanonResponse, model: string): Json {
  const toolCalls = result.toolCalls.map((call) => ({
    id: call.id,
    type: 'function',
    function: { name: call.name, arguments: call.arguments === '' ? '{}' : call.arguments },
  }));
  const message: Json = {
    role: 'assistant',
    content: result.text !== '' ? result.text : toolCalls.length > 0 ? null : '',
  };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: result.id.startsWith('chatcmpl') ? result.id : `chatcmpl_${result.id}`,
    object: 'chat.completion',
    created: CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishFromStop(result.stop) }],
    usage: usageOf(result),
  };
}

export function messagesResult(result: CanonResponse, model: string): Json {
  const content: Json[] = [];
  if (result.text !== '') content.push({ type: 'text', text: result.text });
  for (const call of result.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsObject(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: result.id.startsWith('msg_') ? result.id : `msg_${result.id}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: result.stop === 'end_turn' ? 'end_turn' : result.stop,
    stop_sequence: result.stop === 'stop_sequence' ? result.stopSequence : null,
    usage: { input_tokens: result.inputTokens, output_tokens: result.outputTokens },
  };
}

export function chatStream(result: CanonResponse, model: string): string {
  const id = result.id.startsWith('chatcmpl') ? result.id : `chatcmpl_${result.id}`;
  const chunks: string[] = [];
  chunks.push(sse(null, chunk(id, model, { role: 'assistant' }, null)));
  if (result.text !== '') chunks.push(sse(null, chunk(id, model, { content: result.text }, null)));
  result.toolCalls.forEach((call, index) => {
    chunks.push(
      sse(
        null,
        chunk(
          id,
          model,
          { tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments === '' ? '{}' : call.arguments } }] },
          null,
        ),
      ),
    );
  });
  chunks.push(sse(null, chunk(id, model, {}, finishFromStop(result.stop))));
  chunks.push(sse(null, { id, object: 'chat.completion.chunk', created: CREATED, model, choices: [], usage: usageOf(result) }));
  chunks.push('data: [DONE]\n\n');
  return chunks.join('');
}

export function messagesStream(result: CanonResponse, model: string): string {
  const id = result.id.startsWith('msg_') ? result.id : `msg_${result.id}`;
  const events: string[] = [];
  events.push(
    sse('message_start', {
      type: 'message_start',
      message: {
        id,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: result.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (result.text !== '' || result.toolCalls.length === 0) {
    events.push(sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    if (result.text !== '') {
      events.push(sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: result.text } }));
    }
    events.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of result.toolCalls) {
    events.push(
      sse('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    events.push(
      sse('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: call.arguments === '' ? '{}' : call.arguments },
      }),
    );
    events.push(sse('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  events.push(
    sse('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: result.stop, stop_sequence: result.stop === 'stop_sequence' ? result.stopSequence : null },
      usage: { output_tokens: result.outputTokens },
    }),
  );
  events.push(sse('message_stop', { type: 'message_stop' }));
  return events.join('');
}

function chunk(id: string, model: string, delta: Json, finish: string | null): Json {
  return {
    id,
    object: 'chat.completion.chunk',
    created: CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

function usageOf(result: CanonResponse): Json {
  return {
    prompt_tokens: result.inputTokens,
    completion_tokens: result.outputTokens,
    total_tokens: result.inputTokens + result.outputTokens,
  };
}

function sse(event: string | null, data: unknown): string {
  const name = event === null ? '' : `event: ${event}\n`;
  return `${name}data: ${JSON.stringify(data)}\n\n`;
}

interface Frame {
  event: string | null;
  data: string;
}

function framesOf(raw: string): Frame[] {
  const frames: Frame[] = [];
  for (const block of raw.replace(/\r\n/g, '\n').split('\n\n')) {
    if (block.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length === 0 && event === null) continue;
    frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function mergeGemini(chunks: readonly unknown[], fallbackId: string): CanonResponse {
  let text = '';
  const toolCalls: ToolCall[] = [];
  let stop: StopReason = 'end_turn';
  let inputTokens = 0;
  let outputTokens = 0;
  let id = fallbackId;
  for (const chunk of chunks) {
    if (!isRecord(chunk)) continue;
    if (typeof chunk.responseId === 'string' && chunk.responseId !== '') id = chunk.responseId;
    const usage = isRecord(chunk.usageMetadata) ? chunk.usageMetadata : null;
    if (usage !== null) {
      inputTokens = num(usage.promptTokenCount) || inputTokens;
      outputTokens = num(usage.candidatesTokenCount) || outputTokens;
    }
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) {
      if (isRecord(chunk.promptFeedback)) stop = 'refusal';
      continue;
    }
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') text += part.text;
      if (isRecord(part.functionCall)) {
        const name = asString(part.functionCall.name);
        const args = isRecord(part.functionCall.args) ? part.functionCall.args : {};
        toolCalls.push({ id: `call_${toolCalls.length}_${name}`, name, arguments: JSON.stringify(args) });
      }
    }
    if (candidate.finishReason !== undefined && candidate.finishReason !== null) {
      stop = stopFromGemini(candidate.finishReason, false);
    }
  }
  if (toolCalls.length > 0 && stop === 'end_turn') stop = 'tool_use';
  return { id, text, toolCalls, stop, stopSequence: null, inputTokens, outputTokens };
}

function requestFields(body: Json, system: string, messages: CanonMessage[]): CanonRequest {
  const format = isRecord(body.response_format) ? body.response_format : null;
  const schemaCarrier = format !== null && isRecord(format.json_schema) ? format.json_schema : null;
  const output = isRecord(body.output_format) ? body.output_format : null;
  const jsonSchema = schemaCarrier !== null && isRecord(schemaCarrier.schema) ? schemaCarrier.schema : output !== null && isRecord(output.schema) ? output.schema : null;
  const maxTokens = finite(body.max_tokens) ?? finite(body.max_completion_tokens) ?? finite(body.max_output_tokens);
  return {
    system,
    messages,
    maxTokens,
    temperature: finite(body.temperature),
    topP: finite(body.top_p),
    stop: stopList(body.stop),
    tools: openAITools(body.tools),
    toolChoice: openAIChoiceIn(body.tool_choice),
    jsonMode: format?.type === 'json_object' || format?.type === 'json_schema' || jsonSchema !== null,
    jsonSchema,
    jsonSchemaName: schemaCarrier !== null && typeof schemaCarrier.name === 'string' ? schemaCarrier.name : null,
    stream: body.stream === true,
  };
}

function plain(role: 'user' | 'assistant', text: string): CanonMessage {
  return { role, text, images: [], toolCalls: [], toolCallId: null, toolName: null };
}

function openAIUserContent(message: CanonMessage): string | Json[] {
  if (message.images.length === 0) return message.text;
  const parts: Json[] = [];
  if (message.text !== '') parts.push({ type: 'text', text: message.text });
  for (const image of message.images) parts.push({ type: 'image_url', image_url: { url: imageUrl(image) } });
  return parts;
}

function imageUrl(image: ImagePart): string {
  if (image.data !== null) return `data:${image.mediaType};base64,${image.data}`;
  return image.url ?? '';
}

function anthropicBlocks(message: CanonMessage): Json[] {
  if (message.role === 'tool') {
    return [{ type: 'tool_result', tool_use_id: message.toolCallId ?? '', content: message.text }];
  }
  const blocks: Json[] = [];
  if (message.role === 'user') {
    if (message.text !== '') blocks.push({ type: 'text', text: message.text });
    for (const image of message.images) {
      if (image.data !== null) {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } });
      } else if (image.url !== null) blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
  } else if (message.text !== '') blocks.push({ type: 'text', text: message.text });
  for (const call of message.toolCalls) {
    blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsObject(call.arguments) });
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  return blocks;
}

function geminiParts(message: CanonMessage): Json[] {
  const parts: Json[] = [];
  if (message.role === 'tool') {
    const name = message.toolName ?? message.toolCallId ?? 'tool';
    const parsed = parseJson(message.text);
    const response = isRecord(parsed) ? parsed : { content: message.text };
    parts.push({ functionResponse: { name, response } });
    return parts;
  }
  if (message.text !== '') parts.push({ text: message.text });
  for (const image of message.images) {
    if (image.data !== null) parts.push({ inlineData: { mimeType: image.mediaType, data: image.data } });
    else if (image.url !== null) parts.push({ fileData: { mimeType: image.mediaType, fileUri: image.url } });
  }
  for (const call of message.toolCalls) {
    parts.push({ functionCall: { name: call.name, args: argumentsObject(call.arguments) } });
  }
  if (parts.length === 0) parts.push({ text: '' });
  return parts;
}

function geminiToolConfig(choice: ToolChoice | null): Json {
  if (choice === null || choice === 'auto') return { mode: 'AUTO' };
  if (choice === 'none') return { mode: 'NONE' };
  if (choice === 'required') return { mode: 'ANY' };
  return { mode: 'ANY', allowedFunctionNames: [choice.name] };
}

function openAIToolChoice(choice: ToolChoice): string | Json {
  if (choice === 'auto' || choice === 'none' || choice === 'required') return choice;
  return { type: 'function', function: { name: choice.name } };
}

function anthropicChoiceOut(choice: ToolChoice): Json {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'none') return { type: 'none' };
  if (choice === 'required') return { type: 'any' };
  return { type: 'tool', name: choice.name };
}

function openAIChoiceIn(value: unknown): ToolChoice | null {
  if (value === 'auto' || value === 'none' || value === 'required') return value;
  if (!isRecord(value)) return null;
  const fn = isRecord(value.function) ? value.function : null;
  if (fn !== null && typeof fn.name === 'string') return { name: fn.name };
  return null;
}

function anthropicToolChoice(value: unknown): ToolChoice | null {
  if (!isRecord(value)) return null;
  if (value.type === 'auto') return 'auto';
  if (value.type === 'none') return 'none';
  if (value.type === 'any') return 'required';
  if (value.type === 'tool' && typeof value.name === 'string') return { name: value.name };
  return null;
}

function openAITools(value: unknown): CanonTool[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((tool) => {
    if (!isRecord(tool)) return [];
    const fn = tool.type === 'function' && isRecord(tool.function) ? tool.function : tool;
    if (!isRecord(fn) || typeof fn.name !== 'string') return [];
    const parameters = isRecord(fn.parameters) ? fn.parameters : isRecord(fn.input_schema) ? fn.input_schema : { type: 'object', properties: {} };
    return [{ name: fn.name, description: asString(fn.description), parameters }];
  });
}

function anthropicTools(value: unknown[]): CanonTool[] {
  return value.flatMap((tool) => {
    if (!isRecord(tool) || typeof tool.name !== 'string') return [];
    return [
      {
        name: tool.name,
        description: asString(tool.description),
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

function openAIToolCalls(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((call, index) => {
    if (!isRecord(call) || !isRecord(call.function)) return [];
    return [
      {
        id: asString(call.id) || `call_${index}`,
        name: asString(call.function.name),
        arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
      },
    ];
  });
}

function toolUseBlock(block: Json): ToolCall {
  const input = isRecord(block.input) ? block.input : {};
  return { id: asString(block.id) || 'toolu_unknown', name: asString(block.name), arguments: JSON.stringify(input) };
}

function argumentsObject(value: string): Json {
  const parsed = parseJson(value);
  return isRecord(parsed) ? parsed : {};
}

function partsOf(content: unknown): { text: string; images: ImagePart[] } {
  if (typeof content === 'string') return { text: content, images: [] };
  if (!Array.isArray(content)) return { text: '', images: [] };
  let text = '';
  const images: ImagePart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if ((block.type === 'text' || block.type === 'input_text') && typeof block.text === 'string') text = joinText(text, block.text);
    const image = imageFromOpenAI(block);
    if (image !== null) images.push(image);
  }
  return { text, images };
}

function imageFromOpenAI(block: Json): ImagePart | null {
  if (block.type !== 'image_url' && block.type !== 'input_image') return null;
  const carrier = block.image_url ?? block.image;
  const url = typeof carrier === 'string' ? carrier : isRecord(carrier) ? asString(carrier.url) : '';
  if (url === '') return null;
  return imageFromUrl(url);
}

function imageFromAnthropic(block: Json): ImagePart | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { mediaType: asString(source.media_type) || 'image/png', data: source.data, url: null };
  }
  if (source.type === 'url' && typeof source.url === 'string') return imageFromUrl(source.url);
  return null;
}

function imageFromUrl(url: string): ImagePart {
  const matched = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (matched !== null && matched[2] === ';base64') {
    return { mediaType: matched[1] || 'image/png', data: matched[3] ?? '', url: null };
  }
  return { mediaType: 'image/png', data: null, url };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((block) => (isRecord(block) && typeof block.text === 'string' ? [block.text] : [])).join('\n');
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  if (content === undefined || content === null) return '';
  return JSON.stringify(content);
}

function stopList(value: unknown): string[] {
  if (typeof value === 'string' && value !== '') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function stopFromFinish(finish: unknown): StopReason {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function stopFromAnthropic(reason: unknown, hasTools: boolean): StopReason {
  if (reason === 'max_tokens') return 'max_tokens';
  if (reason === 'tool_use') return 'tool_use';
  if (reason === 'refusal') return 'refusal';
  if (reason === 'stop_sequence') return 'stop_sequence';
  if (hasTools) return 'tool_use';
  return 'end_turn';
}

function stopFromGemini(reason: unknown, hasTools: boolean): StopReason {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'refusal';
  if (hasTools) return 'tool_use';
  return 'end_turn';
}

function finishFromStop(stop: StopReason): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function joinText(left: string, right: string): string {
  if (left === '') return right;
  if (right === '') return left;
  return `${left}\n${right}`;
}

function num(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function finite(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function parseJson(text: string): unknown {
  if (text.trim() === '') return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}
