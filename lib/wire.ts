// The client wires and the upstream wires, and the translations between them.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Every upstream provider speaks
// one of three wires (openai, anthropic, gemini). This file translates in both directions:
//
//   - request translation: the client's body is rewritten into the provider's wire, carrying the
//     request's fields untouched in meaning — its system prompt, its user text (images included), its
//     tools, its stop sequences, its max_tokens, its JSON mode. The provider echoes what it parsed and
//     the client compares that echo with what it sent, so a field rewritten or dropped loses the
//     request. The one exception is deliberate and load-bearing: an OpenAI JSON-mode request (a
//     response_format on the OpenAI wire) can only be served by an openai-wire provider, because the
//     Anthropic wire has no JSON-mode field to carry (its parser hardcodes jsonMode false) and the
//     Gemini wire is not trusted to carry it; the router in main.ts enforces that, this file only
//     translates.
//   - answer translation: every provider answer is first normalised into one shape (text, stop reason
//     in OpenAI's vocabulary, usage in input/output tokens) and then rendered in the wire the client
//     used, streamed or not as it asked. Normalising first means the six client/upstream combinations
//     share one rendering path, and the stop reason and usage are translated exactly once, in one place.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One provider answer, normalised: the text whole, the stop reason in OpenAI's vocabulary, the usage. */
export interface UpstreamAnswer {
  text: string;
  /** OpenAI's finish_reason vocabulary: stop, length, tool_calls or content_filter. */
  stopReason: string;
  usage: { input: number; output: number };
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The fixed `created` of a Chat Completions answer, mirroring the provider's own fixed value. */
const FIXED_ANSWER_CREATED = 1767225600;
/** How wide one text delta is when this entry re-chunks a streamed answer: the provider's own width. */
const SSE_DELTA_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

export function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'content_filter';
  return 'stop';
}

/** Split text into the provider's own delta width, so a re-chunked stream reads the same. */
export function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through. Streaming asks the provider
 * for the usage chunk too (stream_options.include_usage): without it an OpenAI stream carries no usage
 * and the client scores the request invalid on usage.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it. Used when an
 * Anthropic client is served by an openai-wire provider: the provider's stream is converted as it
 * arrives, so nothing is buffered.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** Split an OpenAI image_url's data URL into its media type and base64 payload; null when it is not one. */
function dataUrlOf(url: unknown): { media: string; data: string } | null {
  if (typeof url !== 'string') return null;
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1], data: match[2] };
}

/** One OpenAI message as Anthropic content blocks: text parts and image_url parts, in order. */
function anthropicBlocksOf(content: unknown): Json[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url)) {
      const url = part.image_url.url;
      const data = dataUrlOf(url);
      if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } });
      else if (typeof url === 'string' && url !== '') blocks.push({ type: 'image', source: { type: 'url', url } });
    }
  }
  return blocks;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`. The system
 * message becomes the system prompt, image_url parts become image blocks, tool messages become
 * tool_result blocks, assistant tool_calls become tool_use blocks, tools and tool_choice are rewritten,
 * and the output cap, temperature, top_p, stop sequences and streaming pass through. NOTE: an OpenAI
 * JSON-mode request (response_format) has no Anthropic equivalent — the Anthropic wire's parser
 * hardcodes jsonMode false, so the echo would come back absent and the client would score the request
 * invalid. The router never sends a JSON-mode request down this path.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks = anthropicBlocksOf(raw.content);
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** The human-readable message of a provider's error body, in any of the three wires' envelopes. */
export function errorMessageOf(body: unknown): string {
  if (!isRecord(body)) return '';
  const openai = isRecord(body.error) ? body.error : null;
  if (openai !== null && typeof openai.message === 'string') return openai.message;
  return '';
}

/** One Chat Completions JSON answer as the normalised shape. */
export function parseOpenAIJson(body: Json): UpstreamAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stopReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
  };
}

/** One Chat Completions SSE stream, read whole, as the normalised shape. */
export function parseOpenAIStream(text: string): UpstreamAnswer {
  let out = '';
  let stopReason = 'stop';
  let input = 0;
  let output = 0;
  for (const frame of text.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]' || payload === '') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = numberOrZero(chunk.usage.prompt_tokens);
        output = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (typeof choice.finish_reason === 'string') stopReason = choice.finish_reason;
    }
  }
  return { text: out, stopReason, usage: { input, output } };
}

/** One Anthropic Messages JSON answer as the normalised shape. */
export function parseAnthropicJson(body: Json): UpstreamAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stopReason: finishReasonOf(body.stop_reason),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
  };
}

/** One Anthropic Messages SSE stream, read whole, as the normalised shape. */
export function parseAnthropicStream(text: string): UpstreamAnswer {
  let out = '';
  let stopReason = 'stop';
  let input = 0;
  let output = 0;
  for (const frame of text.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' ) continue;
      let event: unknown;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(event)) continue;
      if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
        input = numberOrZero(event.message.usage.input_tokens);
      }
      if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta' && typeof event.delta.text === 'string') {
        out += event.delta.text;
      }
      if (event.type === 'message_delta') {
        if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stopReason = finishReasonOf(event.delta.stop_reason);
        if (isRecord(event.usage)) output = numberOrZero(event.usage.output_tokens);
      }
    }
  }
  return { text: out, stopReason, usage: { input, output } };
}

/** One Gemini generateContent JSON answer as the normalised shape. */
export function parseGeminiJson(body: Json): UpstreamAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    stopReason: finishReasonOfGemini(candidate.finishReason),
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
  };
}

/** One Gemini streamGenerateContent SSE stream, read whole, as the normalised shape. */
export function parseGeminiStream(text: string): UpstreamAnswer {
  let out = '';
  let stopReason = 'stop';
  let input = 0;
  let output = 0;
  for (const frame of text.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate !== null) {
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
        if (typeof candidate.finishReason === 'string') stopReason = finishReasonOfGemini(candidate.finishReason);
      }
      if (isRecord(chunk.usageMetadata)) {
        input = numberOrZero(chunk.usageMetadata.promptTokenCount);
        output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
    }
  }
  return { text: out, stopReason, usage: { input, output } };
}

/** A normalised answer as a Chat Completions JSON object, naming `model` back to the client. */
export function openAIJsonFromAnswer(answer: UpstreamAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_ANSWER_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: answer.stopReason, logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

/** A normalised answer as the Chat Completions SSE sequence: role, content, finish, usage, [DONE]. */
export function openAIEventsFromAnswer(answer: UpstreamAnswer, model: string): string[] {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_ANSWER_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, answer.stopReason)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_ANSWER_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames;
}

/** A normalised answer as an Anthropic Messages JSON object, naming `model` back to the client. */
export function anthropicJsonFromAnswer(answer: UpstreamAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopReasonOf(answer.stopReason),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A normalised answer as the Anthropic SSE sequence, in the client's expected event order. */
export function anthropicEventsFromAnswer(answer: UpstreamAnswer, model: string): string[] {
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(answer.stopReason), stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames;
}
