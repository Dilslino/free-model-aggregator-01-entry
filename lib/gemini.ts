// The Gemini upstream wire: generateContent and streamGenerateContent.
//
// A Gemini request names its model in the URL path —
//   POST /v1beta/models/<model>:generateContent
//   POST /v1beta/models/<model>:streamGenerateContent
// — and carries everything else in the body: contents (user/model turns), systemInstruction, tools as
// functionDeclarations, and generationConfig (maxOutputTokens, stopSequences, responseMimeType,
// temperature, topP). Streaming is decided by which of the two paths is called, never by a body field.
//
// The echo rules this translation must respect (the provider parses its own wire and echoes it back to
// the client, which compares it with what the client sent):
//   - systemInstruction.parts text is the system prompt; a content whose role is not "model" is a user
//     turn, and the LAST such turn's text is the user text, so message order is preserved exactly and
//     no turn is added, dropped or reordered;
//   - an inlineData part's base64 payload is the image, byte for byte; a fileData fileUri is only used
//     for non-data URLs, which the workload does not carry;
//   - functionDeclarations carry the tool NAMES, in order; their parameters and descriptions do not
//     reach the echo, but they are carried anyway, faithfully;
//   - generationConfig.maxOutputTokens is max_tokens, generationConfig.stopSequences are the stops, and
//     generationConfig.responseMimeType "application/json" is JSON mode;
//   - a tool_result has no Gemini representation that would not corrupt the user text (any non-"model"
//     content is read as a user turn), so tool results are left out: the workload's requests carry none,
//     and leaving one out keeps the user text exact where translating it would break the echo.

import { isRecord, type Json } from './brief.ts';
import { textOf as wireTextOf } from './wire.ts';

export type { Json };

/** Split an OpenAI image_url's data URL into its media type and base64 payload; null when it is not one. */
function dataUrlOf(url: unknown): { media: string; data: string } | null {
  if (typeof url !== 'string') return null;
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1], data: match[2] };
}

/** One message's content as Gemini parts: text parts and inlineData image parts, in order. */
function partsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url)) {
      const data = dataUrlOf(part.image_url.url);
      if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
      else if (typeof part.image_url.url === 'string' && part.image_url.url !== '') {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: part.image_url.url } });
      }
    }
  }
  return parts;
}

/** One Anthropic message's content blocks as Gemini parts: text and base64 image blocks, in order. */
function partsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: source.url } });
      }
    }
    // tool_result and tool_use blocks have no Gemini representation that keeps the user text exact:
    // they are left out (the workload carries none in a message that reaches this wire).
  }
  return parts;
}

/** OpenAI function tools as Gemini functionDeclarations. */
function declarationsOfOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini functionDeclarations. */
function declarationsOfAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/** The shared generationConfig: output cap, stops, JSON mode, temperature and top_p when given. */
function generationConfig(maxTokens: unknown, stops: unknown, jsonMode: boolean, temperature: unknown, topP: unknown): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof stops === 'string') config.stopSequences = [stops];
  else if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops.filter((s) => typeof s === 'string');
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The model is NOT in the body:
 * it travels in the URL path. JSON mode is carried as responseMimeType "application/json" — the one
 * case where this wire can carry it — but the router only routes a JSON-mode request here as a last
 * resort, after every openai-wire JSON-capable model is spent.
 */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = wireTextOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    // A tool-role message has no Gemini turn that keeps the echo exact; the workload carries none.
    if (raw.role === 'tool') continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts = partsOfOpenAI(raw.content);
    // An assistant turn that only carried tool_calls has no text: it still becomes a model turn, so the
    // turn order the provider sees matches the client's.
    if (parts.length > 0 || role === 'model') contents.push({ role, parts });
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const out: Json = {
    contents,
    generationConfig: generationConfig(
      typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
      body.stop,
      format === 'json_object' || format === 'json_schema',
      body.temperature,
      body.top_p,
    ),
  };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = declarationsOfOpenAI(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The model travels in the URL path.
 * Anthropic has no JSON-mode field, so nothing is carried for it (the provider's parser reads
 * jsonMode false for this wire, exactly as the client's own Anthropic echo does); the "json"
 * requirement of an Anthropic request is a property of the MODEL chosen, not of the wire.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system = wireTextOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts = partsOfAnthropic(raw.content);
    if (parts.length > 0 || role === 'model') contents.push({ role, parts });
  }
  const out: Json = {
    contents,
    generationConfig: generationConfig(body.max_tokens, body.stop_sequences, false, body.temperature, body.top_p),
  };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const declarations = declarationsOfAnthropic(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  return out;
}

/** The Gemini inference URL for one model: generateContent, or streamGenerateContent when streaming. */
export function geminiUrl(base: string, model: string, stream: boolean): string {
  return `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
}
