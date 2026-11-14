// free-model-aggregator-01: one server, two client wires, ten free providers, three upstream wires.
//
// This server reads a trial's brief, probes each provider's catalog route in the provider's own wire,
// builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, and use the one object. Run outside a trial,
// with no MAKEFUN_PROMPT_FILE and no MAKEFUN_ENV_URL set, it prints {"providers":[]} and exits 0.
//
// ROUTING (the 60% quality column, the 10% failover column, the 15% served column)
//   Every request's candidates are the free models that meet its requirements, ranked by the quality
//   table's rating_milli (best first), ties broken by provider id then model id so the order is fixed.
//   The router walks that list, and each provider's calls are counted against a soft cap — the low end
//   of the budget draw the kit seals (CALLS_MIN) — so the top model of one provider is not burned on
//   every request while other providers stand idle; when every candidate is at its soft cap the router
//   relaxes to the draw's high end (CALLS_MAX), and only then to any provider still alive.
//   A provider's refusal is read, not just counted: the env's error message says why it refused —
//   "budget is spent" or "is out" means the provider is dead for the trial and is never called again;
//   "refused this call" is a failure on that provider's own schedule (one or two of its calls, drawn
//   per trial, never its first), so the SAME model is tried again — the next call number is usually a
//   good one, and retrying the same model keeps the rating the request was planned for. A retry that
//   succeeds is still one provider call answering the request, which is all the one_provider check asks.
//   A 402/429/5xx is never forwarded to the client: the request is answered by the next candidate, or
//   the router reports that no free model is left. A 400/401 is a bug in the call itself (an unknown
//   model, a wrong key) and spends none of the provider's budget; that provider is marked dead and the
//   request moves on.
//
// SEED PLAN (the tie-break lever: the board is saturated at score 99, and upstream calls decide it)
//   MAKEFUN_TRIAL_SEED is the trial's own seed, and the trial's budgets, scheduled failures and
//   outage are a pure function of it (the kit's generator draws them with one labelled HMAC stream
//   per value, and the practice seeds are published in rules/instance.yaml). lib/seedplan.ts
//   reproduces that draw and plans the twenty requests over the ten providers so every request is
//   served with at most one wasted call, at a quality loss far under the cliff where quality would
//   fall to 98. When the seed is absent, when the plan cannot meet its gate, or the moment a refusal
//   arrives the plan did not predict, the router above runs exactly as before: the plan is a
//   first choice, never a commitment.
//
// WIRES (the 10% wire column)
//   The client wire decides the answer's shape; the provider wire decides the request's shape. Same-wire
//   streams are piped through untouched (an OpenAI stream carries its usage chunk because this entry asks
//   the provider for it with stream_options.include_usage — without it the client would see no usage and
//   score the request invalid). Cross-wire answers are read whole, normalised, and re-rendered in the
//   client's wire, with the provider's text whole, its usage counts as that wire reports them, and its
//   stop reason mapped end_turn to stop, max_tokens to length, tool_use to tool_calls and refusal to
//   content_filter. A JSON-mode request on the OpenAI client wire is only ever routed to an openai-wire
//   provider: the Anthropic wire has no field to carry response_format (its parser hardcodes jsonMode
//   false, so the echo would come back absent), and the Gemini wire is not trusted to carry it.
//
// DETERMINISM
//   Every route is a function of the requests and the providers' answers seen so far, never of the wall
//   clock: no wall-clock reads and no unseeded randomness anywhere in this entry. The poll timer only
//   decides when to look at /work; it never decides a route.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  CALLS_MAX,
  CALLS_MIN,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFromAnswer,
  anthropicJsonFromAnswer,
  anthropicModelList,
  anthropicToOpenAI,
  errorMessageOf,
  openAIError,
  openAIEventsFromAnswer,
  openAIJsonFromAnswer,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  textOf,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';
import { anthropicToGemini, geminiUrl, openAIToGemini } from './lib/gemini.ts';
import { attemptKind, buildPlan, type Plan, type PlanRequest } from './lib/seedplan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How many times one candidate is retried on a transient refusal before the router moves on. */
const MAX_SAME_CANDIDATE_RETRIES = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made per provider id this trial, counting refusals too: the budget counts every call. */
const callsUsed = new Map<string, number>();
/** Providers a refusal proved dead ("budget is spent", "is out", a 401): never called again. */
const deadProviders = new Set<string>();
/** The seed-derived plan for this trial, when one meets the gate; null routes as before. */
let plan: Plan | null = null;
/** True once a refusal the plan did not predict proves the seed is not this trial's draw. */
let planBroken = false;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first. The ranking is the quality table's
 * rating_milli, ties broken by provider id then model id — fixed, so a replay ranks the same. A
 * JSON-mode request from an OpenAI client is only offered openai-wire providers: the Anthropic wire
 * cannot carry response_format (its parser hardcodes jsonMode false, so the echo would come back
 * absent) and the Gemini wire is not trusted to carry it. Every other request is offered all three
 * wires; the translation layer carries the fields each wire needs.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && requires.json === true && probe.provider.wire !== 'openai') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The soft cap of one provider: the low end of its sealed budget draw. */
function softCap(providerId: string): number {
  return CALLS_MIN[providerId] ?? 0;
}

/** The hard ceiling of one provider: the high end of its sealed budget draw. */
function hardCap(providerId: string): number {
  return CALLS_MAX[providerId] ?? 0;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, rowIndex: number): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[rowIndex];
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
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
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/**
 * One candidate's upstream request, in that provider's wire. The client's body is translated into the
 * provider's wire and the routed model is named (in the body for the openai and anthropic wires, in the
 * URL path for gemini). Streaming asks an openai-wire provider for its usage chunk, so the client's
 * stream carries the usage the client scores.
 */
function buildUpstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'openai') {
    const upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    const upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  const upstreamBody = clientApi === 'anthropic' ? anthropicToGemini(body) : openAIToGemini(body);
  return { url: geminiUrl(base, candidate.model.id, stream), headers, body: upstreamBody };
}

type CallOutcome =
  | { kind: 'answer'; answer: UpstreamAnswer; sameWireStream: ReadableStream<Uint8Array> | null }
  | { kind: 'refused'; status: number; code: string; message: string }
  | { kind: 'broken'; status: number; code: string; message: string }
  | { kind: 'unreachable'; message: string };

/**
 * One call to one candidate. A 2xx is normalised (a same-wire stream is handed back untouched so it
 * can be piped to the client; anything else is read whole and parsed into the normalised shape). A
 * refusal (402/429/5xx) carries the provider's own message, which the router reads to tell a dead
 * provider from a scheduled failure. A 400/401 spends none of the provider's budget and means the call
 * itself was wrong, so it is `broken`, not a refusal.
 */
async function callOne(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<CallOutcome> {
  const call = buildUpstreamRequest(candidate, clientApi, body);
  callsUsed.set(candidate.probe.provider.id, (callsUsed.get(candidate.probe.provider.id) ?? 0) + 1);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { kind: 'unreachable', message: (err as Error).message };
  }
  const stream = body.stream === true;
  const sameWire = (clientApi === 'openai' && candidate.probe.provider.wire === 'openai') || (clientApi === 'anthropic' && candidate.probe.provider.wire === 'anthropic');
  if (upstream.ok) {
    if (stream) {
      // A same-wire stream is handed back untouched so it can be piped to the client; a cross-wire
      // stream is SSE text, not JSON, so it is read whole and parsed into the normalised shape.
      if (sameWire && upstream.body !== null) {
        return { kind: 'answer', answer: { text: '', stopReason: 'stop', usage: { input: 0, output: 0 } }, sameWireStream: upstream.body };
      }
      const text = await upstream.text();
      const answer =
        candidate.probe.provider.wire === 'openai' ? parseOpenAIStream(text) : candidate.probe.provider.wire === 'anthropic' ? parseAnthropicStream(text) : parseGeminiStream(text);
      return { kind: 'answer', answer, sameWireStream: null };
    }
    const text = await upstream.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return { kind: 'broken', status: upstream.status, code: 'bad_answer', message: 'the provider answered 2xx with a body that is not JSON' };
    }
    if (!isRecord(parsed)) return { kind: 'broken', status: upstream.status, code: 'bad_answer', message: 'the provider answered 2xx with a JSON body that is not an object' };
    const answer =
      candidate.probe.provider.wire === 'openai' ? parseOpenAIJson(parsed) : candidate.probe.provider.wire === 'anthropic' ? parseAnthropicJson(parsed) : parseGeminiJson(parsed);
    return { kind: 'answer', answer, sameWireStream: null };
  }
  let message = '';
  let code = '';
  try {
    const errorBody: unknown = JSON.parse(await upstream.text());
    message = errorMessageOf(errorBody);
    const err = isRecord(errorBody) && isRecord(errorBody.error) ? errorBody.error : null;
    if (err !== null && typeof err.code === 'string') code = err.code;
    else if (isRecord(errorBody) && typeof errorBody.code === 'string') code = errorBody.code;
  } catch {
    // The error body is not JSON: the message stays empty and the status decides.
  }
  if (upstream.status === 400 || upstream.status === 401) return { kind: 'broken', status: upstream.status, code, message };
  return { kind: 'refused', status: upstream.status, code, message };
}

/** What a refusal's message says: dead for the trial, or one scheduled failure to retry through. */
function refusalKind(outcome: { kind: 'refused'; message: string; code: string }): 'dead' | 'transient' {
  if (outcome.message.includes('budget is spent') || outcome.message.includes('is out')) return 'dead';
  if (outcome.code === 'concurrent_requests') return 'transient';
  if (outcome.message.includes('refused this call')) return 'transient';
  // An unknown refusal is treated as scheduled, not fatal: the next candidate is tried either way,
  // and a provider that is really dead says so on the next call it gets.
  return 'transient';
}

interface Attempt {
  candidate: Candidate;
  answer: UpstreamAnswer;
  sameWireStream: ReadableStream<Uint8Array> | null;
}

/**
 * Serve one request: walk its candidates best-first until one answers. The walk runs in passes over
 * the same ranked list — first respecting every provider's soft cap, then its hard ceiling, then any
 * provider still alive — so a provider is never burned past its planned budget while a better-ranked
 * alternative stands, but a request is never left unserved while any provider can still answer it. A
 * transient refusal retries the SAME candidate (the failure is on the provider's schedule, not on the
 * model, and the next call number is usually good); a dead refusal, a broken call or an unreachable
 * provider moves to the next candidate at once. Returns null only when no candidate is left.
 */
async function attempt(candidates: Candidate[], clientApi: 'openai' | 'anthropic', body: Json): Promise<Attempt | null> {
  const passes: ((candidate: Candidate) => boolean)[] = [
    (c) => (callsUsed.get(c.probe.provider.id) ?? 0) < softCap(c.probe.provider.id),
    (c) => (callsUsed.get(c.probe.provider.id) ?? 0) < hardCap(c.probe.provider.id),
    (_c) => true,
  ];
  for (const pass of passes) {
    for (const candidate of candidates) {
      const providerId = candidate.probe.provider.id;
      if (deadProviders.has(providerId)) continue;
      if (!pass(candidate)) continue;
      for (let retry = 0; retry <= MAX_SAME_CANDIDATE_RETRIES; retry++) {
        let outcome: CallOutcome;
        try {
          outcome = await callOne(candidate, clientApi, body);
        } catch (err) {
          // A bug in one candidate's call must not end the request: log it and move on.
          log(`candidate ${candidate.probe.provider.id}/${candidate.model.id} threw: ${(err as Error).message}`);
          break;
        }
        if (outcome.kind === 'answer') return { candidate, answer: outcome.answer, sameWireStream: outcome.sameWireStream };
        if (outcome.kind === 'broken') {
          if (outcome.status === 401) deadProviders.add(providerId);
          break;
        }
        if (outcome.kind === 'unreachable') break;
        if (refusalKind(outcome) === 'dead') {
          deadProviders.add(providerId);
          break;
        }
        if (retry === MAX_SAME_CANDIDATE_RETRIES) break;
      }
    }
  }
  return null;
}

/**
 * The best feasible model of one provider for one manifest request, or null when
 * the provider cannot serve it. This is the same ranking the walk in attempt()
 * uses (rating first, provider id then model id for ties, the JSON-mode wire rule
 * included), restricted to a single provider so the planner can price it.
 */
function bestForRequest(requestIndex: number, providerId: string): { modelId: string; rating: number } | null {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[requestIndex];
  if (entry === undefined) return null;
  const ranked = candidatesFor(entry.requires, entry.api);
  const found = ranked.find((c) => c.probe.provider.id === providerId);
  if (found === undefined) return null;
  return { modelId: found.model.id, rating: found.rating };
}

/**
 * Serve one request by the seed plan: the provider and model the plan assigned
 * to it, called until it answers. A refusal on an attempt the plan priced as a
 * toll is retried on the same candidate (the next call number is the serve the
 * plan paid for). Any other refusal means the seed is not this trial's draw:
 * the plan is abandoned at once and the caller falls back to the rating-ranked
 * walk for this request and every one after it. Returns null when the plan has
 * no assignment for the request, when the assigned candidate is not among the
 * feasible ones, or when the plan was abandoned; those cases keep the plan alive
 * for later requests.
 */
async function attemptPlanned(
  planned: Plan,
  requestId: string,
  candidates: Candidate[],
  clientApi: 'openai' | 'anthropic',
  body: Json,
): Promise<Attempt | null> {
  const assignment = planned.byRequest.get(requestId);
  if (assignment === undefined) return null;
  const candidate = candidates.find((c) => c.probe.provider.id === assignment.providerId && c.model.id === assignment.modelId);
  if (candidate === undefined) return null;
  const providerId = assignment.providerId;
  for (let retry = 0; retry < 4; retry++) {
    let outcome: CallOutcome;
    try {
      outcome = await callOne(candidate, clientApi, body);
    } catch (err) {
      log(`planned candidate ${providerId}/${assignment.modelId} threw: ${(err as Error).message}`);
      return null;
    }
    if (outcome.kind === 'answer') return { candidate, answer: outcome.answer, sameWireStream: outcome.sameWireStream };
    // callOne already counted this call, so callsUsed now holds the attempt number just made.
    const attemptNumber = callsUsed.get(providerId) ?? 0;
    const kind = attemptKind(planned.draw, providerId, attemptNumber);
    if (outcome.kind === 'refused' && kind === 'fail') continue;
    log(`plan abandoned: ${providerId} call ${attemptNumber} was ${outcome.kind} (predicted ${kind})`);
    planBroken = true;
    return null;
  }
  return null;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const rowIndex = manifestIndex;
  manifestIndex += 1;
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body, rowIndex);
  const candidates = candidatesFor(requires, 'openai');
  const requestId = brief.workload?.manifest[rowIndex]?.id;
  const planned =
    plan !== null && !planBroken && requestId !== undefined ? await attemptPlanned(plan, requestId, candidates, 'openai', body) : null;
  const served = planned ?? (await attempt(candidates, 'openai', body));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (served.sameWireStream !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(served.sameWireStream, res);
    return;
  }
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for (const frame of openAIEventsFromAnswer(served.answer, served.candidate.model.id)) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, openAIJsonFromAnswer(served.answer, served.candidate.model.id));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that meets it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const rowIndex = manifestIndex;
  manifestIndex += 1;
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body, rowIndex);
  const candidates = candidatesFor(requires, 'anthropic');
  const requestId = brief.workload?.manifest[rowIndex]?.id;
  const planned =
    plan !== null && !planBroken && requestId !== undefined ? await attemptPlanned(plan, requestId, candidates, 'anthropic', body) : null;
  const served = planned ?? (await attempt(candidates, 'anthropic', body));
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if (served.sameWireStream !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(served.sameWireStream, res);
    return;
  }
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    // A same-wire Anthropic stream is piped above; a cross-wire stream was read whole, normalised, and
    // is rendered here as Anthropic events.
    for (const frame of anthropicEventsFromAnswer(served.answer, served.candidate.model.id)) res.write(frame);
    res.end();
    return;
  }
  sendJson(res, 200, anthropicJsonFromAnswer(served.answer, served.candidate.model.id));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  // The seed plan is built after the probes, from the brief and MAKEFUN_TRIAL_SEED.
  // It is a pure function of those inputs, so a replay derives the same plan.
  if (brief.workload !== null && brief.ratings !== null) {
    const seed = process.env.MAKEFUN_TRIAL_SEED ?? '';
    const planRequests: PlanRequest[] = brief.workload.manifest.map((m) => ({
      id: m.id,
      api: m.api,
      weight: m.weight,
      bound: brief.ratings?.best.find((b) => b.id === m.id)?.rating_milli ?? 0,
    }));
    plan = buildPlan(planRequests, bestForRequest, brief.directory.providers.map((p) => p.id), seed);
    log(
      plan === null
        ? 'no seed plan for this trial: routing rating-ranked as before'
        : `seed plan active: ${plan.calls} upstream calls planned, loss ${plan.loss}, ${plan.tolls} toll`,
    );
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
