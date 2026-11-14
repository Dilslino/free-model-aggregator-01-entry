# free-model-aggregator-01

One server that serves every request with the best free model it can get: it speaks the OpenAI
and Anthropic client wires, and routes each request over ten simulated free-tier providers that
speak the OpenAI, Anthropic and Gemini wires.

## How it works

- **Brief first.** The trial's brief (`$MAKEFUN_PROMPT_FILE`) carries the provider directory, the
  quality table (each model's rating in milli-points and what it can do) and the workload manifest
  (the twenty requests, in order, with the requirements a model must meet). The entry reads all
  three, probes each provider's catalog route in the provider's own wire, and builds one catalog.
- **Rating-ranked routing.** Every request's candidates are the free models that meet its
  requirements, ranked by `rating_milli` (best first). Calls are spread across providers under soft
  caps taken from the low end of each provider's sealed budget draw, so no single provider is burned
  out while others stand idle.
- **Failover that reads the refusal.** A provider's error message says why it refused: "budget is
  spent" or "is out" marks it dead for the trial; "refused this call" is one of its scheduled
  failures, so the same model is retried — the next call number is usually good, and retrying the
  same model keeps the rating the request was planned for. A 402/429/5xx is never forwarded.
- **All three wires.** The client wire decides the answer's shape; the provider wire decides the
  request's shape. Same-wire streams are piped through untouched (an OpenAI stream carries its usage
  chunk because the entry asks the provider for it). Cross-wire answers are read whole, normalised
  to one shape (text, stop reason, usage) and re-rendered in the client's wire. JSON-mode requests
  on the OpenAI client wire only ever go to an openai-wire provider: the Anthropic wire has no field
  to carry `response_format`.
- **Determinism.** Every route is a function of the requests and the providers' answers seen so far
  — no wall clock, no unseeded randomness anywhere.

## Run

```sh
node main.ts --list        # print the catalog document and exit 0
node main.ts               # serve a trial: reads $MAKEFUN_PROMPT_FILE and $MAKEFUN_ENV_URL,
                           # registers, serves the 20 requests, exits 0 when the work is done
```

Node 22 standard library only; no dependencies. `main.ts` is the entrypoint, `lib/brief.ts` is the
brief parser and catalog probe, `lib/wire.ts` is the OpenAI/Anthropic translation, `lib/gemini.ts`
is the Gemini translation.

## Score (practice trials, kit's own env + scorer)

| trial | score | quality | served | wire | failover | catalog |
|-------|------:|--------:|-------:|-----:|---------:|--------:|
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 |

20/20 requests valid on every trial; 25 upstream calls on trial-0 and trial-1, 28 on trial-2.
Quality 99 (not 100) because the soft caps spread the last no-requirement requests from the 1475
model onto the 1470 models once the top provider reaches its planned budget — a 5-milli trade for
keeping every provider alive. State exports are byte-identical across repeated runs of the same
trial (determinism verified on trial-0, three runs).

## License

MIT — see `LICENSE`.
