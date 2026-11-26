# Evaluated champion release (#109)

This is the release consumer for #29–30 and #38–40. It does not evaluate,
register, choose, or promote a model. It resolves `champion` to a registry
version and source run, validates upstream evaluation evidence, publishes that
run's exact serving bundle, and calls the existing
`src.serving.stage_artifacts upload --deploy` path into Oracle staging (#92).

## Dependency status

Inspected `main` at `0af67d0`, issues #109, #29–30 and #38–40, and merged PRs
#99, #107, #108 and #110. The current trainer logs training statistics and
`model/movielens-v1.joblib`, with a `model_version` tag and dataset provenance.
It does not log held-out selection results or create a registered version.
The notebook computes Precision@10, Recall@10 and NDCG@10 using a chronological
holdout; it does not implement a registry promotion or an incumbent retention
rule. No registry producer/loader exists on this main, and #38–40 have no
assignee. No new owner is assigned here.

The primary metric, K, retention/tie rule, and registry/promotion ownership
remain dependencies. The contract below is a proposed, tested handoff for
those tasks, not a claim that the evaluator already emits it. Keep
`CHAMPION_RELEASE_ENABLED` unset until the producers and project decisions
are agreed. A missing decision, missing metric, failed run, or provenance
mismatch fails before model publication or deployment.

The existing deployed candidate is from run
`642458df9a7a4369a352845619dd0c09`; its artifact is
`model/movielens-v1.joblib`. The separate candidate-copy branch uses
`models/candidates/642458df9a7a4369a352845619dd0c09/movielens-v1.joblib`.
Neither this run nor a completed training run is evidence of a winning
held-out comparison. Do not bootstrap a promotion from that fact. Development
and release tests use fixture bundles and local MLflow, with no live alias,
production object, or VM changes.

## Proposed producer contract

The registry producer registers the **exact raw `.joblib`**, rather than a
pyfunc directory or a newly serialized model. Its immutable version has:

- `run_id`: source training run (must be `FINISHED`).
- `source`: `runs:/<run_id>/<artifact_path>` or
  `<source_run.info.artifact_uri>/<artifact_path>`.
- Version tag `release.evaluation_run_id`: a separate, finished evaluation run.
- Alias `champion`: set by the upstream producer only after evaluation.

The evaluation run logs the agreed primary metric to MLflow metrics and writes
`release/decision.json`. Schema v1 has these fields (metavariables are filled
by the producer; the consumer supplies no metric or policy default):

| Field | Required value |
| --- | --- |
| `schema_version` | `1` |
| `decision` | `promoted` or `retained` |
| `registered_model` | Exact registry model name |
| `registry_version` | Exact version resolved from `champion`, as a string |
| `source_run_id` | Exact source training run ID |
| `model_artifact` | Canonical run-relative `.joblib` path |
| `model_sha256` | SHA-256 of the evaluated serving bundle's bytes |
| `bundle_version` | Original bundle `model_version`, equal to source run's tag |
| `evaluation.primary_metric` | Name of the agreed metric logged on the evaluation run |
| `evaluation.k` | Agreed positive integer K |
| `evaluation.retention_rule` | Agreed policy/rule reference or description |
| `evaluation.heldout_sha256` | SHA-256 identifying the shared held-out data |
| `evaluation.candidate_run_ids` | Evaluated source run IDs, including this winner |

A `retained` decision only needs the schema, decision and registry/run identity
fields and causes no storage writes or deployment. Usually retaining a
champion simply leaves its existing version/decision unchanged, which also
skips publication once its deployment receipt exists. The consumer validates
evidence and identity; correctness of candidate comparison, tie handling and
the selection decision belongs to the upstream evaluator.

The source run must retain #108's `dataset_uri`, `dataset_sha256`,
`dataset_manifest_sha256`, and `model_version` tags. The consumer verifies that
exact manifest and its ratings checksum, and uses **that manifest's
`movies.csv`**, including its SHA-256 and byte size. There is no catalog default,
latest-dataset lookup, or fallback to v1. A run predating these tags needs an
upstream provenance solution; this consumer does not guess one.

## Operation and records

After agreement, set repository variable `CHAMPION_REGISTERED_MODEL` to the
agreed registered name and `CHAMPION_RELEASE_ENABLED=true`. The **Release
evaluated champion** workflow polls every 15 minutes and can be manually
retried. It never runs on a training completion event. GitHub may delay
scheduled jobs; this is polling, not an MLflow webhook. Enable it only after
this PR's workflow inputs have reached main.

It uses the existing B2 variables/secrets and authenticated MLflow variables
and password from #108. `GITHUB_TOKEN` has `actions: write` for the existing
deploy dispatch. The publisher is serialized with the `champion-release`
concurrency group; Oracle staging retains its existing deployment queue and
#110's backend recreation/permissions fixes.

The release ID is SHA-256 of the canonical JSON registry name, registry
version and source run identity. Keys are:

```text
models/champions/<release_id>/v<registry_version>/<source_run_id>/model.joblib
models/champions/<release_id>/v<registry_version>/<source_run_id>/release.json
models/champions/<release_id>/v<registry_version>/<source_run_id>/deployment.json
```

The model's bytes and internal version are preserved. Registry version and
source run distinguish releases even though the current trainer writes the
same internal `movielens-v1` label. The catalog stays at its immutable processed
dataset key. `release.json` locks the intended provenance before uploading;
its presence alone does **not** mean the upload or deployment succeeded.
`deployment.json` records the verified handoff, including registry identity,
evaluation decision/digest, model/catalog keys and checksums, source run, and
the actual Actions deployment run ID/URL. A dispatch receipt records acceptance,
not deployment success: use that run's checks and conclusion to confirm rollout.

The deploy dispatch passes exact keys, expected bundle version, run ID,
model/catalog SHA-256 and release ID. Staging checks those expected bytes
before unpickling, then retains its validation, transfer checksum checks and
backend health checks. The VM also checks `/recommend/1?top_k=3` for a nonempty
response, finite scores, the expected version and fallback disabled. The release
consumer requires user 1 to exist in the source bundle's history.

## Retrying

- Existing objects are never overwritten by retries. `--reuse-verified` reads
  back the full object and checks its checksum and size against the same source
  bytes. Different bytes at an existing key fail closed. Default upload-only
  behavior still refuses an existing key.
- If upload succeeds and dispatch fails, retry the release workflow. It reuses
  verified objects and finishes the dispatch. A corrupt model/catalog or
  changed provenance prevents dispatch.
- If dispatch was accepted but saving its receipt failed, the retry finds the
  exact `Oracle staging <release_id>` title across main's dispatch run history,
  verifies the model again, and writes the missing receipt. An API lookup error
  prevents sending a new dispatch. A short visibility timeout fails the attempt;
  the next poll reconciles it. GitHub offers no atomic dispatch-and-receipt
  transaction, so an exceptionally delayed/invisible accepted run can still
  result in a duplicate dispatch; publication remains immutable.
- An existing receipt skips upload and dispatch. If its deployment run failed,
  rerun that recorded Actions run after fixing the cause; the release poll does
  not create a new deployment for every failed rollout. Never edit release
  records or repoint an already released version to different provenance.
- Republishing an old champion after a later champion/rollback requires a new
  evaluated registry version. Do not use alias rewinds to request a rollout of
  an already receipted version. External/manual deployments are not inferred
  from the registry alias.

Fixture validation covers the real local MLflow registry/alias/artifact APIs,
unchanged/retained champions, provenance and checksum failures, upload and
dispatch retries, receipt recovery, alias changes during preparation, exact
workflow inputs, and a known-user FastAPI response. No live rollout was used
as a development test. With #88, the first authorized rollout still needs its
recorded Actions result and live HTTPS health/recommendation verification.
