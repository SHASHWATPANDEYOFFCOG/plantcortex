# PlantCortex — Judge Q&A Prep

Crisp, honest answers to the questions judges are most likely to probe. Lead with the
answer, then the evidence. Never bluff — the honest framing is stronger than a dodge.

---

## The hard ones (prepare these cold)

**Q. "The LLM is quota-blocked — so is any of this actually AI?"**
> The intelligence is in the **architecture**, not a single API call. Cross-modal entity
> resolution, Personalized-PageRank multi-hop retrieval, GraphRAG community detection,
> compliance obligation-matching, and causal-edge mining all run **deterministically and
> offline** — that's what you're seeing. The LLM is one *pluggable* stage (extraction
> polish, answer prose, VLM tag-reading, voice ASR); it's wired, cached, and lights up
> with zero code changes when a quota'd key is present. We deliberately built it so a
> flaky network can't kill the demo — which is exactly what a plant control room needs.
> *(The provided free-tier key hit its rate limit; we proved the LLM path works — the
> incident-extraction and JSON-mode calls succeeded before the quota ran out.)*

**Q. "Isn't this just RAG / chat-with-your-PDFs?"**
> No — and we can show it live with the `baseline` toggle. Plain vector RAG **can't**
> answer "which pump had seal failures AND an overdue inspection AND a near-miss," because
> that answer exists in no single chunk. Our graph joins it via PPR and shows the
> reasoning path. Flip baseline on and the same question degrades. Three things RAG
> doesn't do: (1) resolve the same asset across a *drawing, spreadsheet, and clause* into
> one node, (2) reason over *typed relationships*, (3) turn history into a *causal model*
> that anticipates failure.

**Q. "The P&ID tags come from a sidecar file — isn't that cheating?"**
> The **structural work — reading which equipment connects to which — is done from the
> raw pixels** with OpenCV (7/7 process lines recovered, and it correctly rejects the
> dashed instrument signal). Tag *identities* come from the drawing's text/vector layer,
> which real vector P&IDs genuinely expose — that's standard in P&ID digitization tools.
> Scanned drawings use the VLM path instead. We separated the two deliberately, following
> the 2025–26 benchmark finding that raw VLMs are unreliable on diagram detail.

**Q. "Your extraction recall is 76%, not 100%. Why?"**
> 100% on every core type — Equipment, FailureMode, Incident, Procedure, RegulatoryClause,
> WorkOrder, TacitNote. The 76% is dragged down by `Component`, `HAS_CAUSE`, and `PART_OF`,
> which by design come from P&ID component detection and the M6 causal miner, not the
> offline text pass. We **report that honestly in the eval** rather than hiding it — and
> the miner does learn `HAS_CAUSE` (vibration→seal) from the data, which is the N3 story.

**Q. "It's a synthetic corpus. Would it work on real plant documents?"**
> The corpus is synthetic but *structurally faithful* — jargon-dense shorthand work
> orders, OISD-style clause structure, scanned inspection images, a real P&ID render. The
> pipeline is format-driven, not memorized: swap in real PDFs and the same parse → TLP →
> extract → resolve path runs. The ontology and the `GraphRepo`/`VectorStore`/`LLMClient`
> interfaces are production-shaped. Next step is a pilot on a de-identified real unit.

---

## Technical depth

**Q. Why NetworkX instead of Neo4j?**
> Demo reliability on a machine without Docker — no external service to crash on stage.
> It's behind a `GraphRepo` interface; switching to Neo4j is a drop-in with no caller
> changes (the `docker-compose.yml` path is included). PPR is `networkx.pagerank` today;
> Neo4j GDS has the same primitive at scale.

**Q. How does the multi-hop retrieval actually work?**
> HippoRAG-style: we link query entities to graph nodes, run **Personalized PageRank**
> seeded from them, gather the chunks those top nodes are `MENTIONS`-linked to, and
> re-rank with lexical relevance so the *join* answer isn't buried under an asset's
> routine work orders. The answer returns the graph path so the reasoning is auditable.

**Q. How do you prevent hallucination?**
> Three guards: (1) every answer must carry citations from retrieved chunks; (2) a
> graceful **refusal** template fires when support is weak — we demo an out-of-corpus
> question being declined; (3) extraction is **ontology-constrained** — the model may
> only emit our node/edge types and may never invent an equipment tag.

**Q. How is the compliance gap detected — is it hardcoded?**
> No. Each clause's obligation is scored against the procedure that **REQUIRES** it,
> weighted toward safety-critical terms (gas/atmosphere/oxygen). SOP-17 scores 0% on
> clause 7.3's critical terms → GAP; 5.1 and 6.2 score high → covered. Zero false
> positives across the standard. Change the SOP text and the verdict changes.

**Q. What about scanned and handwritten documents?**
> Three-tier answer. (1) *Scanned-but-printed* (our inspection records): the pipeline
> rasterizes and routes them to the OCR/VLM path — in this build, Tesseract isn't
> installed, so inspection facts thread from structured metadata and we say so openly.
> (2) *Handwritten log books*: today's frontier VLMs read clean handwriting usefully but
> unreliably for compliance-grade facts — our design rule is that low-confidence
> extractions enter the graph flagged `confidence<0.5` and are excluded from compliance
> verdicts; a human-review queue is the production pattern. (3) The provenance model is
> exactly what makes this safe: every fact carries its extractor + confidence, so a
> handwritten-sourced claim can never silently masquerade as a verified one.

**Q. Where do personnel fit — do you track who did the work?**
> Yes — technicians on work orders become `Person` nodes linked via chunk mentions, so
> "which jobs did R. Sharma close on P-101A" is a graph query. We keep people synthetic/
> anonymized in the demo deliberately; in production this joins the HR/CMMS identity.

**Q. What's the cost model at scale?**
> Ingestion is a one-time LLM cost, cached to disk. Retrieval PPR + BM25 + local
> embeddings are ~free per query. Answer synthesis is the only per-query LLM call, and
> `DEMO_STRICT` proves it's optional. On-prem OSS models make marginal cost effectively
> zero for data-sovereign plants.

---

## Business & scale

**Q. What's the ROI / who buys this?**
> Reliability & maintenance engineers and plant safety/compliance leads. Value: minutes→
> seconds on cross-document questions, compliance gaps caught before an auditor finds
> them, and captured tacit knowledge that would otherwise retire. In a refinery, one
> avoided unplanned pump trip pays for the deployment.

**Q. How does it scale to a whole refinery / multiple plants?**
> Interfaces, not rewrites: `GraphRepo`→Neo4j, `VectorStore`→Qdrant, `LLMClient`→on-prem.
> The graph is namespaced per unit; communities and PPR scale sub-linearly with good
> indexing. Connectors to SAP-PM/Maximo (work orders) and PI (live vibration) replace the
> seed generator with real feeds.

**Q. Data sovereignty — plants won't send data to a cloud LLM.**
> Correct, and we designed for it. Embeddings already default to a **local** model; the
> `LLMClient` abstraction swaps a cloud API for an on-site Llama/Qwen with no other
> change. Everything can run air-gapped.

---

## If you're stuck / buying time
- "Great question — the honest answer is…" (judges reward candor over spin).
- Redirect to a live proof: "Let me *show* you rather than tell you" → baseline toggle,
  refusal, or the compliance PDF.
- Know your three numbers cold: **1,304-node graph · 100% linkage · compliance 1/1, 0 FP.**
