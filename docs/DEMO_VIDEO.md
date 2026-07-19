# PlantCortex — 3-Minute Demo Video Shot List

Target: **3:00**. Record screen at 1080p, 30fps. Keep the cursor calm; pre-open all tabs.
Record the risky live actions (drag-ingest, PDF export) **twice** and keep the clean take.

**Pre-flight**
- `python -m scripts.seed_demo` (fresh graph + reserve files staged).
- `python -m uvicorn api.main:app --port 8000` — leave running.
- Tabs open: desktop `http://localhost:8000`, mobile view `http://localhost:8000/field`
  (use browser device-toolbar at ~390px width), the deck `web/deck.html`.
- Set `DEMO_STRICT=1` so nothing needs the network mid-record.
- File manager open at `data/seed/reserve/` for the drag.

---

| # | Time | Screen | Voiceover (say this) | Capture notes |
|---|------|--------|----------------------|---------------|
| 1 | 0:00–0:15 | Deck slide 1 → 3 | "A plant already knows everything about its equipment — but that knowledge is scattered across drawings, spreadsheets, PDFs, and the heads of people about to retire. PlantCortex fuses it into one brain." | Slow zoom on the P-101A convergence slide. |
| 2 | 0:15–0:35 | Desktop `/` header | "One knowledge graph: 1,304 nodes, 100% of equipment linked across at least three document types." | Cursor traces the header stat + the live graph pane. |
| 3 | 0:35–0:55 | Drag `reserve_work_orders.xlsx` onto the app | "Ingestion is live. I drop in a new work-order sheet and the graph grows in real time — a brand-new pump appears." | Show the websocket toast + node count ticking up. Two takes. |
| 4 | 0:55–1:25 | Type the T2 question in chat | "Now the question plain search can't answer: which pump had seal failures **and** an overdue inspection **and** a near-miss? PlantCortex joins it across documents and shows the reasoning path — every claim cited." | Click a citation → source viewer opens the exact page. |
| 5 | 1:25–1:40 | Toggle `baseline`, re-ask | "Flip to a vector-only baseline — the same question degrades to a flat lookup with no path. That gap **is** the graph's value." | Put the two answers side by side if possible. |
| 6 | 1:40–2:05 | `/field` → Compliance tab → Scan | "On the field app: scan OISD-STD-105. It flags that the confined-space procedure lets people enter **without the gas test the standard requires** — and exports an audit-ready evidence pack." | Click Export → show the red-GAP PDF. |
| 7 | 2:05–2:25 | `/field` → Capture tab → dictate/type a note | "A retiring operator records what they know about P-101A. Seconds later it's a citable source in the graph." | Then Lookup P-101A → the note appears in the dossier. |
| 8 | 2:25–2:45 | Desktop → ⚡ Failure patterns chip | "And it doesn't just answer — it anticipates. It learned that vibration precedes P-101A's seal failures, so a **new** seal-leak work order is met with its probable cause." | Linger on the pattern card + the N3 cause line. |
| 9 | 2:45–3:00 | Deck final slide | "76% extraction recall, hybrid beats baseline where questions get hard, compliance one-for-one, sub-second answers, 81 tests — and it all runs offline. The plant already knows. Now it can tell you." | End card with the three novelties + repo/URL. |

---

**Editing**
- Add lower-third labels for each novelty as it appears: `N1 · cross-modal`, `N2 · multi-hop`, `N3 · anticipate`.
- Keep a subtle keyboard-tick / UI sound on the ingest toast and the PDF export — they're the "wow" moments.
- No music under the voiceover's key lines (beats 4–6); light ambient bed elsewhere.
- If you overrun, cut beat 2 (it's the softest); protect 3, 5, 6, 8.

**Backup if something breaks on camera**
- Pre-record beat 3 and beat 6 as isolated clips to splice in.
- The whole flow works under `DEMO_STRICT=1`, so wifi loss mid-record is survivable.
