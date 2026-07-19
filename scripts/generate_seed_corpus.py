"""Generate the PlantCortex demo corpus: "Prayag Petro Refinery - Unit 2".

A small but *internally consistent* fictional plant where the SAME equipment tags
(P-101A, V-201, E-301, ...) thread through every document type. That consistency is
what makes the multi-hop queries and the live graph demo light up.

Deterministic (seeded) so the hidden patterns are controlled and reproducible:
  * P-101A suffers 4 mechanical-seal failures over 5 years; 3 are preceded within
    30 days by a "high vibration" work order  -> the M6 Pattern Card / N3 story.
  * SOP-17 (Confined Space Entry) deliberately OMITS pre-entry gas testing that
    OISD-STD-105 cl. 7.3 requires  -> the M5 compliance-gap the agent must catch.
  * One P-101A inspection is overdue                -> the T2 multi-hop join.

Run:  python -m scripts.generate_seed_corpus
Output: data/seed/**  plus data/seed/manifest.json and data/seed/gold/gold_extraction.json

NOTE on regulatory docs: the OISD/Factories-Act files here are SYNTHETIC,
representative stand-ins written for the demo — not the official published text.
They are clearly labelled as such in-document.
"""

from __future__ import annotations

import json
import math
import random
import struct
import wave
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image, ImageDraw, ImageFont

from core.ontology import normalize_tag

# --------------------------------------------------------------------------- #
# Paths & determinism
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "data" / "seed"
RNG = random.Random(42)

TODAY = date(2026, 7, 9)
START = date(2021, 7, 1)            # ~5-year window
PLANT = "Prayag Petro Refinery - Unit 2"


# --------------------------------------------------------------------------- #
# The plant model (single source of consistency)
# --------------------------------------------------------------------------- #
EQUIPMENT = [
    {"tag": "P-101A", "name": "Feed Charge Pump A", "type": "pump",
     "location": "Pump House B"},
    {"tag": "P-101B", "name": "Feed Charge Pump B", "type": "pump",
     "location": "Pump House B"},
    {"tag": "V-201", "name": "Feed Surge Drum", "type": "vessel",
     "location": "Process Area N"},
    {"tag": "E-301", "name": "Feed/Effluent Exchanger", "type": "heat exchanger",
     "location": "Process Area N"},
    {"tag": "FV-112", "name": "Feed Flow Control Valve", "type": "control valve",
     "location": "Process Area N"},
    {"tag": "PT-108", "name": "Feed Header Pressure Transmitter", "type": "instrument",
     "location": "Process Area N"},
    {"tag": "T-401", "name": "Slop Storage Tank", "type": "tank",
     "location": "Tank Farm"},
    {"tag": "C-501", "name": "Off-gas Compressor", "type": "compressor",
     "location": "Compressor House"},
]
EQUIP_BY_TAG = {e["tag"]: e for e in EQUIPMENT}

# Components (PART_OF Equipment).
COMPONENTS = [
    ("P-101A", "mechanical seal"), ("P-101A", "bearing DE"),
    ("P-101A", "bearing NDE"), ("P-101A", "impeller"), ("P-101A", "coupling"),
    ("P-101B", "mechanical seal"), ("P-101B", "bearing DE"), ("P-101B", "impeller"),
    ("E-301", "tube bundle"), ("E-301", "channel gasket"),
    ("FV-112", "actuator"), ("FV-112", "positioner"),
    ("V-201", "relief valve"), ("V-201", "level gauge"),
    ("C-501", "bearing DE"), ("C-501", "mechanical seal"),
]

# P&ID process connectivity (CONNECTED_TO). Line tags L-20xx.
PID_LINES = [
    ("T-401", "P-101A", "L-2001"),
    ("T-401", "P-101B", "L-2002"),
    ("P-101A", "FV-112", "L-2003"),
    ("P-101B", "FV-112", "L-2004"),
    ("FV-112", "E-301", "L-2005"),
    ("E-301", "V-201", "L-2006"),
    ("V-201", "C-501", "L-2007"),
]

# Ontology-guided FMEA causal chain (HAS_CAUSE). Powers N3 cause ranking.
# (failure_mode_code, target)  where target is a FailureMode code or "<tag>::<component>"
HAS_CAUSE = [
    ("ELP", "P-101A::mechanical seal"),   # seal leak caused by the seal itself
    ("ELP", "VIB"),                        # ... and vibration precedes/causes it
    ("VIB", "P-101A::bearing DE"),         # vibration caused by DE bearing
    ("OHE", "P-101A::bearing DE"),         # overheating caused by bearing
    ("INL", "E-301::channel gasket"),      # internal leak caused by exchanger gasket
    ("STU", "FV-112::actuator"),           # valve stuck caused by actuator
]

TECHS = ["R. Sharma", "A. Khan", "S. Iyer", "M. Das", "P. Nair", "V. Reddy",
         "K. Menon", "J. Patel", "T. Bose", "N. Rao"]

FAILMODE_USED = {
    "ELP": "External leakage (process medium)",
    "INL": "Internal leakage",
    "VIB": "Vibration",
    "OHE": "Overheating",
    "NOI": "Noise",
    "STU": "Stuck / seized",
    "AIR": "Abnormal instrument reading",
    "PLU": "Plugged / choked",
}


# --------------------------------------------------------------------------- #
# Gold-set accumulator (we KNOW ground truth because we generate it)
# --------------------------------------------------------------------------- #
class Gold:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}

    def node(self, ntype: str, key: str, props: dict, doc_id: str) -> None:
        self.nodes.setdefault(key, {"type": ntype, "key": key, "props": props,
                                    "doc_ids": []})
        if doc_id not in self.nodes[key]["doc_ids"]:
            self.nodes[key]["doc_ids"].append(doc_id)

    def edge(self, etype: str, src: str, tgt: str, doc_id: str) -> None:
        k = f"{src}-[{etype}]->{tgt}"
        self.edges.setdefault(k, {"type": etype, "source": src, "target": tgt,
                                  "doc_ids": []})
        if doc_id not in self.edges[k]["doc_ids"]:
            self.edges[k]["doc_ids"].append(doc_id)

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"nodes": list(self.nodes.values()), "edges": list(self.edges.values())},
            indent=2), encoding="utf-8")


GOLD = Gold()
MANIFEST: list[dict] = []


def _equip_key(tag: str) -> str:
    return f"Equipment:{normalize_tag(tag)}"


def _comp_key(tag: str, name: str) -> str:
    return f"Component:{normalize_tag(tag)}:{name.strip().lower()}"


def _fm_key(code: str) -> str:
    return f"FailureMode:{code}"


def register_backbone() -> None:
    """Equipment / Component / FailureMode / FMEA nodes+edges into the gold set.

    These are attributed to the P&ID + the equipment register; downstream docs
    reference the same canonical keys, which is exactly the N1 story.
    """
    pid_doc = "PID-U2-001"
    for e in EQUIPMENT:
        GOLD.node("Equipment", _equip_key(e["tag"]),
                  {"tag": normalize_tag(e["tag"]), "name": e["name"],
                   "type": e["type"], "unit": "Unit 2", "location": e["location"]},
                  pid_doc)
    for tag, comp in COMPONENTS:
        GOLD.node("Component", _comp_key(tag, comp),
                  {"name": comp, "parent_equipment": normalize_tag(tag)}, pid_doc)
        GOLD.edge("PART_OF", _comp_key(tag, comp), _equip_key(tag), pid_doc)
    for code, name in FAILMODE_USED.items():
        GOLD.node("FailureMode", _fm_key(code), {"code": code, "name": name}, pid_doc)
    for code, target in HAS_CAUSE:
        if "::" in target:
            tag, comp = target.split("::", 1)
            tgt_key = _comp_key(tag, comp)
        else:
            tgt_key = _fm_key(target)
        GOLD.edge("HAS_CAUSE", _fm_key(code), tgt_key, pid_doc)
    for a, b, line in PID_LINES:
        GOLD.edge("CONNECTED_TO", _equip_key(a), _equip_key(b), pid_doc)


# --------------------------------------------------------------------------- #
# 1. P&ID drawing (matplotlib)
# --------------------------------------------------------------------------- #
# layout: (tag, x, y)  on a simple flow left->right
PID_LAYOUT = {
    "T-401": (1, 5), "P-101A": (3, 6.2), "P-101B": (3, 3.8), "FV-112": (5, 5),
    "E-301": (7, 5), "V-201": (9, 5), "PT-108": (5, 7), "C-501": (11, 5),
}


def _draw_symbol(ax, tag: str, x: float, y: float) -> None:
    etype = EQUIP_BY_TAG[tag]["type"]
    if etype == "pump" or etype == "compressor":
        ax.add_patch(plt.Circle((x, y), 0.45, fill=False, lw=2))
        ax.plot([x, x + 0.45], [y, y + 0.3], "k", lw=2)   # pump volute tick
    elif etype in ("vessel", "tank"):
        ax.add_patch(plt.Rectangle((x - 0.45, y - 0.8), 0.9, 1.6, fill=False, lw=2))
    elif etype == "heat exchanger":
        ax.add_patch(plt.Circle((x, y), 0.55, fill=False, lw=2))
        ax.plot([x - 0.55, x + 0.55], [y, y], "k", lw=1.5)
        ax.plot([x, x], [y - 0.55, y + 0.55], "k", lw=1.5)
    elif etype == "control valve":
        ax.plot([x - 0.4, x + 0.4, x - 0.4, x + 0.4, x - 0.4],
                [y - 0.3, y + 0.3, y + 0.3, y - 0.3, y - 0.3], "k", lw=2)  # bowtie
        ax.plot([x, x], [y + 0.3, y + 0.7], "k", lw=2)                     # actuator stem
        ax.add_patch(plt.Circle((x, y + 0.85), 0.18, fill=False, lw=2))
    elif etype == "instrument":
        ax.add_patch(plt.Circle((x, y), 0.35, fill=False, lw=2))
    ax.text(x, y - 1.0, tag, ha="center", va="top", fontsize=9, fontweight="bold")


def make_pnid() -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    for a, b, line in PID_LINES:
        (x0, y0), (x1, y1) = PID_LAYOUT[a], PID_LAYOUT[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="0.3"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, line, fontsize=7, color="0.4")
    # instrument signal (dashed) PT-108 -> FV-112
    ax.plot([PID_LAYOUT["PT-108"][0], PID_LAYOUT["FV-112"][0]],
            [PID_LAYOUT["PT-108"][1], PID_LAYOUT["FV-112"][1] + 0.9],
            "k--", lw=1)
    for tag, (x, y) in PID_LAYOUT.items():
        _draw_symbol(ax, tag, x, y)
    ax.set_title(f"{PLANT}   |   P&ID  PID-U2-001  Feed Section   (Rev 3)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(2, 9)
    ax.axis("off")
    out = SEED_DIR / "pnid" / "PID-U2-001.png"
    dpi = 150                              # save resolution (no tight bbox)
    fig.canvas.draw()
    render_dpi = fig.dpi                   # dpi that ax.transData actually uses
    scale = dpi / render_dpi              # map render-space px -> saved-image px
    render_h = fig.get_figheight() * render_dpi

    def to_px(x: float, y: float) -> tuple[float, float]:
        dx, dy = ax.transData.transform((x, y))       # render-space, origin bottom-left
        return float(dx * scale), float((render_h - dy) * scale)  # image top-left px

    fig.savefig(out, dpi=dpi)
    height_px = fig.get_figheight() * dpi

    cls_map = {"pump": "pump", "compressor": "compressor", "vessel": "vessel",
               "tank": "tank", "heat exchanger": "heat_exchanger",
               "control valve": "valve", "instrument": "instrument"}
    symbols, tags = [], []
    for tag, (x, y) in PID_LAYOUT.items():
        cx, cy = to_px(x, y)
        rx = abs(to_px(x + 0.6, y)[0] - cx)          # symbol radius in px
        symbols.append({"tag": tag, "cls": cls_map[EQUIP_BY_TAG[tag]["type"]],
                        "bbox": [cx - rx, cy - rx, cx + rx, cy + rx],
                        "center": [cx, cy]})
        tx, ty = to_px(x, y - 1.0)                    # tag label position
        w = 6 * len(tag)
        tags.append({"text": tag, "bbox": [tx - w, ty - 4, tx + w, ty + 20]})
    layer = {"image": "PID-U2-001.png",
             "width": int(fig.get_figwidth() * dpi), "height": int(height_px),
             "symbols": symbols, "tags": tags}
    (SEED_DIR / "pnid" / "PID-U2-001.layer.json").write_text(
        json.dumps(layer, indent=2), encoding="utf-8")
    plt.close(fig)
    MANIFEST.append({"doc_id": "PID-U2-001", "filename": "pnid/PID-U2-001.png",
                     "doc_type": "pnid", "source_kind": "image", "page_count": 1,
                     "layer_file": "pnid/PID-U2-001.layer.json"})


# --------------------------------------------------------------------------- #
# 2. Maintenance work orders (~600, XLSX)  -- with the seeded pattern
# --------------------------------------------------------------------------- #
GENERIC_WO = {
    "pump": [("brg noise, minor vib", "lubd brg, chkd algn"),
             ("gland leak minor", "tightnd gland, ok"),
             ("routine pm - vibn chk", "vibn within limit, cln done"),
             ("cplg guard loose", "cplg guard refitted")],
    "compressor": [("hi vib alarm", "chkd algn, lubd brg"),
                   ("oil leak minor", "tightnd flange, cln")],
    "vessel": [("relief vlv insp due", "relief vlv tested ok"),
               ("level gauge foggy", "cln level gauge")],
    "tank": [("routine insp", "no abnrml found"),
             ("vent choke susp", "cln vent, ok")],
    "heat exchanger": [("dp hi susp choke", "chkd, cln plan raised"),
                       ("channel gskt weep", "tightnd bolts")],
    "control valve": [("vlv stiff opn", "adj positioner"),
                      ("positioner drift", "recalib positioner")],
    "instrument": [("xmtr reading drift", "recalib xmtr"),
                   ("abnrml reading susp", "chkd, recalib")],
}


# Benign routine work — deliberately contains NO failure-mode keywords.
BENIGN_WO = [
    ("routine pm as per schedule", "completed, no abnrml found"),
    ("lube top-up", "greased, ok"),
    ("cln unit exterior", "cleaned"),
    ("6-monthly insp", "insp ok, within limit"),
    ("fastener/guard chk", "retightened, ok"),
    ("paint touch-up", "done"),
    ("tag/label refresh", "relabelled"),
    ("housekeeping round", "area cleaned"),
    ("condition monitoring reading", "reading normal, filed"),
    ("spares availability chk", "verified in store"),
]


def _rand_date(start: date, end: date) -> date:
    return start + timedelta(days=RNG.randint(0, (end - start).days))


def make_work_orders() -> None:
    rows: list[dict] = []
    wo_seq = 1000

    def add(dt: date, tag: str, wtype: str, prob: str, act: str) -> str:
        nonlocal wo_seq
        wo_seq += 1
        wo_id = f"WO-{wo_seq}"
        rows.append({"WO_ID": wo_id, "Date": dt.isoformat(), "Equipment_Tag": tag,
                     "Type": wtype, "Problem_Text": prob, "Action_Text": act,
                     "Technician": RNG.choice(TECHS),
                     "Downtime_Hrs": RNG.choice([0, 1, 2, 3, 4, 6, 8, 12]),
                     "Status": "closed"})
        return wo_id

    # --- SEEDED PATTERN: P-101A seal failures + vibration precursors ---
    # 4 seal failures spaced ~14 months apart; 3 preceded within 30 days by hi-vib.
    seal_dates = [START + timedelta(days=d) for d in (150, 560, 985, 1420)]
    precede_flags = [True, True, False, True]      # 3 of 4 preceded by hi-vib
    seeded_wo_ids: list[tuple[str, str]] = []       # (wo_id, kind)
    for sd, preceded in zip(seal_dates, precede_flags):
        if preceded:
            vib_dt = sd - timedelta(days=RNG.randint(10, 26))
            vid = add(vib_dt, "P-101A", "corrective",
                      "brg noise p101a hi vib, excsv vibn DE end",
                      "monitored, vibn trending up, seel chk advised")
            seeded_wo_ids.append((vid, "VIB"))
        wid = add(sd, "P-101A", "corrective",
                  "mech seel lkg p101a, seel failure, oil on baseplate",
                  "chng mech seel, algn chkd, brg inspd")
        seeded_wo_ids.append((wid, "ELP"))

    # A couple of preventive seal-related PMs on P-101A for realism.
    for d in (300, 730, 1100):
        add(START + timedelta(days=d), "P-101A", "preventive",
            "pm - seel & brg condition chk", "insp ok, lubd")

    # --- NOISE: ~590 generic WOs across all equipment ---
    # Mostly benign routine work (no failure signature) so the seeded failure patterns
    # stand out as statistically distinctive; ~25% carry a minor, type-typical issue.
    n_generic = 592
    end = TODAY - timedelta(days=1)
    for _ in range(n_generic):
        e = RNG.choice(EQUIPMENT)
        if RNG.random() < 0.72:
            prob, act = RNG.choice(BENIGN_WO)
            wtype = "preventive"
        else:
            prob, act = RNG.choice(GENERIC_WO[e["type"]])
            wtype = RNG.choices(["preventive", "corrective"], weights=[0.5, 0.5])[0]
        add(_rand_date(START, end), e["tag"], wtype, prob, act)

    rows.sort(key=lambda r: r["Date"])
    df = pd.DataFrame(rows)
    out = SEED_DIR / "work_orders" / "work_orders.xlsx"
    df.to_excel(out, index=False, engine="openpyxl")
    # also a CSV for quick eyeballing / eval convenience
    df.to_csv(SEED_DIR / "work_orders" / "work_orders.csv", index=False)

    doc_id = "DOC-WO-001"
    MANIFEST.append({"doc_id": doc_id, "filename": "work_orders/work_orders.xlsx",
                     "doc_type": "work_order", "source_kind": "xlsx",
                     "page_count": 1, "row_count": len(rows)})

    # Gold: the seeded WOs and their PERFORMED_ON / EXHIBITS edges.
    row_by_id = {r["WO_ID"]: i for i, r in enumerate(rows)}
    for wo_id, kind in seeded_wo_ids:
        GOLD.node("WorkOrder", f"WorkOrder:{wo_id}",
                  {"wo_id": wo_id, "type": "corrective",
                   "row": row_by_id[wo_id]}, doc_id)
        GOLD.edge("PERFORMED_ON", f"WorkOrder:{wo_id}", _equip_key("P-101A"), doc_id)
        GOLD.edge("EXHIBITS", f"WorkOrder:{wo_id}", _fm_key(kind), doc_id)

    print(f"  work orders: {len(rows)} rows "
          f"(seeded P-101A: {len(seeded_wo_ids)} incl. "
          f"{sum(1 for _, k in seeded_wo_ids if k=='ELP')} seal failures)")


# --------------------------------------------------------------------------- #
# 3. SOPs (fpdf2) -- SOP-17 has the deliberate compliance gap
# --------------------------------------------------------------------------- #
def _cell(pdf: FPDF, h: float, text: str) -> None:
    """Full-width multi_cell that always resets x to the left margin afterwards."""
    pdf.multi_cell(pdf.epw, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _pdf(path: Path, title: str, subtitle: str, blocks: list[tuple[str, str]]) -> int:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    _cell(pdf, 8, title)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    _cell(pdf, 6, subtitle)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    for heading, body in blocks:
        if heading:
            pdf.set_font("Helvetica", "B", 11)
            _cell(pdf, 7, heading)
        pdf.set_font("Helvetica", "", 10)
        _cell(pdf, 6, body)
        pdf.ln(1)
    pdf.output(str(path))
    return pdf.page_no()


SOPS = [
    dict(sop_id="SOP-05", rev="2", title="Centrifugal Pump Start-up and Shutdown",
         scope="Feed charge pumps P-101A and P-101B", governs=["P-101A", "P-101B"],
         requires=[("OISD-STD-105", "5.1")],
         steps=["Verify P-101A/P-101B suction from T-401 lined up via L-2001/L-2002.",
                "Obtain valid work permit per OISD-STD-105 clause 5.1 before any intrusive job.",
                "Confirm mechanical seal flush plan energised.",
                "Check bearing lubrication and vibration baseline before start.",
                "Start pump, observe discharge pressure at PT-108, ramp FV-112."]),
    dict(sop_id="SOP-09", rev="4", title="Lock-Out Tag-Out (LOTO) for Rotating Equipment",
         scope="All rotating equipment in Unit 2 (P-101A/B, C-501)",
         governs=["P-101A", "P-101B", "C-501"],
         requires=[("PESA-EL-11", "3.2")],
         steps=["Identify all energy sources; de-energise and isolate.",
                "Apply personal locks and tags per PESA-EL-11 clause 3.2.",
                "Verify zero energy before work commences."]),
    dict(sop_id="SOP-12", rev="1", title="Hot Work Permit Procedure",
         scope="Welding/grinding in Unit 2 process areas",
         governs=["E-301", "V-201"],
         requires=[("OISD-STD-105", "6.2")],
         steps=["Raise hot work permit per OISD-STD-105 clause 6.2.",
                "Post trained fire watch with extinguisher for the job duration.",
                "Gas test the area for flammable atmosphere before ignition source use.",
                "Cordon area and remove combustibles within 15 m."]),
    # ---- SOP-17: DELIBERATE GAP (no pre-entry gas / atmosphere testing) ----
    dict(sop_id="SOP-17", rev="1", title="Confined Space Entry Procedure",
         scope="Vessel and drum entry, incl. feed surge drum V-201",
         governs=["V-201"],
         requires=[("OISD-STD-105", "7.3")],
         steps=["Obtain confined space entry permit and identify entry supervisor.",
                "Isolate the vessel (V-201) and blind all process connections.",
                "Ventilate the vessel with forced air for at least 30 minutes.",
                "Post a standby attendant at the manway during entry.",
                "Log entry and exit of all personnel on the permit."]),
    dict(sop_id="SOP-21", rev="3", title="Pump Mechanical Seal Replacement",
         scope="Feed charge pumps P-101A and P-101B mechanical seals",
         governs=["P-101A", "P-101B"],
         requires=[("OISD-STD-105", "5.1")],
         steps=["Apply LOTO per SOP-09 and confirm zero energy.",
                "Drain and depressurise pump casing.",
                "Remove coupling, back-pull assembly, extract worn mechanical seal.",
                "Fit new seal; check DE/NDE bearing condition and shaft run-out.",
                "Re-align pump to driver and record vibration on restart."]),
    dict(sop_id="SOP-24", rev="2", title="Shell-and-Tube Heat Exchanger Cleaning",
         scope="Feed/effluent exchanger E-301", governs=["E-301"],
         requires=[("OISD-STD-105", "5.1")],
         steps=["Isolate and drain E-301; obtain work permit per OISD-STD-105 cl. 5.1.",
                "Remove channel cover; inspect channel gasket condition.",
                "Hydro-jet tube bundle; record differential pressure improvement."]),
    dict(sop_id="SOP-30", rev="1", title="Field Instrument Calibration",
         scope="Feed section instruments PT-108 and FV-112 positioner",
         governs=["PT-108", "FV-112"],
         requires=[("OISD-STD-105", "5.1")],
         steps=["Raise permit; place loop under maintenance.",
                "Apply 5-point calibration to PT-108; record as-found/as-left.",
                "Stroke FV-112 and calibrate positioner 0-100%."]),
]


def make_sops() -> None:
    for sop in SOPS:
        blocks = [
            ("", f"{PLANT}"),
            ("1. Scope", sop["scope"]),
            ("2. Applicable Equipment", ", ".join(sop["governs"])),
            ("3. Referenced Standards",
             "; ".join(f"{s} clause {c}" for s, c in sop["requires"])),
            ("4. Procedure",
             "\n".join(f"{i+1}. {s}" for i, s in enumerate(sop["steps"]))),
        ]
        fname = f"sops/{sop['sop_id']}.pdf"
        pages = _pdf(SEED_DIR / fname,
                     f"{sop['sop_id']}  {sop['title']}  (Rev {sop['rev']})",
                     f"Standard Operating Procedure - {PLANT}", blocks)
        MANIFEST.append({"doc_id": sop["sop_id"], "filename": fname,
                         "doc_type": "sop", "source_kind": "pdf", "page_count": pages})
        # Gold
        GOLD.node("Procedure", f"Procedure:{sop['sop_id']}",
                  {"sop_id": sop["sop_id"], "title": sop["title"],
                   "revision": sop["rev"]}, sop["sop_id"])
        for tag in sop["governs"]:
            GOLD.edge("GOVERNS", f"Procedure:{sop['sop_id']}", _equip_key(tag),
                      sop["sop_id"])
        for std, cl in sop["requires"]:
            GOLD.edge("REQUIRES", f"Procedure:{sop['sop_id']}",
                      f"RegulatoryClause:{std}:{cl}", sop["sop_id"])
    print(f"  SOPs: {len(SOPS)} (SOP-17 confined-space gap embedded)")


# --------------------------------------------------------------------------- #
# 4. Regulatory corpus (synthetic representative clauses)
# --------------------------------------------------------------------------- #
DISCLAIMER = ("SYNTHETIC / representative text prepared for the PlantCortex demo. "
              "This is NOT the official published standard text.")

REGS = [
    dict(std="OISD-STD-105", title="Work Permit System",
         clauses=[
             ("5.1", "A valid work permit shall be obtained and displayed before any "
              "maintenance, intrusive or non-routine job on process equipment."),
             ("6.2", "For hot work, a fire watch shall be posted and the area gas "
              "tested for flammable atmosphere prior to and during the activity."),
             ("7.3", "Prior to confined space entry, the atmosphere shall be tested "
              "for oxygen (19.5-23.5%), flammable gas (<10% LEL) and toxic gases "
              "(H2S/CO). Gas testing shall be repeated at defined intervals during "
              "entry. Entry is prohibited until acceptable readings are recorded."),
         ],
         covers=["P-101A", "P-101B", "V-201", "E-301", "C-501"]),
    dict(std="FACT-ACT", title="Factories Act 1948 - Confined Space Precautions (excerpt)",
         clauses=[
             ("36", "No person shall enter any confined space in which dangerous "
              "fumes are likely to be present unless the space has been tested and "
              "certified safe, or breathing apparatus is worn."),
         ],
         covers=["V-201"]),
    dict(std="PESA-EL-11", title="Electrical Isolation and Lock-Out Guidance",
         clauses=[
             ("3.2", "All electrical energy sources shall be isolated, locked and "
              "tagged, and verified at zero energy before work on rotating machinery."),
         ],
         covers=["P-101A", "P-101B", "C-501"]),
    dict(std="OISD-STD-106", title="Process Design and Operating Safety (excerpt)",
         clauses=[
             ("4.4", "Pressure relief devices on vessels shall be inspected and "
              "function-tested at intervals not exceeding the statutory period."),
         ],
         covers=["V-201"]),
    dict(std="OISD-GDN-176", title="Mechanical Integrity and Inspection (excerpt)",
         clauses=[
             ("8.1", "Rotating equipment shall be subject to periodic condition "
              "monitoring; overdue inspections shall be tracked and closed out."),
         ],
         covers=["P-101A", "P-101B", "E-301", "C-501"]),
]


def make_regulatory() -> None:
    for reg in REGS:
        blocks = [("", DISCLAIMER)]
        for cl, text in reg["clauses"]:
            blocks.append((f"Clause {cl}", text))
        blocks.append(("Applicability (demo annotation)",
                       "Referenced equipment: " + ", ".join(reg["covers"])))
        fname = f"regulatory/{reg['std']}.pdf"
        pages = _pdf(SEED_DIR / fname, f"{reg['std']}  {reg['title']}",
                     "Regulatory standard (synthetic) - PlantCortex demo corpus", blocks)
        MANIFEST.append({"doc_id": reg["std"], "filename": fname,
                         "doc_type": "regulatory", "source_kind": "pdf",
                         "page_count": pages})
        for cl, text in reg["clauses"]:
            GOLD.node("RegulatoryClause", f"RegulatoryClause:{reg['std']}:{cl}",
                      {"standard": reg["std"], "clause_no": cl,
                       "text_summary": text[:80]}, reg["std"])
            for tag in reg["covers"]:
                GOLD.edge("COVERS", f"RegulatoryClause:{reg['std']}:{cl}",
                          _equip_key(tag), reg["std"])
    print(f"  regulatory: {len(REGS)} standards (OISD-STD-105 cl.7.3 = gas-test rule)")


# --------------------------------------------------------------------------- #
# 5. Incident & near-miss reports (fpdf2)
# --------------------------------------------------------------------------- #
INCIDENTS = [
    dict(id="INC-2022-014", date=START + timedelta(days=152), sev="near-miss",
         tag="P-101A", fm="ELP",
         desc="Mechanical seal on feed charge pump P-101A failed releasing a small "
              "quantity of hot hydrocarbon onto the baseplate. No ignition. Operators "
              "had reported high bearing vibration on P-101A three weeks earlier."),
    dict(id="INC-2023-006", date=START + timedelta(days=562), sev="minor",
         tag="P-101A", fm="ELP",
         desc="Repeat mechanical seal leak on P-101A during running. Pump tripped on "
              "low flow. Spill contained. Vibration had been trending high on the DE "
              "bearing prior to failure."),
    dict(id="INC-2025-003", date=START + timedelta(days=1422), sev="minor",
         tag="P-101A", fm="ELP",
         desc="P-101A seal failure recurred; oil mist observed. Root cause linked to "
              "DE bearing degradation and elevated vibration preceding the event."),
    # A confined-space near-miss at V-201 that reinforces the SOP-17 gap story.
    dict(id="INC-2024-011", date=START + timedelta(days=1050), sev="near-miss",
         tag="V-201", fm="AIR",
         desc="Entrant into feed surge drum V-201 reported dizziness shortly after "
              "entry. No pre-entry gas test had been recorded on the permit. Entrant "
              "was withdrawn; subsequent test showed low oxygen. Highlights confined "
              "space entry procedure shortfall."),
    dict(id="INC-2022-021", date=START + timedelta(days=300), sev="minor",
         tag="E-301", fm="INL",
         desc="Internal leakage suspected across E-301 causing off-spec feed "
              "temperature. Channel gasket found weeping."),
    dict(id="INC-2023-018", date=START + timedelta(days=820), sev="near-miss",
         tag="C-501", fm="OHE",
         desc="Off-gas compressor C-501 bearing overheating alarm; unit tripped "
              "before damage. Lubrication starvation suspected."),
    dict(id="INC-2021-009", date=START + timedelta(days=60), sev="minor",
         tag="FV-112", fm="STU",
         desc="Feed flow control valve FV-112 stuck partially open; actuator "
              "sluggish. Manual intervention required to stabilise flow."),
    dict(id="INC-2024-002", date=START + timedelta(days=980), sev="minor",
         tag="PT-108", fm="AIR",
         desc="Feed header pressure transmitter PT-108 gave abnormal high reading "
              "leading to spurious FV-112 movement. Transmitter recalibrated."),
    dict(id="INC-2025-010", date=START + timedelta(days=1500), sev="near-miss",
         tag="T-401", fm="PLU",
         desc="Slop tank T-401 vent partially choked; slight pressure build-up "
              "detected during routine round."),
    dict(id="INC-2022-030", date=START + timedelta(days=410), sev="minor",
         tag="P-101B", fm="NOI",
         desc="Feed charge pump P-101B abnormal noise from DE bearing; standby pump "
              "swapped in. No release."),
]


def make_incidents() -> None:
    for inc in INCIDENTS:
        blocks = [
            ("", f"{PLANT}"),
            ("Incident ID", inc["id"]),
            ("Date", inc["date"].isoformat()),
            ("Severity", inc["sev"]),
            ("Equipment", inc["tag"]),
            ("Description", inc["desc"]),
        ]
        fname = f"incidents/{inc['id']}.pdf"
        pages = _pdf(SEED_DIR / fname, f"Incident Report {inc['id']}",
                     f"Incident / Near-miss - {PLANT}", blocks)
        MANIFEST.append({"doc_id": inc["id"], "filename": fname,
                         "doc_type": "incident", "source_kind": "pdf",
                         "page_count": pages})
        GOLD.node("Incident", f"Incident:{inc['id']}",
                  {"incident_id": inc["id"], "date": inc["date"].isoformat(),
                   "severity": inc["sev"]}, inc["id"])
        GOLD.edge("OCCURRED_AT", f"Incident:{inc['id']}", _equip_key(inc["tag"]),
                  inc["id"])
        GOLD.edge("EXHIBITS", f"Incident:{inc['id']}", _fm_key(inc["fm"]), inc["id"])
    print(f"  incidents: {len(INCIDENTS)} (3 tied to P-101A seal pattern; "
          f"1 confined-space near-miss at V-201)")


# --------------------------------------------------------------------------- #
# 6. Inspection records -- scanned-style (image-in-PDF) to exercise OCR path
# --------------------------------------------------------------------------- #
INSPECTIONS = [
    # (tag, last_date, interval_months)  -- P-101A one is OVERDUE
    ("P-101A", date(2025, 3, 10), 6),    # next due 2025-09-10 -> OVERDUE vs 2026-07
    ("V-201", date(2026, 2, 1), 12),
    ("E-301", date(2026, 4, 15), 12),
    ("PT-108", date(2026, 5, 20), 6),
    ("T-401", date(2026, 1, 5), 12),
]


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    return date(y, m % 12 + 1, min(d.day, 28))


def make_inspections() -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small = font
    for i, (tag, last, interval) in enumerate(INSPECTIONS, 1):
        next_due = _add_months(last, interval)
        overdue = next_due < TODAY
        insp_id = f"INSP-{i:03d}"
        img = Image.new("RGB", (1000, 640), "white")
        d = ImageDraw.Draw(img)
        d.text((40, 30), f"{PLANT}", fill="black", font=font)
        d.text((40, 70), f"EQUIPMENT INSPECTION RECORD  {insp_id}", fill="black",
               font=font)
        lines = [
            f"Equipment Tag : {tag}",
            f"Equipment     : {EQUIP_BY_TAG[tag]['name']}",
            f"Inspection    : Periodic condition monitoring",
            f"Last Inspected: {last.isoformat()}",
            f"Interval      : {interval} months",
            f"Next Due      : {next_due.isoformat()}",
            f"Status        : {'OVERDUE' if overdue else 'OK / within period'}",
            f"Inspector     : {RNG.choice(TECHS)}",
        ]
        for j, ln in enumerate(lines):
            d.text((60, 150 + j * 46), ln, fill=(20, 20, 20), font=small)
        # a faint 'scanned' rotation to force the OCR path later
        img = img.rotate(RNG.uniform(-0.8, 0.8), fillcolor="white", expand=False)
        fname = f"inspections/{insp_id}.pdf"
        img.save(SEED_DIR / fname, "PDF", resolution=150)
        MANIFEST.append({"doc_id": insp_id, "filename": fname,
                         "doc_type": "inspection", "source_kind": "image",
                         "page_count": 1, "overdue": overdue, "equipment": tag})
    print(f"  inspections: {len(INSPECTIONS)} scanned-style "
          f"(P-101A OVERDUE for T2 join)")


# --------------------------------------------------------------------------- #
# 7. Knowledge-capture voice note (transcript + placeholder WAV)
# --------------------------------------------------------------------------- #
VOICE_TRANSCRIPT = (
    "This is Ramesh, senior operator, Unit 2. A word on feed charge pump P-101A. "
    "Whenever you hear the DE bearing getting noisy and you see the vibration "
    "climbing, that mechanical seal is about to go. We have lost the seal on "
    "P-101A three or four times over the years, and every single time the "
    "vibration warned us a few weeks before it leaked. So please trend the "
    "vibration on P-101A, and if it starts rising, plan a seal change at the next "
    "shutdown instead of waiting for it to fail."
)


def make_voice_note() -> None:
    note_id = "VN-001"
    txt = SEED_DIR / "voice" / f"{note_id}.txt"
    txt.write_text(VOICE_TRANSCRIPT, encoding="utf-8")
    # Placeholder WAV (a short tone). Real audio/TTS can replace this later;
    # the transcript is the substantive content for the demo.
    wav_path = SEED_DIR / "voice" / f"{note_id}.wav"
    framerate, dur = 8000, 2
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        frames = bytearray()
        for n in range(framerate * dur):
            val = int(3000 * math.sin(2 * math.pi * 300 * n / framerate))
            frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))
    MANIFEST.append({"doc_id": note_id, "filename": f"voice/{note_id}.wav",
                     "doc_type": "voice_note", "source_kind": "voice",
                     "page_count": 0, "transcript_file": f"voice/{note_id}.txt"})
    GOLD.node("TacitNote", f"TacitNote:{note_id}",
              {"note_id": note_id, "author_role": "Senior Operator"}, note_id)
    GOLD.edge("ABOUT", f"TacitNote:{note_id}", _equip_key("P-101A"), note_id)
    GOLD.edge("ABOUT", f"TacitNote:{note_id}", _fm_key("ELP"), note_id)
    GOLD.edge("ABOUT", f"TacitNote:{note_id}", _fm_key("VIB"), note_id)
    print("  voice note: VN-001 (P-101A tacit knowledge; WAV is a placeholder tone)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    for sub in ("pnid", "work_orders", "sops", "incidents", "regulatory",
                "inspections", "voice", "gold"):
        (SEED_DIR / sub).mkdir(parents=True, exist_ok=True)

    print(f"Generating seed corpus -> {SEED_DIR}")
    register_backbone()
    make_pnid()
    make_work_orders()
    make_sops()
    make_regulatory()
    make_incidents()
    make_inspections()
    make_voice_note()

    (SEED_DIR / "manifest.json").write_text(
        json.dumps({"plant": PLANT, "generated": datetime.now().isoformat(),
                    "documents": MANIFEST}, indent=2), encoding="utf-8")
    GOLD.dump(SEED_DIR / "gold" / "gold_extraction.json")

    n_docs = len(MANIFEST)
    print(f"\nDone. {n_docs} document sources; "
          f"gold set = {len(GOLD.nodes)} nodes, {len(GOLD.edges)} edges.")
    print(f"Manifest: data/seed/manifest.json")
    print(f"Gold:     data/seed/gold/gold_extraction.json")


if __name__ == "__main__":
    main()
