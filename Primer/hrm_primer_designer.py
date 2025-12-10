import streamlit as st
import pandas as pd
from typing import List, Optional, Dict, Any

from Bio.Seq import Seq

# Reuse your main primer engine
from autoprimer import PrimerDesigner, PrimerPair  # PrimerPair is your simple class in autoprimer.py

# -----------------------------
# HRM / amplicon thermo helpers (from multiplex methodology)
# -----------------------------

import math

def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def rough_amp_tm(seq: str, monovalent_mM: float = 50.0, free_Mg_mM: float = 2.0) -> float:
    """
    Coarse but fast approximation of dsDNA amplicon Tm,
    matching the rough_amp_tm used in your multiplex app.
    """
    seq = "".join([c for c in seq.upper() if c in "ACGT"])
    L = max(len(seq), 1)
    base = 69.3 + 0.41 * (gc_content(seq) * 100.0) - (650.0 / L)
    Na = max(monovalent_mM, 1e-3) / 1000.0  # convert mM → M
    salt = 16.6 * math.log10(Na)
    mg_adj = 0.6 * max(free_Mg_mM, 0.0)
    return base + salt + mg_adj


GC_TAILS = ["", "G", "GC", "GCG", "GCGC", "GCGCG", "GCGCGC"]
AT_TAILS = ["", "A", "AT", "ATAT", "ATATAT", "ATATATAT"]


def clean_dna(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in "ACGT")


def build_amplicon_from_positions(template: str, start: int, product_size: int) -> Optional[str]:
    """
    Use Primer3's reported forward_start + product_size.
    This aligns with how you're already using product_size in autoprimer.
    """
    if start is None or product_size is None:
        return None
    template = clean_dna(template)
    if start < 0 or start + product_size > len(template):
        return None
    return template[start:start + product_size]


def build_amplicon_from_primers(template: str, forward: str, reverse: str) -> Optional[str]:
    """
    Mode B: user provides target sequence + primers.
    We find exact matches of forward and reverse complement within the template
    and return the amplicon sequence between them (inclusive).
    """
    template = clean_dna(template)
    fwd = clean_dna(forward)
    rev = clean_dna(reverse)

    if not template or not fwd or not rev:
        return None

    f_pos = template.find(fwd)
    if f_pos == -1:
        return None

    rev_rc = str(Seq(rev).reverse_complement())
    r_pos = template.find(rev_rc)
    if r_pos == -1:
        return None

    # Ensure forward comes first; if not, swap
    if r_pos < f_pos:
        f_pos, r_pos = r_pos, f_pos

    end = r_pos + len(rev_rc)
    if end > len(template):
        return None

    return template[f_pos:end]


def optimize_pair_for_target_tm(
    template: str,
    primer_pair: PrimerPair,
    target_tm: float,
    monovalent_mM: float,
    free_Mg_mM: float,
    tail_penalty: float = 0.02,
) -> Optional[Dict[str, Any]]:
    """
    Multiplex-style Tm nudging: for a given primer pair,
    scan GC/AT tail combinations and pick the variant whose
    **amplicon** Tm is closest to the desired target_tm.

    Returns a dict with new sequences + Tm and a composite score.
    """
    core_amplicon = build_amplicon_from_positions(
        template,
        primer_pair.forward_start,
        primer_pair.product_size,
    )
    if not core_amplicon:
        return None

    best: Optional[Dict[str, Any]] = None

    for tF in GC_TAILS + AT_TAILS:
        for tR in GC_TAILS + AT_TAILS:
            # Approximate amplicon = 5' tail + core + 3' tail.
            # (Good enough for HRM Tm; structurally, tails become part of product.)
            amplicon_with_tails = f"{tF}{core_amplicon}{tR}"
            amp_tm = rough_amp_tm(amplicon_with_tails, monovalent_mM, free_Mg_mM)

            delta = abs(amp_tm - target_tm)

            # Primer3's penalty already encodes "normal primer scoring":
            # self-dimers, hairpins, GC, Tm, etc.
            # We blend it with |ΔTm| and a small cost for long tails.
            score = (
                delta
                + tail_penalty * (len(tF) + len(tR))
                + 0.1 * getattr(primer_pair, "penalty", 0.0)
            )

            candidate = {
                "forward_with_tail": tF + primer_pair.forward_seq,
                "reverse_with_tail": tR + primer_pair.reverse_seq,
                "tails": (tF, tR),
                "amplicon_tm": amp_tm,
                "delta_tm": amp_tm - target_tm,
                "score": score,
                "product_size": primer_pair.product_size,
                "forward_tm": primer_pair.forward_tm,
                "reverse_tm": primer_pair.reverse_tm,
                "gc_f": primer_pair.gc_content_f,
                "gc_r": primer_pair.gc_content_r,
                "penalty": primer_pair.penalty,
                "gene_target": primer_pair.gene_target,
            }

            if best is None or candidate["score"] < best["score"]:
                best = candidate

    return best


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(
    page_title="AutoPrimer — HRM Designer",
    layout="wide",
)

st.title("AutoPrimer — High-Resolution Melt (HRM) Designer")
st.caption(
    "Predict and design amplicons for HRM assays.\n"
    "Uses your existing PrimerDesigner + multiplex-style amplicon Tm modeling."
)

# Sidebar: buffer conditions + basic design params
st.sidebar.header("Buffer & design conditions")

monovalent_mM = st.sidebar.number_input(
    "Monovalent salt (Na⁺/K⁺, mM)",
    min_value=1.0,
    max_value=200.0,
    value=50.0,
    step=1.0,
)
free_Mg_mM = st.sidebar.number_input(
    "Free Mg²⁺ (mM)",
    min_value=0.0,
    max_value=6.0,
    value=2.0,
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Primer3 constraints (for auto design)")

opt_size = st.sidebar.slider("Optimal primer size", 15, 30, 20)
min_size = st.sidebar.slider("Minimum primer size", 15, 25, 18)
max_size = st.sidebar.slider("Maximum primer size", 20, 35, 25)

opt_tm = st.sidebar.slider("Optimal primer Tm (°C)", 50.0, 70.0, 60.0, 0.5)
min_tm = st.sidebar.slider("Minimum primer Tm (°C)", 45.0, 65.0, 57.0, 0.5)
max_tm = st.sidebar.slider("Maximum primer Tm (°C)", 55.0, 75.0, 63.0, 0.5)

min_gc = st.sidebar.slider("Minimum GC (%)", 20.0, 50.0, 40.0, 1.0)
max_gc = st.sidebar.slider("Maximum GC (%)", 50.0, 80.0, 60.0, 1.0)
max_poly_x = st.sidebar.slider("Max poly-X runs", 3, 6, 4)

product_min = st.sidebar.number_input("Min product size (bp)", 40, 2000, 75)
product_max = st.sidebar.number_input("Max product size (bp)", 40, 3000, 250)

num_pairs = st.sidebar.number_input(
    "Primer pairs to return",
    min_value=1,
    max_value=100,
    value=20,
)

primer3_params = {
    "PRIMER_OPT_SIZE": opt_size,
    "PRIMER_MIN_SIZE": min_size,
    "PRIMER_MAX_SIZE": max_size,
    "PRIMER_OPT_TM": opt_tm,
    "PRIMER_MIN_TM": min_tm,
    "PRIMER_MAX_TM": max_tm,
    "PRIMER_MIN_GC": min_gc,
    "PRIMER_MAX_GC": max_gc,
    "PRIMER_MAX_POLY_X": max_poly_x,
    "PRIMER_SALT_MONOVALENT": monovalent_mM,
    "PRIMER_PRODUCT_SIZE_RANGE": [[product_min, product_max]],
    "PRIMER_NUM_RETURN": num_pairs,
}

designer = PrimerDesigner()

mode = st.radio(
    "Mode",
    [
        "A) Target → auto primers → HRM Tm",
        "B) Target + your primers → HRM Tm",
        "C) Desired HRM Tm → suggest primers",
        "D) HRM Multiplex — multiple targets",
    ],
    horizontal=False,
)

st.markdown("---")

# Shared target input (only for Modes A, B, C - not Mode D)
if not mode.startswith("D)"):
    st.subheader("Target region / template sequence")
    target_seq = st.text_area(
        "Paste template (or just the amplicon region) here",
        height=200,
        placeholder="ATGCGT... (DNA, A/C/G/T only; we'll clean it for you)",
    )

    clean_target = clean_dna(target_seq)

    col_pos1, col_pos2 = st.columns(2)
    with col_pos1:
        use_explicit_region = st.checkbox(
            "Specify target sub-region within template (optional)",
            value=False,
        )
    with col_pos2:
        region_start = st.number_input(
            "Region start (0-based index, inclusive)",
            min_value=0,
            value=0,
        )
        region_end = st.number_input(
            "Region end (0-based index, exclusive)",
            min_value=0,
            value=0,
        )

    target_region = None
    if use_explicit_region:
        if region_end > region_start and region_end <= len(clean_target):
            target_region = (region_start, region_end)
        else:
            st.warning("Region indices are invalid for the current template length.")

    st.markdown("---")
else:
    # Initialize variables for Mode D (they won't be used, but prevents errors)
    clean_target = ""
    target_region = None

# -----------------------------
# Mode A: auto-design primers, predict HRM Tm
# -----------------------------
if mode.startswith("A)"):
    st.subheader("Mode A — Auto-design primers and predict HRM Tm")

    if st.button("Design primers & compute HRM Tm", type="primary"):
        if not clean_target:
            st.error("Please provide a valid DNA sequence.")
        else:
            with st.spinner("Designing primers with Primer3 and computing amplicon Tm..."):
                pairs: List[PrimerPair] = designer.design_primers(
                    clean_target,
                    target_region=target_region,
                    custom_params=primer3_params,
                    add_t7_promoter=False,
                    gene_target="HRM target",
                )

                rows = []
                for idx, p in enumerate(pairs, start=1):
                    amp_seq = build_amplicon_from_positions(
                        clean_target,
                        p.forward_start,
                        p.product_size,
                    )
                    if not amp_seq:
                        continue

                    amp_tm = rough_amp_tm(
                        amp_seq,
                        monovalent_mM=monovalent_mM,
                        free_Mg_mM=free_Mg_mM,
                    )

                    rows.append(
                        {
                            "Rank": idx,
                            "Forward primer": p.forward_seq,
                            "Reverse primer": p.reverse_seq,
                            "Product size (bp)": p.product_size,
                            "Predicted HRM Tm (°C)": round(amp_tm, 2),
                            "Primer Tm F (°C)": round(p.forward_tm, 2),
                            "Primer Tm R (°C)": round(p.reverse_tm, 2),
                            "GC% F": round(p.gc_content_f, 1),
                            "GC% R": round(p.gc_content_r, 1),
                            "Primer3 penalty": round(p.penalty, 3),
                            "Gene target": p.gene_target,
                        }
                    )

                if not rows:
                    st.warning("Primer3 returned no valid primer pairs for these settings.")
                else:
                    df = pd.DataFrame(rows)
                    df = df.sort_values("Predicted HRM Tm (°C)")
                    st.dataframe(df, use_container_width=True)
                    st.success(f"Generated {len(df)} primer pairs with predicted HRM Tm values.")


# -----------------------------
# Mode B: user-supplied primers → HRM Tm
# -----------------------------
elif mode.startswith("B)"):
    st.subheader("Mode B — Use your primers and compute HRM Tm")

    col_f, col_r = st.columns(2)
    with col_f:
        user_fwd = st.text_input(
            "Forward primer (5'→3')",
            value="",
            placeholder="e.g. ATGCGT...",
        )
    with col_r:
        user_rev = st.text_input(
            "Reverse primer (5'→3')",
            value="",
            placeholder="e.g. TCGATC...",
        )

    if st.button("Locate amplicon & compute HRM Tm", type="primary"):
        if not clean_target:
            st.error("Please provide a target/template sequence.")
        elif not user_fwd or not user_rev:
            st.error("Please provide both forward and reverse primers.")
        else:
            with st.spinner("Finding amplicon and computing HRM Tm..."):
                amp_seq = build_amplicon_from_primers(clean_target, user_fwd, user_rev)
                if not amp_seq:
                    st.error("Could not locate both primers within the template (exact match required).")
                else:
                    amp_tm = rough_amp_tm(
                        amp_seq,
                        monovalent_mM=monovalent_mM,
                        free_Mg_mM=free_Mg_mM,
                    )
                    st.write(f"**Amplicon length:** {len(amp_seq)} bp")
                    st.write(f"**Predicted HRM Tm:** `{amp_tm:.2f} °C`")
                    st.code(amp_seq, language="text")


# -----------------------------
# Mode C: desired HRM Tm → suggest primers
# -----------------------------
elif mode.startswith("C)"):
    st.subheader("Mode C — Specify desired HRM Tm, get tailored primers")

    desired_tm = st.number_input(
        "Desired amplicon HRM Tm (°C)",
        min_value=60.0,
        max_value=100.0,
        value=85.0,
        step=0.1,
    )
    max_suggestions = st.number_input(
        "Number of primer pairs to suggest",
        min_value=1,
        max_value=50,
        value=15,
    )

    tail_penalty = st.slider(
        "Tail cost weight (penalty per nt of 5' tail)",
        min_value=0.00,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Higher values discourage long tails; lower values let the algorithm use longer GC-rich tails to hit the target Tm.",
    )

    if st.button("Design primers targeting this HRM Tm", type="primary"):
        if not clean_target:
            st.error("Please provide a target/template sequence.")
        else:
            with st.spinner("Designing base primers and scanning GC/AT tails for desired HRM Tm..."):
                base_pairs: List[PrimerPair] = designer.design_primers(
                    clean_target,
                    target_region=target_region,
                    custom_params=primer3_params,
                    add_t7_promoter=False,
                    gene_target=f"HRM target {desired_tm:.1f}°C",
                )

                tuned: List[Dict[str, Any]] = []
                for p in base_pairs:
                    best = optimize_pair_for_target_tm(
                        clean_target,
                        p,
                        target_tm=desired_tm,
                        monovalent_mM=monovalent_mM,
                        free_Mg_mM=free_Mg_mM,
                        tail_penalty=tail_penalty,
                    )
                    if best:
                        tuned.append(best)

                if not tuned:
                    st.warning("No viable primer/tail combinations found for this target Tm and these constraints.")
                else:
                    tuned.sort(key=lambda d: d["score"])
                    tuned = tuned[: max_suggestions]

                    df = pd.DataFrame(
                        [
                            {
                                "Forward primer (with tail)": t["forward_with_tail"],
                                "Reverse primer (with tail)": t["reverse_with_tail"],
                                "Product size (bp)": t["product_size"],
                                "Predicted HRM Tm (°C)": round(t["amplicon_tm"], 2),
                                "ΔTm (predicted − target, °C)": round(t["delta_tm"], 2),
                                "5' tail F": t["tails"][0],
                                "5' tail R": t["tails"][1],
                                "Primer Tm F (°C)": round(t["forward_tm"], 2),
                                "Primer Tm R (°C)": round(t["reverse_tm"], 2),
                                "GC% F": round(t["gc_f"], 1),
                                "GC% R": round(t["gc_r"], 1),
                                "Primer3 penalty": round(t["penalty"], 3),
                                "Gene target": t["gene_target"],
                            }
                            for t in tuned
                        ]
                    )

                    st.dataframe(df, use_container_width=True)
                    st.success(f"Found {len(df)} primer designs tuned toward {desired_tm:.1f} °C HRM Tm.")


# -----------------------------
# Mode D: HRM Multiplex — multiple targets
# -----------------------------
elif mode.startswith("D)"):
    st.subheader("Mode D — HRM Multiplex for multiple targets (2–4)")

    st.markdown(
        "Design distinct amplicons for several targets in the same reaction and "
        "nudge each product's HRM Tm into a separate 'slot' so melt peaks are resolvable."
    )

    n_targets = st.slider(
        "Number of targets in this multiplex",
        min_value=2,
        max_value=4,
        value=2,
        step=1,
    )

    col_tm1, col_tm2 = st.columns(2)
    with col_tm1:
        base_tm = st.number_input(
            "Center of Tm ladder (°C)",
            min_value=65.0,
            max_value=95.0,
            value=82.0,
            step=0.5,
            help="The mid-point of your desired melt peaks. Actual peaks will be spaced around this."
        )
    with col_tm2:
        delta_tm = st.number_input(
            "Desired separation between melt peaks (°C)",
            min_value=0.5,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Target spacing between adjacent peaks. Actual spacing will be approximate."
        )

    auto_assign_slots = st.checkbox(
        "Auto-assign Tm ladder (recommended)",
        value=True,
        help="If checked, we place each target onto a regular Tm ladder around the center. "
             "Otherwise, you can set a custom desired Tm for each target."
    )

    tail_penalty = st.slider(
        "Tail cost weight (penalty per nt of 5' tail)",
        min_value=0.00,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Higher values discourage long tails; lower values allow more GC-rich tails to hit a Tm slot."
    )

    # Compute ladder Tms if auto-assign
    ladder_tms = []
    center_index = (n_targets - 1) / 2.0
    for i in range(n_targets):
        ladder_tms.append(base_tm + (i - center_index) * delta_tm)

    st.markdown("### Targets")

    target_names: List[str] = []
    target_seqs: List[str] = []
    target_desired_tms: List[float] = []

    for i in range(n_targets):
        st.markdown(f"**Target {i+1}**")
        col_info = st.columns([2, 1])
        with col_info[0]:
            name = st.text_input(
                f"Name / label for target {i+1}",
                value=f"Target {i+1}",
                key=f"mux_name_{i}",
            )
        with col_info[1]:
            if auto_assign_slots:
                # Show the auto-assigned Tm for this target
                st.write(f"Assigned Tm slot: **{ladder_tms[i]:.1f} °C**")
                desired_tm_i = ladder_tms[i]
            else:
                desired_tm_i = st.number_input(
                    f"Desired HRM Tm for {i+1} (°C)",
                    min_value=60.0,
                    max_value=100.0,
                    value=ladder_tms[i] if ladder_tms else 82.0,
                    step=0.1,
                    key=f"mux_tm_{i}",
                )

            target_desired_tms.append(desired_tm_i)

        seq = st.text_area(
            f"Template sequence for {name}",
            height=120,
            key=f"mux_seq_{i}",
            placeholder="ATGCGT...",
        )

        target_names.append(name)
        target_seqs.append(clean_dna(seq))

        st.markdown("---")

    st.markdown("### Multiplex optimization options")

    anchor_idx = st.selectbox(
        "Which target is most constrained / should act as the anchor?",
        options=list(range(n_targets)),
        format_func=lambda i: target_names[i] if target_names[i] else f"Target {i+1}",
        help=(
            "The anchor target is the one whose amplicon Tm is least negotiable. "
            "We'll design this first, then space other targets' Tm slots around its actual predicted Tm."
        ),
    )

    if st.button("Design multiplex primers", type="primary"):
        # Basic validation
        any_seq = any(s for s in target_seqs)
        if not any_seq:
            st.error("Please provide at least one non-empty target sequence.")
        else:
            with st.spinner("Designing primers for each target and tuning HRM Tm slots..."):
                all_rows = []
                predicted_tms = [None] * n_targets

                # --- 1) Anchor target first ---
                anchor_name = target_names[anchor_idx]
                anchor_seq = target_seqs[anchor_idx]

                if not anchor_seq:
                    st.error("The chosen anchor target has no sequence; please provide it.")
                    st.stop()

                # Design base primers for the anchor
                try:
                    base_pairs_anchor: List[PrimerPair] = designer.design_primers(
                        anchor_seq,
                        target_region=None,
                        custom_params=primer3_params,
                        add_t7_promoter=False,
                        gene_target=f"{anchor_name} (multiplex HRM anchor)",
                    )
                except Exception as e:
                    st.error(f"Primer design failed for anchor target: {e}")
                    st.stop()

                if not base_pairs_anchor:
                    st.error("No primer pairs found for the anchor target with current constraints.")
                    st.stop()

                best_anchor = None
                desired_anchor_tm = base_tm  # Use base_tm as target for anchor
                for p in base_pairs_anchor:
                    tuned = optimize_pair_for_target_tm(
                        anchor_seq,
                        p,
                        target_tm=desired_anchor_tm,
                        monovalent_mM=monovalent_mM,
                        free_Mg_mM=free_Mg_mM,
                        tail_penalty=tail_penalty * 2.0,  # heavier tail penalty for anchor
                    )
                    if tuned is None:
                        continue
                    if best_anchor is None or tuned["score"] < best_anchor["score"]:
                        best_anchor = tuned

                if best_anchor is None:
                    st.error("Could not find a suitable anchor design. Try relaxing constraints.")
                    st.stop()

                anchor_tm_actual = best_anchor["amplicon_tm"]
                predicted_tms[anchor_idx] = anchor_tm_actual

                # --- 2) Define desired Tm slots for all targets relative to the anchor ---
                desired_tms_effective = [None] * n_targets
                for i in range(n_targets):
                    offset = i - anchor_idx
                    desired_tms_effective[i] = anchor_tm_actual + offset * delta_tm

                # --- 3) Record anchor row first ---
                all_rows.append({
                    "Target": anchor_name,
                    "Desired HRM Tm (°C)": round(desired_tms_effective[anchor_idx], 2),
                    "Predicted HRM Tm (°C)": round(anchor_tm_actual, 2),
                    "ΔTm (predicted − desired, °C)": round(anchor_tm_actual - desired_tms_effective[anchor_idx], 2),
                    "Forward primer (with tail)": best_anchor["forward_with_tail"],
                    "Reverse primer (with tail)": best_anchor["reverse_with_tail"],
                    "5' tail F": best_anchor["tails"][0],
                    "5' tail R": best_anchor["tails"][1],
                    "Product size (bp)": best_anchor["product_size"],
                    "Primer Tm F (°C)": round(best_anchor["forward_tm"], 2),
                    "Primer Tm R (°C)": round(best_anchor["reverse_tm"], 2),
                    "GC% F": round(best_anchor["gc_f"], 1),
                    "GC% R": round(best_anchor["gc_r"], 1),
                    "Primer3 penalty": round(best_anchor["penalty"], 3),
                    "Status": "Anchor OK",
                })

                # --- 4) Now design non-anchor targets around the anchor Tm ---
                for idx in range(n_targets):
                    if idx == anchor_idx:
                        continue

                    name = target_names[idx]
                    seq = target_seqs[idx]
                    desired_tm = desired_tms_effective[idx]

                    if not seq:
                        all_rows.append({
                            "Target": name,
                            "Desired HRM Tm (°C)": round(desired_tm, 2),
                            "Status": "No sequence provided",
                        })
                        continue

                    try:
                        base_pairs: List[PrimerPair] = designer.design_primers(
                            seq,
                            target_region=None,
                            custom_params=primer3_params,
                            add_t7_promoter=False,
                            gene_target=f"{name} (multiplex HRM)",
                        )
                    except Exception as e:
                        all_rows.append({
                            "Target": name,
                            "Desired HRM Tm (°C)": round(desired_tm, 2),
                            "Status": f"Primer design failed: {e}",
                        })
                        continue

                    if not base_pairs:
                        all_rows.append({
                            "Target": name,
                            "Desired HRM Tm (°C)": round(desired_tm, 2),
                            "Status": "No primer pairs found for this target.",
                        })
                        continue

                    best_candidate = None
                    for p in base_pairs:
                        tuned = optimize_pair_for_target_tm(
                            seq,
                            p,
                            target_tm=desired_tm,
                            monovalent_mM=monovalent_mM,
                            free_Mg_mM=free_Mg_mM,
                            tail_penalty=tail_penalty,
                        )
                        if tuned is None:
                            continue
                        if best_candidate is None or tuned["score"] < best_candidate["score"]:
                            best_candidate = tuned

                    if best_candidate is None:
                        all_rows.append({
                            "Target": name,
                            "Desired HRM Tm (°C)": round(desired_tm, 2),
                            "Status": "Could not hit this Tm slot.",
                        })
                        continue

                    predicted_tms[idx] = best_candidate["amplicon_tm"]

                    all_rows.append({
                        "Target": name,
                        "Desired HRM Tm (°C)": round(desired_tm, 2),
                        "Predicted HRM Tm (°C)": round(best_candidate["amplicon_tm"], 2),
                        "ΔTm (predicted − desired, °C)": round(best_candidate["delta_tm"], 2),
                        "Forward primer (with tail)": best_candidate["forward_with_tail"],
                        "Reverse primer (with tail)": best_candidate["reverse_with_tail"],
                        "5' tail F": best_candidate["tails"][0],
                        "5' tail R": best_candidate["tails"][1],
                        "Product size (bp)": best_candidate["product_size"],
                        "Primer Tm F (°C)": round(best_candidate["forward_tm"], 2),
                        "Primer Tm R (°C)": round(best_candidate["reverse_tm"], 2),
                        "GC% F": round(best_candidate["gc_f"], 1),
                        "GC% R": round(best_candidate["gc_r"], 1),
                        "Primer3 penalty": round(best_candidate["penalty"], 3),
                        "Status": "OK",
                    })

                # Display table
                df = pd.DataFrame(all_rows)

                # Try to sort by predicted HRM Tm if available
                if "Predicted HRM Tm (°C)" in df.columns:
                    df = df.sort_values("Predicted HRM Tm (°C)", na_position="last")

                st.dataframe(df, use_container_width=True)

                # Report minimal spacing between peaks if we have predictions
                predicted_tms_clean = sorted([tm for tm in predicted_tms if tm is not None])
                if len(predicted_tms_clean) >= 2:
                    diffs = [
                        predicted_tms_clean[i+1] - predicted_tms_clean[i]
                        for i in range(len(predicted_tms_clean) - 1)
                    ]
                    min_sep = min(diffs)
                    st.info(
                        f"Minimum predicted peak separation: **{min_sep:.2f} °C** "
                        f"(desired: {delta_tm:.2f} °C)."
                    )
                    if min_sep < (delta_tm * 0.7):
                        st.warning(
                            "Some peaks are predicted to be closer together than desired. "
                            "You may want to increase ΔTm, adjust primer constraints, or try different targets."
                        )
                else:
                    st.info("Not enough successful designs to estimate peak separation.")

