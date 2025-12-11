import streamlit as st
import pandas as pd
import itertools
from typing import List, Optional, Dict, Any

from Bio.Seq import Seq

# Reuse your main primer engine
from autoprimer import PrimerDesigner, PrimerPair  # PrimerPair is your simple class in autoprimer.py

# -----------------------------
# HRM / amplicon thermo helpers (from multiplex methodology)
# -----------------------------

import math

# Instrument-specific calibration offset for HRM peaks (°C).
# This gets updated from the Streamlit sidebar.
INSTRUMENT_TM_OFFSET = 0.0

def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def rough_amp_tm(seq: str, monovalent_mM: float = 50.0, free_Mg_mM: float = 2.0) -> float:
    """
    Approximate dsDNA amplicon Tm using a PCR-product formula (von Ahsen et al. 2001):

        Tm_raw = 77.1 + 0.41(%GC) - 528/L + 11.7 * log10([Na+])

    where [Na+] is in mol/L and L is the amplicon length in bp.

    This is substantially closer to real PCR product / HRM peaks than the older
    69.3 + 0.41*GC - 650/L + 16.6*log10([Na+]) style formula.

    We then apply an instrument-specific offset (INSTRUMENT_TM_OFFSET), which you
    can set from the sidebar to align the model to your actual HRM machine.
    """
    # Clean + basic sanity
    seq = "".join([c for c in seq.upper() if c in "ACGT"])
    L = len(seq)
    if L == 0:
        return 0.0

    gc_percent = gc_content(seq) * 100.0  # 0–100
    # Convert monovalent salt from mM → M for the formula
    Na_M = max(monovalent_mM, 1e-3) / 1000.0

    # von Ahsen-style PCR product Tm
    tm_raw = 77.1 + 0.41 * gc_percent - (528.0 / L) + 11.7 * math.log10(Na_M)

    # For now, we fold Mg²⁺ and dye/system quirks into a single calibration offset
    return tm_raw + INSTRUMENT_TM_OFFSET


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


def visualize_primer_binding(
    template: str,
    forward_seq: str,
    reverse_seq: str,
    forward_start: Optional[int] = None,
    product_size: Optional[int] = None,
    forward_tail: str = "",
    reverse_tail: str = "",
) -> str:
    """
    Create a text visualization showing primer binding positions and amplicon.
    Returns HTML string for display in Streamlit.
    """
    template = clean_dna(template)
    fwd = clean_dna(forward_seq)
    rev = clean_dna(reverse_seq)
    
    if not template or not fwd or not rev:
        return "<p>Cannot visualize: missing sequence data.</p>"
    
    # Find primer positions
    if forward_start is not None and product_size is not None:
        # Use provided positions
        f_pos = forward_start
        r_pos_end = forward_start + product_size
        rev_rc = str(Seq(rev).reverse_complement())
        r_pos = r_pos_end - len(rev_rc)
    else:
        # Find positions by sequence matching
        f_pos = template.find(fwd)
        if f_pos == -1:
            return "<p>Cannot visualize: forward primer not found in template.</p>"
        
        rev_rc = str(Seq(rev).reverse_complement())
        r_pos = template.find(rev_rc)
        if r_pos == -1:
            return "<p>Cannot visualize: reverse primer not found in template.</p>"
        
        # Ensure forward comes first
        if r_pos < f_pos:
            f_pos, r_pos = r_pos, f_pos
            fwd, rev_rc = rev_rc, fwd
        
        r_pos_end = r_pos + len(rev_rc)
    
    # Determine what portion of template to show (with some context)
    context_before = 20
    context_after = 20
    show_start = max(0, f_pos - context_before)
    show_end = min(len(template), r_pos_end + context_after)
    
    # Extract the region to display
    display_seq = template[show_start:show_end]
    f_display_pos = f_pos - show_start
    r_display_pos = r_pos - show_start
    r_display_end = r_pos_end - show_start
    
    # Build HTML visualization
    html_parts = []
    html_parts.append("<div style='font-family: monospace; font-size: 12px; line-height: 1.6;'>")
    
    # Show template region info
    html_parts.append(f"<p><strong>Template region:</strong> (positions {show_start+1}-{show_end} of {len(template)} bp)</p>")
    
    # Template sequence with highlighting (what's in the original template)
    html_parts.append("<div style='background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;'>")
    html_parts.append("<p style='margin: 0 0 8px 0; font-size: 11px; color: #666;'><strong>Original template sequence:</strong></p>")
    
    # Before forward primer
    if f_display_pos > 0:
        html_parts.append(f"<span style='color: #666;'>{display_seq[:f_display_pos]}</span>")
    
    # Forward primer binding (highlighted)
    fwd_binding = display_seq[f_display_pos:f_display_pos+len(fwd)]
    html_parts.append(f"<span style='background: #90EE90; color: #000; font-weight: bold;' title='Forward primer binding region'>{fwd_binding}</span>")
    
    # Amplicon region (between primers)
    amp_start = f_display_pos + len(fwd)
    amp_end = r_display_pos
    if amp_end > amp_start:
        amplicon = display_seq[amp_start:amp_end]
        html_parts.append(f"<span style='background: #FFE4B5; color: #000;' title='Amplicon region'>{amplicon}</span>")
    
    # Reverse primer binding (highlighted)
    rev_binding = display_seq[r_display_pos:r_display_end]
    html_parts.append(f"<span style='background: #87CEEB; color: #000; font-weight: bold;' title='Reverse primer binding region'>{rev_binding}</span>")
    
    # After reverse primer
    if r_display_end < len(display_seq):
        html_parts.append(f"<span style='color: #666;'>{display_seq[r_display_end:]}</span>")
    
    html_parts.append("</div>")
    
    # Show final PCR product (including tails) if tails are present
    if forward_tail or reverse_tail:
        html_parts.append("<div style='background: #E6F3FF; padding: 10px; border-radius: 5px; margin: 10px 0; border: 2px solid #4A90E2;'>")
        html_parts.append("<p style='margin: 0 0 8px 0; font-size: 11px; color: #4A90E2;'><strong>Final PCR product (includes tails):</strong></p>")
        
        # Forward tail (attached to 5' end of forward primer)
        if forward_tail:
            html_parts.append(f"<span style='background: #FFB6C1; color: #000; font-weight: bold; border: 2px solid #FF69B4; padding: 2px;' title='5&apos; tail attached to forward primer'>{forward_tail}</span>")
        
        # Forward primer binding region
        html_parts.append(f"<span style='background: #90EE90; color: #000; font-weight: bold;'>{fwd_binding}</span>")
        
        # Amplicon region (between primers)
        if amp_end > amp_start:
            html_parts.append(f"<span style='background: #FFE4B5; color: #000;'>{amplicon}</span>")
        
        # Reverse primer binding region
        html_parts.append(f"<span style='background: #87CEEB; color: #000; font-weight: bold;'>{rev_binding}</span>")
        
        # Reverse tail (attached to 5' end of reverse primer)
        if reverse_tail:
            html_parts.append(f"<span style='background: #FFB6C1; color: #000; font-weight: bold; border: 2px solid #FF69B4; padding: 2px;' title='5&apos; tail attached to reverse primer'>{reverse_tail}</span>")
        
        total_amp_len = len(forward_tail) + (r_pos_end - f_pos) + len(reverse_tail)
        html_parts.append(f"<p style='margin: 8px 0 0 0; font-size: 11px; color: #4A90E2;'><strong>Total product length:</strong> {total_amp_len} bp (template: {r_pos_end - f_pos} bp + tails: {len(forward_tail) + len(reverse_tail)} bp)</p>")
        html_parts.append("</div>")
    
    # Legend and primer details
    html_parts.append("<div style='margin: 10px 0;'>")
    html_parts.append("<p><strong>Primers:</strong></p>")
    html_parts.append("<ul style='margin: 5px 0; padding-left: 20px;'>")
    
    fwd_full = forward_tail + fwd if forward_tail else fwd
    tail_info_f = f" <span style='color: #FF69B4; font-weight: bold;'>(tail: {forward_tail})</span>" if forward_tail else ""
    html_parts.append(f"<li><span style='background: #90EE90; padding: 2px 5px;'>Forward</span>: "
                      f"<code>{fwd_full}</code>{tail_info_f} (binds at position {f_pos+1})</li>")
    
    rev_full = reverse_tail + rev if reverse_tail else rev
    tail_info_r = f" <span style='color: #FF69B4; font-weight: bold;'>(tail: {reverse_tail})</span>" if reverse_tail else ""
    html_parts.append(f"<li><span style='background: #87CEEB; padding: 2px 5px;'>Reverse</span>: "
                      f"<code>{rev_full}</code>{tail_info_r} (binds at position {r_pos+1})</li>")
    
    if forward_tail or reverse_tail:
        html_parts.append(f"<li><strong>Tails (5&apos; primer extensions):</strong> Forward={forward_tail or '(none)'}, Reverse={reverse_tail or '(none)'}</li>")
        html_parts.append(f"<li style='color: #666; font-size: 11px; margin-top: 5px;'><em>Tails are attached to the 5&apos; end of primers and become part of the final PCR product. "
                         f"The final amplicon = {forward_tail or ''}[forward primer]{amplicon if amp_end > amp_start else ''}[reverse primer]{reverse_tail or ''}</em></li>")
    else:
        html_parts.append("<li><strong>Tails:</strong> None</li>")
    
    template_amp_len = r_pos_end - f_pos
    final_amp_len = template_amp_len + len(forward_tail) + len(reverse_tail)
    html_parts.append(f"<li><strong>Template region:</strong> {template_amp_len} bp (positions {f_pos+1}-{r_pos_end})</li>")
    if forward_tail or reverse_tail:
        html_parts.append(f"<li><strong>Final PCR product:</strong> {final_amp_len} bp (template + tails)</li>")
    
    html_parts.append("</ul>")
    html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    return "".join(html_parts)


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
                "forward_start": primer_pair.forward_start,  # Store for visualization
                "forward_seq": primer_pair.forward_seq,  # Store core sequences
                "reverse_seq": primer_pair.reverse_seq,
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

# --- HRM calibration ---
st.sidebar.markdown("#### HRM calibration")
st.sidebar.caption(
    "Use a known amplicon (with a measured HRM peak) to set this offset so that the "
    "predicted Tm matches your instrument. The same offset is then applied to all designs."
)
offset_user = st.sidebar.number_input(
    "Instrument Tm offset (observed − predicted, °C)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.5,
)

# Update the global offset used in rough_amp_tm
INSTRUMENT_TM_OFFSET = float(offset_user)

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
                    
                    # Visualization - show all
                    if len(pairs) > 0:
                        st.markdown("---")
                        st.subheader("Primer Binding Visualizations")
                        for idx, p in enumerate(pairs, start=1):
                            with st.expander(f"Pair {idx}: {p.forward_seq} / {p.reverse_seq}", expanded=(idx == 1)):
                                viz_html = visualize_primer_binding(
                                    clean_target,
                                    p.forward_seq,
                                    p.reverse_seq,
                                    forward_start=p.forward_start,
                                    product_size=p.product_size,
                                )
                                st.markdown(viz_html, unsafe_allow_html=True)


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
                    
                    # Visualization
                    st.markdown("---")
                    st.subheader("Primer Binding Visualization")
                    viz_html = visualize_primer_binding(
                        clean_target,
                        user_fwd,
                        user_rev,
                    )
                    st.markdown(viz_html, unsafe_allow_html=True)


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
                    
                    # Visualization - show all
                    if len(tuned) > 0:
                        st.markdown("---")
                        st.subheader("Primer Binding Visualizations")
                        for idx, t in enumerate(tuned, start=1):
                            with st.expander(f"Design {idx}: {t['forward_with_tail']} / {t['reverse_with_tail']}", expanded=(idx == 1)):
                                # Use stored position information
                                if "forward_start" in t and "forward_seq" in t:
                                    viz_html = visualize_primer_binding(
                                        clean_target,
                                        t["forward_seq"],
                                        t["reverse_seq"],
                                        forward_start=t["forward_start"],
                                        product_size=t["product_size"],
                                        forward_tail=t["tails"][0],
                                        reverse_tail=t["tails"][1],
                                    )
                                    st.markdown(viz_html, unsafe_allow_html=True)
                                else:
                                    # Fallback: try to find positions by sequence matching
                                    fwd_core = t["forward_with_tail"][len(t["tails"][0]):]
                                    rev_core = t["reverse_with_tail"][len(t["tails"][1]):]
                                    f_pos = clean_target.find(fwd_core)
                                    if f_pos != -1:
                                        rev_rc = str(Seq(rev_core).reverse_complement())
                                        r_pos = clean_target.find(rev_rc)
                                        if r_pos != -1:
                                            product_size = (r_pos + len(rev_rc)) - f_pos
                                            viz_html = visualize_primer_binding(
                                                clean_target,
                                                fwd_core,
                                                rev_core,
                                                forward_start=f_pos,
                                                product_size=product_size,
                                                forward_tail=t["tails"][0],
                                                reverse_tail=t["tails"][1],
                                            )
                                            st.markdown(viz_html, unsafe_allow_html=True)
                                        else:
                                            st.warning("Could not locate reverse primer in template for visualization.")
                                    else:
                                        st.warning("Could not locate forward primer in template for visualization.")


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

    free_slot_assignment = st.checkbox(
        "Let optimization choose which target goes to which Tm slot",
        value=False,
        help=(
            "If checked, the Tm ladder is treated as a set of slots, and the algorithm "
            "assigns each target to the slot it can hit best, instead of locking "
            "Target 1 → Slot 1, Target 2 → Slot 2, etc."
        ),
    )

    if not free_slot_assignment:
        use_anchor = st.checkbox(
            "Use anchor target (design one target first, then space others around it)",
            value=True,
            help=(
                "If checked, one target is designed first as an anchor, and other targets "
                "are spaced around its actual predicted Tm. If unchecked, each target is "
                "designed independently to hit its assigned Tm slot."
            ),
        )
        
        if use_anchor:
            anchor_idx = st.selectbox(
                "Which target is most constrained / should act as the anchor?",
                options=list(range(n_targets)),
                format_func=lambda i: target_names[i] if target_names[i] else f"Target {i+1}",
                help=(
                    "The anchor target is the one whose amplicon Tm is least negotiable. "
                    "We'll design this first, then space other targets' Tm slots around its actual predicted Tm."
                ),
            )
        else:
            anchor_idx = 0  # dummy, won't be used when no anchor
    else:
        use_anchor = False  # not used in free-assignment mode
        anchor_idx = 0  # dummy, won't be used in free-assignment mode

    if st.button("Design multiplex primers", type="primary"):
        # Basic validation
        any_seq = any(s for s in target_seqs)
        if not any_seq:
            st.error("Please provide at least one non-empty target sequence.")
        else:
            with st.spinner("Designing primers for each target and tuning HRM Tm slots..."):
                all_rows = []
                predicted_tms = [None] * n_targets
                visualization_data = {}  # Store (target_name, candidate) for visualization

                if free_slot_assignment:
                    # --- Free slot assignment path ---
                    slot_tms = target_desired_tms  # ladder values from UI (auto or manual)

                    # Precompute best candidate for each (target, slot) pair
                    best_for_target_slot = {}
                    cost_matrix = {}

                    for i, (name, seq) in enumerate(zip(target_names, target_seqs)):
                        if not seq:
                            continue

                        try:
                            base_pairs: List[PrimerPair] = designer.design_primers(
                                seq,
                                target_region=None,
                                custom_params=primer3_params,
                                add_t7_promoter=False,
                                gene_target=f"{name} (multiplex HRM free-assign)",
                            )
                        except Exception as e:
                            all_rows.append({
                                "Target": name,
                                "Status": f"Primer design failed: {e}",
                            })
                            continue

                        if not base_pairs:
                            all_rows.append({
                                "Target": name,
                                "Status": "No primer pairs found for this target.",
                            })
                            continue

                        for slot_idx, slot_tm in enumerate(slot_tms):
                            best_candidate = None
                            for p in base_pairs:
                                tuned = optimize_pair_for_target_tm(
                                    seq,
                                    p,
                                    target_tm=slot_tm,
                                    monovalent_mM=monovalent_mM,
                                    free_Mg_mM=free_Mg_mM,
                                    tail_penalty=tail_penalty,
                                )
                                if tuned is None:
                                    continue
                                if best_candidate is None or tuned["score"] < best_candidate["score"]:
                                    best_candidate = tuned

                            if best_candidate is not None:
                                best_for_target_slot[(i, slot_idx)] = best_candidate
                                cost_matrix[(i, slot_idx)] = best_candidate["score"]

                    # Determine which targets actually have any viable slot
                    valid_targets = sorted({
                        i for (i, slot_idx) in best_for_target_slot.keys()
                    })

                    if not valid_targets:
                        st.error("No viable primer/slot combinations were found. Try relaxing constraints.")
                        st.stop()

                    # Brute-force optimal assignment (targets → distinct slots)
                    n = len(valid_targets)
                    slot_indices = list(range(len(slot_tms)))
                    best_perm = None
                    best_total_cost = float("inf")

                    for perm in itertools.permutations(slot_indices, n):
                        total = 0.0
                        feasible = True
                        for t_idx, s_idx in zip(valid_targets, perm):
                            if (t_idx, s_idx) not in cost_matrix:
                                feasible = False
                                break
                            total += cost_matrix[(t_idx, s_idx)]
                        if feasible and total < best_total_cost:
                            best_total_cost = total
                            best_perm = dict(zip(valid_targets, perm))

                    if best_perm is None:
                        st.error("Could not find a feasible assignment of targets to Tm slots.")
                        st.stop()

                    # Build results table
                    for i in range(n_targets):
                        name = target_names[i]
                        seq = target_seqs[i]

                        if not seq:
                            all_rows.append({
                                "Target": name,
                                "Status": "No sequence provided",
                            })
                            continue

                        slot_idx = best_perm.get(i, None)
                        if slot_idx is None:
                            all_rows.append({
                                "Target": name,
                                "Status": "Not assigned to any Tm slot (no viable design).",
                            })
                            continue

                        candidate = best_for_target_slot.get((i, slot_idx))
                        if candidate is None:
                            all_rows.append({
                                "Target": name,
                                "Status": "Internal error: assignment has no candidate.",
                            })
                            continue

                        assigned_tm = slot_tms[slot_idx]
                        predicted_tms[i] = candidate["amplicon_tm"]
                        if seq:  # Only store if we have a sequence
                            visualization_data[name] = (seq, candidate)  # Store for visualization

                        all_rows.append({
                            "Target": name,
                            "Assigned Tm slot (°C)": round(assigned_tm, 2),
                            "Predicted HRM Tm (°C)": round(candidate["amplicon_tm"], 2),
                            "ΔTm (predicted − slot, °C)": round(candidate["amplicon_tm"] - assigned_tm, 2),
                            "Forward primer (with tail)": candidate["forward_with_tail"],
                            "Reverse primer (with tail)": candidate["reverse_with_tail"],
                            "5' tail F": candidate["tails"][0],
                            "5' tail R": candidate["tails"][1],
                            "Product size (bp)": candidate["product_size"],
                            "Primer Tm F (°C)": round(candidate["forward_tm"], 2),
                            "Primer Tm R (°C)": round(candidate["reverse_tm"], 2),
                            "GC% F": round(candidate["gc_f"], 1),
                            "GC% R": round(candidate["gc_r"], 1),
                            "Primer3 penalty": round(candidate["penalty"], 3),
                            "Status": "Assigned",
                        })

                else:
                    # --- Fixed assignment path ---
                    desired_tms_effective = target_desired_tms.copy()
                    
                    if use_anchor:
                        # --- Anchor-first logic ---
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
                        # Use the assigned Tm slot for the anchor (from ladder_tms or user input)
                        desired_anchor_tm = target_desired_tms[anchor_idx]
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
                        visualization_data[anchor_name] = (anchor_seq, best_anchor)  # Store for visualization

                        # --- 2) Record anchor row first ---
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

                        # --- 3) Now design non-anchor targets ---
                        target_indices = [idx for idx in range(n_targets) if idx != anchor_idx]
                    else:
                        # --- No anchor: design all targets independently ---
                        target_indices = list(range(n_targets))

                    # --- Design all targets (non-anchor if using anchor, or all if not) ---
                    for idx in target_indices:
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
                        visualization_data[name] = (seq, best_candidate)  # Store for visualization

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

                # Display table (for both paths)
                if not free_slot_assignment:
                    df = pd.DataFrame(all_rows)

                    # Try to sort by predicted HRM Tm if available
                    if "Predicted HRM Tm (°C)" in df.columns:
                        df = df.sort_values("Predicted HRM Tm (°C)", na_position="last")

                    st.dataframe(df, use_container_width=True)
                    
                    # Visualization - show all
                    if visualization_data:
                        st.markdown("---")
                        st.subheader("Primer Binding Visualizations")
                        for target_name in sorted(visualization_data.keys()):
                            target_seq, candidate = visualization_data[target_name]
                            with st.expander(f"Target: {target_name}", expanded=True):
                                if "forward_start" in candidate and "forward_seq" in candidate:
                                    viz_html = visualize_primer_binding(
                                        target_seq,
                                        candidate["forward_seq"],
                                        candidate["reverse_seq"],
                                        forward_start=candidate["forward_start"],
                                        product_size=candidate["product_size"],
                                        forward_tail=candidate["tails"][0],
                                        reverse_tail=candidate["tails"][1],
                                    )
                                    st.markdown(viz_html, unsafe_allow_html=True)
                                else:
                                    st.warning("Position information not available for visualization.")

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
                else:
                    # For free assignment, display table
                    df = pd.DataFrame(all_rows)

                    # Sort by assigned Tm slot
                    if "Assigned Tm slot (°C)" in df.columns:
                        df = df.sort_values("Assigned Tm slot (°C)", na_position="last")

                    st.dataframe(df, use_container_width=True)
                    
                    # Visualization - show all
                    if visualization_data:
                        st.markdown("---")
                        st.subheader("Primer Binding Visualizations")
                        for target_name in sorted(visualization_data.keys()):
                            target_seq, candidate = visualization_data[target_name]
                            with st.expander(f"Target: {target_name}", expanded=True):
                                if "forward_start" in candidate and "forward_seq" in candidate:
                                    viz_html = visualize_primer_binding(
                                        target_seq,
                                        candidate["forward_seq"],
                                        candidate["reverse_seq"],
                                        forward_start=candidate["forward_start"],
                                        product_size=candidate["product_size"],
                                        forward_tail=candidate["tails"][0],
                                        reverse_tail=candidate["tails"][1],
                                    )
                                    st.markdown(viz_html, unsafe_allow_html=True)
                                else:
                                    st.warning("Position information not available for visualization.")

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

