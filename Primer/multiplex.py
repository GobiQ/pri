# AutoPrimer5 — Multiplex Mode (15 targets × 3 channels)
# Minimal Streamlit app (integrated with your autoprimer5.py catalog)
# --------------------------------------------------------------
# New in this version
# • Imports organism/target list directly from autoprimer5.py
# • Auto-loads target nucleotide sequences from the catalog (no NCBI required)
# • Robust catalog parsing with user-provided sequence key hints
# • Auto-assigns 15 targets to 3×5 Tm slots while optimizing primer quality
# • Debug panel to show why entries might appear as <none>
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import re, io, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# Optional libs
try:
    import primer3
    HAVE_PRIMER3 = True
except Exception:
    HAVE_PRIMER3 = False

try:
    from Bio.Seq import Seq
    from Bio import SeqIO
    HAVE_BIO = True
except Exception:
    HAVE_BIO = False

# -----------------------------
# Catalog & sequence sources (LOCAL + optional NCBI fetch)
# -----------------------------

# Local, built‑in sample entries (you can keep or replace)
def get_organism_suggestions_with_gene_targets():
    """Get the 14 specific organisms from the image as default selections"""
    return {
        "🦠 Viruses and Viroids": [
            ("Hop Latent Viroid", "Hop Latent Viroid", {
                "Structural domains": ["P (pathogenicity)", "C (central)", "V (variable)", "T1 (terminal left)", "T2 (terminal right)"],
                "Secondary structure": ["rod-like structure", "base-pairing", "hairpin loops", "bulges", "thermodynamic stability"],
                "Variants": ["HLVd variants", "sequence diversity", "geographic strains", "hop cultivar adaptation"],
                "Pathogenicity": ["latent infection", "hop stunt", "yield reduction", "brewing quality", "bitter compound"],
                "Detection": ["RT-PCR", "northern blot", "dot blot", "in situ hybridization", "high-throughput sequencing"]
            }),
            ("Lettuce Chlorosis Virus", "Lettuce chlorosis virus", {
                "Essential genes": ["RNA1 (replication)", "RNA2 (movement)", "P1 (replicase)", "P2 (helicase)", "P3 (polymerase)", "MP (movement protein)", "CP (coat protein)"],
                "Crinivirus features": ["bipartite RNA genome", "whitefly transmission", "phloem-limited", "long flexuous particles", "genome activation"],
                "Host range": ["lettuce", "tomato", "pepper", "cucumber", "melon", "squash", "bean"],
                "Vector transmission": ["whitefly transmission", "Bemisia tabaci", "Trialeurodes vaporariorum", "semi-persistent", "circulative"],
                "Pathogenicity": ["chlorosis", "yellowing", "stunting", "phloem necrosis", "yield reduction"]
            }),
            ("Alfalfa Mosaic Virus", "Alfalfa mosaic virus", {
                "Essential genes": ["RNA1 (P1, P2)", "RNA2 (P3)", "RNA3 (MP, CP)", "P1 (replicase)", "P2 (helicase)", "P3 (polymerase)", "MP (movement protein)", "CP (coat protein)"],
                "Alfamovirus features": ["tripartite RNA genome", "bacilliform particles", "coat protein requirement", "genome activation", "replication"],
                "Host range": ["alfalfa", "tobacco", "tomato", "pepper", "bean", "cucumber", "lettuce"],
                "Vector transmission": ["aphid transmission", "non-persistent", "stylet-borne", "Myzus persicae", "Aphis gossypii"],
                "Pathogenicity": ["mosaic symptoms", "yellowing", "stunting", "systemic infection", "yield reduction"]
            }),
            ("Beet Curly Top Virus", "Beet curly top virus", {
                "Essential genes": ["C1 (replication initiator)", "C2 (transcription activator)", "C3 (replication enhancer)", "C4 (pathogenicity)", "V1 (coat protein)", "V2 (movement protein)"],
                "Geminivirus features": ["circular ssDNA", "bipartite genome", "rolling circle replication", "transcription activation", "silencing suppression"],
                "Host range": ["beet", "tomato", "pepper", "bean", "spinach", "squash", "cucumber"],
                "Vector transmission": ["beet leafhopper", "Circulifer tenellus", "persistent transmission", "propagative transmission"],
                "Pathogenicity factors": ["C4 (symptom determinant)", "C2 (transcription activator)", "V2 (movement protein)", "silencing suppression", "host range determination"]
            }),
            ("Cannabis Cryptic Virus", "Cannabis cryptic virus", {
                "Essential genes": ["RNA1 (RdRp)", "RNA2 (CP)", "RdRp (RNA-dependent RNA polymerase)", "CP (coat protein)", "MP (movement protein)"],
                "Cryptic virus features": ["persistent infection", "latent infection", "no symptoms", "vertical transmission", "seed transmission"],
                "Host specificity": ["Cannabis sativa", "hemp", "marijuana", "endophytic", "systemic infection"],
                "Transmission": ["seed transmission", "pollen transmission", "no vector", "vertical transmission", "graft transmission"],
                "Molecular features": ["dsRNA genome", "icosahedral particles", "persistent infection", "no cell-to-cell movement", "replication in cytoplasm"]
            }),
            ("Tomato Ring Spot Virus", "Tomato Ring Spot Virus", {
                "Essential genes": ["RNA polymerase", "helicase", "protease", "replicase"],
                "Structural genes": ["coat protein", "movement protein", "nucleocapsid protein"],
                "Pathogenicity genes": ["silencing suppressor", "host range determinant", "symptom determinant"]
            }),
            ("Tobacco Mosaic Virus", "Tobacco Mosaic Virus", {
                "Essential genes": ["RNA polymerase", "helicase", "protease", "replicase"],
                "Structural genes": ["coat protein", "movement protein", "nucleocapsid protein"],
                "Pathogenicity genes": ["silencing suppressor", "host range determinant", "symptom determinant"]
            }),
            ("Arabis Mosaic Virus", "Arabis Mosaic Virus", {
                "Essential genes": ["RNA polymerase", "helicase", "protease", "replicase"],
                "Structural genes": ["coat protein", "movement protein", "nucleocapsid protein"],
                "Pathogenicity genes": ["silencing suppressor", "host range determinant", "symptom determinant"]
            }),
            ("Tomato Mosaic Virus", "Tomato Mosaic Virus", {
                "Essential genes": ["RNA polymerase", "helicase", "protease", "replicase"],
                "Structural genes": ["coat protein", "movement protein", "nucleocapsid protein"],
                "Pathogenicity genes": ["silencing suppressor", "host range determinant", "symptom determinant"]
            })
        ],
        
        "🍄 Fungus and Pseudofungi": [
            ("Gray mold", "Botrytis cinerea", {
                "Essential genes": ["ACT1 (actin)", "TUB2 (tubulin)", "EF1A (elongation factor)", "RPB2 (RNA polymerase)", "HSP70 (heat shock protein)"],
                "Pathogenicity genes": ["BCG1 (α-galactosidase)", "BMP1 (metalloprotease)", "BCP1 (cerato-platanin)", "BOA1 (botrydial)", "BCR1 (ABC transporter)"],
                "Cell wall degrading": ["BcPG1-6 (polygalacturonases)", "BcPME1 (pectin methylesterase)", "BcXYL1 (xylanase)", "BcCEL1 (cellulase)", "BcCUT1 (cutinase)"],
                "Secondary metabolites": ["BOT1-5 (botrydial cluster)", "DHN1 (1,8-dihydroxynaphthalene)", "PKS1-13 (polyketide synthases)", "NPS1-6 (nonribosomal peptide synthetases)"],
                "Resistance mechanisms": ["ABC1-50 (ABC transporters)", "MFS1-20 (major facilitator superfamily)", "CYP1-20 (cytochrome P450s)", "GST1-10 (glutathione S-transferases)"]
            }),
            ("Pythium root rot", "Pythium myriotylum", {
                "Essential genes": ["ACT1 (actin)", "TUB1 (tubulin)", "EF1A (elongation factor)", "RPB1 (RNA polymerase)", "COX1 (cytochrome oxidase)"],
                "Pathogenicity genes": ["PmNEP1 (necrosis-inducing protein)", "PmPG1 (polygalacturonase)", "PmPME1 (pectin methylesterase)", "PmCUT1 (cutinase)", "PmCP1 (cysteine protease)"],
                "Cell wall degrading": ["PmPG1-5 (polygalacturonases)", "PmPME1 (pectin methylesterase)", "PmXYL1 (xylanase)", "PmCEL1 (cellulase)", "PmCUT1 (cutinase)"],
                "Secondary metabolites": ["PmPKS1-5 (polyketide synthases)", "PmNPS1-3 (nonribosomal peptide synthetases)", "PmTER1-2 (terpene synthases)"],
                "Resistance mechanisms": ["PmCYP1-10 (cytochrome P450s)", "PmGST1-5 (glutathione S-transferases)", "PmABC1-20 (ABC transporters)"]
            }),
            ("Fusarium wilt", "Fusarium oxysporum", {
                "Essential genes": ["ACT1 (actin)", "TUB2 (tubulin)", "EF1A (elongation factor)", "RPB2 (RNA polymerase)", "LSU (large subunit rRNA)"],
                "Pathogenicity genes": ["SIX1-14 (secreted in xylem)", "FTF1 (transcription factor)", "FMK1 (MAPK)", "SGE1 (cutinase)", "PEL1 (pectate lyase)"],
                "Secondary metabolite genes": ["FUM1 (fumonisin biosynthesis)", "TRI5 (trichothecene biosynthesis)", "PKS4 (polyketide synthase)", "BIK1 (bikaverin)", "FUS1 (fusarin)"],
                "Cell wall genes": ["CHI1 (chitinase)", "GEL1 (β-1,3-glucanase)", "CHS1 (chitin synthase)", "FKS1 (β-1,3-glucan synthase)", "PMI1 (mannose-6-phosphate isomerase)"],
                "Resistance targets": ["CYP51 (sterol 14α-demethylase)", "SDH (succinate dehydrogenase)", "QoI (cytochrome bc1)", "MBC (β-tubulin)", "DMI (sterol biosynthesis)"],
                "Formae speciales": ["f.sp. lycopersici (tomato)", "f.sp. cubense (banana)", "f.sp. vasinfectum (cotton)", "f.sp. melonis (melon)", "f.sp. niveum (watermelon)"],
                "Virulence factors": ["Fmk1 (MAPK)", "Ste12 (transcription factor)", "PacC (pH response)", "AreA (nitrogen regulation)", "CreA (carbon catabolite)"],
                "Host interaction": ["adhesion proteins", "penetration enzymes", "colonization factors", "host defense suppression", "nutrient acquisition"]
            }),
            ("Fusarium root rot", "Fusarium solani", {
                "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "HSP70"],
                "Pathogenicity factors": ["FSOL1-10 (F. solani specific)", "CUT1-5 (cutinases)", "PEL1-3 (pectate lyases)", "XYL1-2 (xylanases)", "CEL1-2 (cellulases)"],
                "Secondary metabolites": ["FUM1-3 (fumonisin)", "TRI1-16 (trichothecene)", "ZEA1-2 (zearalenone)", "FUS1-5 (fusarin)", "BIK1-3 (bikaverin)"],
                "Resistance mechanisms": ["CYP51A1-B1", "SDH1-4", "ABC1-20", "MFS1-15", "GST1-10"]
            }),
            ("Powdery mildew", "Golovinomyces ambrosiae", {
                "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "ITS1-2"],
                "Pathogenicity genes": ["GAH1 (haustorium formation)", "GAC1 (conidiophore development)", "GAS1 (spore germination)", "GAP1 (penetration)", "GAA1 (appressorium formation)"],
                "Effectors": ["CSEP1-100 (candidate secreted effector proteins)", "GAE1-50 (G. ambrosiae effectors)", "AVR1-10 (avirulence candidates)", "HAU1-20 (haustorial expressed)"],
                "Sterol biosynthesis": ["CYP51A1", "CYP51B1", "ERG1 (squalene epoxidase)", "ERG7 (lanosterol synthase)", "ERG11 (sterol 14α-demethylase)"]
            }),
            ("Fusarium ear rot", "Fusarium proliferatum", {
                "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "ITS1-2"],
                "Fumonisin biosynthesis": ["FUM1-21 (fumonisin cluster)", "FUM8 (polyketide synthase)", "FUM6 (aminotransferase)", "FUM3 (C-5 oxygenase)", "FUM19 (transporter)"],
                "Pathogenicity": ["FPR1-10 (F. proliferatum specific)", "CUT1-3", "PEL1-2", "XYL1", "CEL1"],
                "Host interaction": ["HOST1-5 (host-specific)", "ADH1-3 (adhesion)", "INV1-2 (invasion)", "COL1-3 (colonization)"]
            })
        ],
        
        "🐛 Insect Pests": [
            ("Two-spotted spider mite", "Tetranychus urticae", {
                "Essential genes": ["ACT1 (actin)", "TUB1 (tubulin)", "EF1A (elongation factor)", "RPL32 (ribosomal protein L32)", "RPS3 (ribosomal protein S3)"],
                "Detoxification genes": ["CYP1-100 (cytochrome P450s)", "GST1-30 (glutathione S-transferases)", "EST1-20 (esterases)", "UGT1-15 (UDP-glucuronosyltransferases)", "ABC1-50 (ABC transporters)"],
                "Acaricide resistance": ["AChE (acetylcholinesterase)", "VGSC (voltage-gated sodium channel)", "RDL (GABA receptor)", "nAChR (nicotinic acetylcholine receptor)", "GluCl (glutamate-gated chloride channel)"],
                "Development genes": ["JH1-3 (juvenile hormone)", "ECR (ecdysone receptor)", "USP (ultraspiracle)", "E74 (ecdysone response)", "BR-C (broad complex)"],
                "Reproduction genes": ["VIT1-6 (vitellogenin)", "VTG1-3 (vitellogenin)", "CHR (chorion)", "EGG1-5 (egg development)", "EMB1-10 (embryogenesis)"]
            }),
            ("Silverleaf whitefly", "Bemisia tabaci", {
                "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "16S rRNA"],
                "Insecticide resistance": ["CYP6CM1", "CYP4C64", "CYP4G61", "CYP4G70", "GST1-15", "EST1-10", "ABC1-30", "nAChR (nicotinic receptor)", "VGSC (sodium channel)"],
                "Biotype markers": ["mtCOI (mitochondrial COI)", "ITS1", "16S rRNA", "28S rRNA", "RAPD markers"],
                "Virus transmission": ["GroEL (endosymbiont)", "HSP70", "cyclophilin", "importin-α", "karyopherin", "VP1-4 (viral proteins)"],
                "Endosymbiont genes": ["Portiera (P-endosymbiont)", "Hamiltonella", "Arsenophonus", "Cardinium", "Wolbachia", "Rickettsia"]
            })
        ]
    }

LOCAL_CATALOG = get_organism_suggestions_with_gene_targets()

# No hardcoded sequences - rely entirely on NCBI fetching like autoprimer5.py
LOCAL_SEQUENCES = {}

@st.cache_data
def load_catalog() -> Any:
    """Return the embedded local catalog by default (no imports required)."""
    return LOCAL_CATALOG


def clean_seq(s: str) -> str:
    return re.sub(r"[^ACGTNacgtn]", "", s).upper().replace("U", "T")

# ---- NCBI fetching (optional, like the original app) ----
try:
    from Bio import Entrez
    HAVE_ENTREZ = True
except Exception:
    HAVE_ENTREZ = False

@st.cache_data(show_spinner=False)
def ncbi_search_and_fetch(organism: str, gene: str, email: str, api_key: Optional[str], db: str = "nucleotide", retmax: int = 1) -> Optional[str]:
    """Search NCBI for a nucleotide record for organism+gene and return FASTA sequence (first hit)."""
    import urllib.parse, urllib.request
    import json

    def eutils_get(url, params):
        q = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{q}") as resp:
            return resp.read().decode()

    # Use Biopython if available
    if HAVE_ENTREZ:
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        handle = Entrez.esearch(db=db, term=f"{organism}[Organism] AND {gene}[Gene]", retmax=retmax)
        rec = Entrez.read(handle)
        ids = rec.get("IdList", [])
        if not ids:
            return None
        fetch = Entrez.efetch(db=db, id=ids[0], rettype="fasta", retmode="text")
        fasta = fetch.read()
        seq = "".join([line.strip() for line in fasta.splitlines() if not line.startswith(">")])
        return clean_seq(seq) if seq else None

    # Fallback to raw E-utilities (no Biopython)
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_xml = eutils_get(f"{base}/esearch.fcgi", {
        "db": db, "term": f"{organism}[Organism] AND {gene}[Gene]", "retmax": str(retmax),
        "retmode": "json", **({"api_key": api_key} if api_key else {})
    })
    data = json.loads(search_xml)
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return None
    fasta = eutils_get(f"{base}/efetch.fcgi", {
        "db": db, "id": ids[0], "rettype": "fasta", "retmode": "text", **({"api_key": api_key} if api_key else {})
    })
    seq = "".join([line.strip() for line in fasta.splitlines() if not line.startswith(">")])
    return clean_seq(seq) if seq else None

# -----------------------------
# Sequence extractor (prefers local, then NCBI if enabled)
# -----------------------------

def find_seq_in_obj(obj: Any, key_hints: List[str]) -> Optional[str]:
    # direct string
    if isinstance(obj, str):
        s = clean_seq(obj)
        if len(s) >= 60:
            return s
        return None
    if isinstance(obj, dict):
        for k in key_hints:
            if k in obj and isinstance(obj[k], str):
                s = clean_seq(obj[k])
                if len(s) >= 60:
                    return s
        for v in obj.values():
            s = find_seq_in_obj(v, key_hints)
            if s:
                return s
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            s = find_seq_in_obj(v, key_hints)
            if s:
                return s
    return None

@st.cache_data
def flatten_catalog_with_sequences(catalog: Any, key_hints_csv: str, email: str = "", api_key: Optional[str] = None, enable_ncbi: bool = False) -> List[Dict[str,str]]:
    """Flatten catalog into entries with (organism, target, sequence).
    Order of preference: LOCAL_SEQUENCES -> embedded sequence fields -> NCBI fetch (if enabled).
    """
    key_hints = [k.strip() for k in key_hints_csv.split(',') if k.strip()]
    out: List[Dict[str,str]] = []
    missing: List[Tuple[str,str]] = []

    def push_entry(org: str, tgt: str, seq: Optional[str], path: str, label_hint: Optional[str] = None):
        label = label_hint or f"{org} — {tgt}"
        if seq:
            out.append({"label": label, "organism": org, "target": tgt, "sequence": seq, "path": path})
        else:
            missing.append((org, tgt))

    def walk(obj: Any, path: str = "$", category: Optional[str] = None, subcat: Optional[str] = None):
        if obj is None:
            return
        
        # Handle the new catalog structure: (common_name, scientific_name, gene_targets)
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            common_name, scientific_name, gene_targets = obj
            if isinstance(gene_targets, dict):
                # Create entries for each gene category
                for gene_category, genes in gene_targets.items():
                    for gene in genes:
                        # Extract gene name (remove description in parentheses)
                        gene_name = gene.split('(')[0].strip()
                        label = f"{scientific_name} — {gene_name}"
                        push_entry(scientific_name, gene_name, None, f"{path}.{gene_category}.{gene_name}", label)
                return
        
        # If this node has an explicit sequence, infer labels
        seq_here = find_seq_in_obj(obj, key_hints)
        if seq_here:
            organism = None
            target = None
            if isinstance(obj, dict):
                organism = obj.get("scientific") or obj.get("organism") or obj.get("species") or obj.get("name") or obj.get("common") or category
                target = obj.get("target") or obj.get("gene") or obj.get("marker") or obj.get("locus") or subcat or "Target"
            elif isinstance(obj, (list, tuple)) and len(obj) >= 3 and all(isinstance(x, str) for x in obj[:3]):
                organism = obj[1]
                target = obj[2]
            push_entry(organism or "Unknown", target or "Target", seq_here, path)
            return
        # Otherwise recurse; when we find a leaf dict with organism + target but no seq, try LOCAL then NCBI
        if isinstance(obj, dict):
            # Detect a leaf with organism/target names
            if any(k in obj for k in ("scientific","organism","species","name","common")) and any(k in obj for k in ("target","gene","marker","locus")):
                org = obj.get("scientific") or obj.get("organism") or obj.get("species") or obj.get("name") or obj.get("common") or category or "Unknown"
                tgt = obj.get("target") or obj.get("gene") or obj.get("marker") or obj.get("locus") or subcat or "Target"
                # 1) local override
                seq = LOCAL_SEQUENCES.get((org, tgt)) or None
                if seq:
                    seq = clean_seq(seq)
                # 2) embedded fields
                if not seq:
                    seq = find_seq_in_obj(obj, key_hints)
                # 3) NCBI fetch
                if not seq and enable_ncbi and email:
                    seq = ncbi_search_and_fetch(org, tgt, email=email, api_key=api_key)
                push_entry(org, tgt, seq, path)
                return
            for k, v in obj.items():
                if category is None:
                    walk(v, f"{path}.{k}", category=k, subcat=None)
                elif subcat is None and isinstance(v, (list, dict, tuple)):
                    walk(v, f"{path}.{k}", category=category, subcat=k)
                else:
                    walk(v, f"{path}.{k}", category=category, subcat=subcat)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", category=category, subcat=subcat)

    walk(catalog)
    # Deduplicate entries
    seen = set()
    dedup = []
    for e in out:
        key = (e["organism"], e["target"], e["sequence"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    # Attach missing info for UI
    dedup.append({"_missing": missing})
    return dedup

# -----------------------------
# Thermo + tuning helpers (coarse but fast)
# -----------------------------

def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq: return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def rough_amp_tm(seq: str, monovalent_mM: float = 50.0, free_Mg_mM: float = 2.0) -> float:
    L = max(len(seq), 1)
    base = 69.3 + 0.41 * (gc_content(seq) * 100.0) - (650.0 / L)
    Na = max(monovalent_mM, 1e-3) / 1000.0
    salt = 16.6 * math.log10(Na)
    mg_adj = 0.6 * max(free_Mg_mM, 0.0)
    return base + salt + mg_adj

GC_TAILS = ["", "G", "GC", "GCG", "GCGC", "GCGCG", "GCGCGC"]
AT_TAILS = ["", "A", "AT", "ATAT", "ATATAT", "ATATATAT"]

@dataclass
class Candidate:
    fwd: str
    rev: str
    prod_len: int
    base_tm: float

@dataclass
class AssignmentChoice:
    candidate: Candidate
    f_tail: str
    r_tail: str
    tm_after: float
    cost: float

@dataclass
class PrimerPair:
    fwd: str
    rev: str
    product_len: int
    product_tm: float
    channel: str
    slot_tm: float
    slot_index: int
    organism: str
    target_name: str
    tails: Tuple[str, str]
    issues: List[str]

# -----------------------------
# NCBI Connector
# -----------------------------
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=0.34):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class ResilientNCBIConnector:
    """Resilient NCBI connector with exponential backoff and retry logic"""
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        if HAVE_ENTREZ:
            Entrez.email = email
            Entrez.tool = "autoprimertm"
            if api_key:
                Entrez.api_key = api_key
        self.base_delay = 0.34 if not api_key else 0.1
        self.max_retries = 3
        self.timeout = 30
        self.throttle_sec = 0.34
    
    def search_sequences(self, query: str, database: str = "nucleotide", max_results: int = 100) -> List[str]:
        """Search with timeout and retry logic"""
        if not HAVE_ENTREZ or not HAVE_BIO:
            return []
        
        time.sleep(self.throttle_sec)
        
        try:
            handle = Entrez.esearch(
                db=database, 
                term=query, 
                retmax=max_results,
                timeout=self.timeout
            )
            search_results = Entrez.read(handle)
            handle.close()
            return search_results["IdList"]
        except Exception as e:
            st.error(f"NCBI search error: {e}")
            return []
    
    @retry_with_backoff(max_retries=3, base_delay=0.34)
    def fetch_sequence(self, seq_id: str, database: str = "nucleotide") -> Optional[str]:
        """Fetch with timeout and retry logic"""
        if not HAVE_ENTREZ or not HAVE_BIO:
            return None
            
        try:
            handle = Entrez.efetch(
                db=database, 
                id=seq_id, 
                rettype="fasta", 
                retmode="text",
                timeout=self.timeout
            )
            record = SeqIO.read(handle, "fasta")
            handle.close()
            return str(record.seq)
        except Exception as e:
            st.error(f"NCBI fetch error: {e}")
            return None
    
    def fetch_organism_sequences(self, organism_name: str, max_sequences: int = 10) -> List[Dict]:
        """Fetch multiple sequences for an organism with metadata using autoprimer5.py methodology"""
        if not HAVE_ENTREZ or not HAVE_BIO or not organism_name:
            st.write(f"🔍 Debug: Cannot fetch for {organism_name} - HAVE_ENTREZ={HAVE_ENTREZ}, HAVE_BIO={HAVE_BIO}")
            return []
            
        try:
            st.write(f"🔍 Debug: Starting fetch for {organism_name}")
            # Get gene targets for this organism
            suggestions = get_organism_suggestions_with_gene_targets()
            organism_targets = None
            
            # Find matching organism and extract gene targets
            for category, organisms in suggestions.items():
                for item in organisms:
                    if len(item) == 3:
                        common_name, scientific_name, gene_targets = item
                        if (scientific_name.lower() == organism_name.lower() or 
                            organism_name.lower() in scientific_name.lower() or
                            scientific_name.lower() in organism_name.lower()):
                            organism_targets = {
                                'organism': scientific_name,
                                'common_name': common_name,
                                'gene_targets': gene_targets
                            }
                            st.write(f"🔍 Debug: Found organism targets for {organism_name}")
                            break
                if organism_targets:
                    break
            
            if not organism_targets:
                st.write(f"🔍 Debug: No organism targets found for {organism_name}")
            
            sequences = []
            
            if organism_targets:
                # Fetch sequences for specific gene targets using enhanced search
                gene_targets = organism_targets['gene_targets']
                for gene_category, genes in gene_targets.items():
                    for gene in genes[:2]:  # Limit to 2 genes per category
                        gene_name = gene.split('(')[0].strip()
                        
                        # Enhanced search queries like autoprimer5.py
                        search_queries = [
                            f'"{organism_name}"[organism] AND {gene_name}[gene]',
                            f'"{organism_name}"[organism] AND {gene_name}[title]',
                            f'"{organism_name}"[organism] AND {gene_name}[All Fields]',
                            f'"{organism_name}"[organism] AND {gene_name}[mRNA]',
                            f'"{organism_name}"[organism] AND {gene_name}[CDS]'
                        ]
                        
                        seq_ids = []
                        for query in search_queries:
                            try:
                                st.write(f"🔍 Debug: Searching with query: {query}")
                                results = self.search_sequences(query, database="nucleotide", max_results=2)
                                if results:
                                    seq_ids = results
                                    st.write(f"🔍 Debug: Found {len(results)} sequence IDs")
                                    break
                                else:
                                    st.write(f"🔍 Debug: No results for query: {query}")
                            except Exception as e:
                                st.write(f"🔍 Debug: Search error for {query}: {e}")
                                continue
                        
                        # If no nucleotide hits, try protein-to-nucleotide linking
                        if not seq_ids:
                            seq_ids = self._find_nuccore_via_protein(organism_name, [gene_name], max_results=2)
                        
                        for seq_id in seq_ids:
                            st.write(f"🔍 Debug: Fetching sequence {seq_id}")
                            sequence = self.fetch_sequence(seq_id)
                            if sequence and len(sequence) > 100:
                                # Clean sequence
                                clean_seq = re.sub(r'[^ATGCatgc]', '', sequence.upper())
                                if len(clean_seq) >= 100:
                                    sequences.append({
                                        "id": seq_id,
                                        "organism": organism_name,
                                        "target": gene_name,
                                        "sequence": clean_seq[:10000],  # Limit to 10kb like autoprimer5.py
                                        "label": f"{organism_name} — {gene_name}"
                                    })
                                    st.write(f"🔍 Debug: Added sequence {seq_id} ({len(clean_seq)} bp)")
                                    if len(sequences) >= max_sequences:
                                        return sequences
                            else:
                                st.write(f"🔍 Debug: Sequence {seq_id} too short or invalid")
            else:
                # Fallback: general organism search
                search_query = f'"{organism_name}"[organism]'
                seq_ids = self.search_sequences(
                    search_query, 
                    database="nucleotide", 
                    max_results=max_sequences
                )
                
                for seq_id in seq_ids[:max_sequences]:
                    sequence = self.fetch_sequence(seq_id)
                    if sequence:
                        clean_seq = re.sub(r'[^ATGCatgc]', '', sequence.upper())
                        if len(clean_seq) >= 100:
                            sequences.append({
                                "id": seq_id,
                                "organism": organism_name,
                                "target": f"sequence_{seq_id}",
                                "sequence": clean_seq[:10000],
                                "label": f"{organism_name} — sequence_{seq_id}"
                            })
            
            return sequences
            
        except Exception as e:
            st.error(f"Error fetching sequences: {e}")
            return []
    
    def _find_nuccore_via_protein(self, organism_name: str, gene_variations: List[str], max_results: int = 5) -> List[str]:
        """Find nucleotide sequences via protein database linking (from autoprimer5.py)"""
        if not HAVE_ENTREZ or not HAVE_BIO:
            return []
        try:
            ids = []
            for gene_var in gene_variations:
                q = f'"{organism_name}"[Organism] AND ({gene_var}[All Fields])'
                handle = Entrez.esearch(db="protein", term=q, retmax=max_results)
                rec = Entrez.read(handle)
                handle.close()
                if not rec.get("IdList"):
                    continue
                for pid in rec["IdList"]:
                    link = Entrez.elink(dbfrom="protein", db="nuccore", id=pid)
                    link_rec = Entrez.read(link)
                    link.close()
                    for lset in link_rec:
                        for linksetdb in lset.get("LinkSetDb", []):
                            if linksetdb.get("DbTo") == "nuccore":
                                ids.extend([x["Id"] for x in linksetdb.get("Link", [])])
            return ids[:max_results]
        except Exception:
            return []

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AutoPrimer5 — Multiplex (multiplexprimers.py)", layout="wide")
st.title("AutoPrimer5 — Multiplex (15 targets • 3 channels × 5 Tm slots)")

with st.sidebar:
    st.header("NCBI Configuration")
    ncbi_email = st.text_input("NCBI Email", value="your.email@example.com", help="Required for NCBI API access")
    ncbi_api_key = st.text_input("NCBI API Key (optional)", type="password", help="Optional API key for higher rate limits")
    
    st.header("Sequence Fetching")
    
    # Add button to fetch sequences from NCBI for all 15 targets
    if ncbi_email and ncbi_email != "your.email@example.com":
        if st.button("🔄 Fetch Sequences for All 15 Targets", type="primary"):
            # Reset auto-fetch flag to allow manual fetching
            if 'auto_fetch_attempted' in st.session_state:
                del st.session_state['auto_fetch_attempted']
            
            with st.spinner("Fetching sequences for all 15 target organisms..."):
                try:
                    ncbi = ResilientNCBIConnector(ncbi_email, ncbi_api_key if ncbi_api_key else None)
                    
                    # Get the actual selected targets from the main section (same as auto-fetch)
                    all_organisms = []
                    suggestions = get_organism_suggestions_with_gene_targets()
                    for category, organisms in suggestions.items():
                        for item in organisms:
                            if len(item) == 3:
                                common_name, scientific_name, gene_targets = item
                                all_organisms.append({
                                    "common_name": common_name,
                                    "scientific_name": scientific_name,
                                    "gene_targets": gene_targets,
                                    "label": f"{scientific_name} ({common_name})"
                                })
                    
                    all_organisms = all_organisms[:15]  # Take first 15
                    
                    # Fetch sequences for each organism
                    all_ncbi_sequences = []
                    for i, organism in enumerate(all_organisms):
                        organism_name = organism['scientific_name']
                        st.write(f"Fetching sequences for {organism_name}...")
                        ncbi_sequences = ncbi.fetch_organism_sequences(organism_name, max_sequences=3)
                        if ncbi_sequences:
                            all_ncbi_sequences.extend(ncbi_sequences)
                            st.success(f"Found {len(ncbi_sequences)} sequences for {organism_name}")
                        else:
                            st.warning(f"No sequences found for {organism_name}")
                    
                    if all_ncbi_sequences:
                        st.success(f"Successfully fetched {len(all_ncbi_sequences)} total sequences!")
                        # Store in session state for use in the main app
                        st.session_state['ncbi_sequences'] = all_ncbi_sequences
                        st.rerun()
                    else:
                        st.warning("No sequences found for any organisms")
                except Exception as e:
                    st.error(f"Error fetching sequences: {e}")
    
    # Add reset button to clear auto-fetch flag
    if 'auto_fetch_attempted' in st.session_state:
        if st.button("🔄 Reset Auto-Fetch", help="Clear auto-fetch flag to allow automatic fetching again"):
            del st.session_state['auto_fetch_attempted']
            st.rerun()
    
    # Auto-fetch sequences if email is provided and no sequences are available yet
    if (ncbi_email and ncbi_email != "your.email@example.com" and 
        ('ncbi_sequences' not in st.session_state or not st.session_state['ncbi_sequences'])):
        
        # Debug info
        st.info(f"🔍 Debug: Email='{ncbi_email}', NCBI sequences in session: {'ncbi_sequences' in st.session_state}")
        
        # Check if we should auto-fetch (only if user hasn't explicitly fetched yet)
        if 'auto_fetch_attempted' not in st.session_state:
            st.session_state['auto_fetch_attempted'] = True
            
            with st.spinner("🔄 Auto-fetching sequences for all 15 targets..."):
                try:
                    # Check if BioPython is available
                    if not HAVE_ENTREZ or not HAVE_BIO:
                        st.error("❌ BioPython not available. Please install: pip install biopython")
                        st.stop()
                    
                    # Test BioPython connection
                    st.write("🔍 Debug: Testing BioPython connection...")
                    try:
                        from Bio import Entrez
                        Entrez.email = ncbi_email
                        test_handle = Entrez.esearch(db="nucleotide", term="test", retmax=1)
                        test_result = Entrez.read(test_handle)
                        test_handle.close()
                        st.write("✅ BioPython connection test successful")
                    except Exception as e:
                        st.error(f"❌ BioPython connection test failed: {e}")
                        st.stop()
                    
                    ncbi = ResilientNCBIConnector(ncbi_email, ncbi_api_key if ncbi_api_key else None)
                    
                    # Get the actual selected targets from the main section
                    # We need to access the selected gene targets that were created in the main section
                    # Since we can't directly access them here, we'll recreate the same logic
                    all_organisms = []
                    suggestions = get_organism_suggestions_with_gene_targets()
                    for category, organisms in suggestions.items():
                        for item in organisms:
                            if len(item) == 3:
                                common_name, scientific_name, gene_targets = item
                                all_organisms.append({
                                    "common_name": common_name,
                                    "scientific_name": scientific_name,
                                    "gene_targets": gene_targets,
                                    "label": f"{scientific_name} ({common_name})"
                                })
                    
                    all_organisms = all_organisms[:15]  # Take first 15
                    st.write(f"🔍 Debug: Found {len(all_organisms)} organisms to fetch")
                    
                    all_ncbi_sequences = []
                    for i, organism in enumerate(all_organisms):
                        organism_name = organism['scientific_name']
                        st.write(f"Fetching sequences for {organism_name}...")
                        ncbi_sequences = ncbi.fetch_organism_sequences(organism_name, max_sequences=3)
                        if ncbi_sequences:
                            all_ncbi_sequences.extend(ncbi_sequences)
                            st.write(f"✅ Found {len(ncbi_sequences)} sequences for {organism_name}")
                        else:
                            st.write(f"⚠️ No sequences found for {organism_name}")
                    
                    if all_ncbi_sequences:
                        st.session_state['ncbi_sequences'] = all_ncbi_sequences
                        st.success(f"✅ Auto-fetched {len(all_ncbi_sequences)} sequences!")
                        st.rerun()
                    else:
                        st.warning("⚠️ No sequences found during auto-fetch. Try manual fetch.")
                except Exception as e:
                    st.error(f"❌ Auto-fetch failed: {e}")
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
        else:
            st.info("🔄 Auto-fetch already attempted. Use 'Reset Auto-Fetch' button to retry.")
    
    st.header("Catalog & Conditions")
    seq_key_hints = st.text_input("Sequence field key(s) (comma-separated)", value="sequence,target_sequence,amplicon,region,locus_seq,seq")
    catalog_obj = load_catalog()
    
    st.header("Debug Options")
    debug_mode = st.checkbox("Enable Debug Output", value=False, help="Show detailed debug information during primer design")
    
    # Use NCBI sequences if available, otherwise use local catalog
    if 'ncbi_sequences' in st.session_state and st.session_state['ncbi_sequences']:
        entries = st.session_state['ncbi_sequences']
    else:
        entries = flatten_catalog_with_sequences(catalog_obj, seq_key_hints, ncbi_email, ncbi_api_key, enable_ncbi=True)
    st.metric("Catalog entries (raw)", 0 if catalog_obj is None else (len(catalog_obj) if hasattr(catalog_obj, "__len__") else "?"))
    st.metric("Entries with sequences", len(entries))
    if len(entries) == 0:
        st.warning("No sequences found. Click 'Fetch Sequences for All 15 Targets' to get sequences from NCBI.")
    monovalent_mM = st.number_input("Monovalent salt (mM)", 10.0, 200.0, 50.0, 1.0)
    free_Mg_mM   = st.number_input("Free Mg²⁺ (mM)", 0.0, 6.0, 2.0, 0.1)

    st.markdown("**Tm slots per channel (°C)** — keep ≥3 °C spacing")
    default_grid = {
        "FAM": [79.5, 83.0, 86.5, 90.0, 93.5],
        "HEX": [78.0, 81.5, 85.0, 88.5, 92.0],
        "Cy5": [80.5, 84.0, 87.5, 91.0, 94.5],
    }
    grid = {}
    for dye in ["FAM", "HEX", "Cy5"]:
        cols = st.columns(5)
        grid[dye] = []
        for i, col in enumerate(cols):
            with col:
                grid[dye].append(st.number_input(f"{dye} s{i+1}", 70.0, 100.0, default_grid[dye][i], 0.1, key=f"{dye}_{i}"))

st.markdown("### 1) **15 Target Organisms** (automatically populated)")

# Get all 15 organisms from the catalog
all_organisms = []
suggestions = get_organism_suggestions_with_gene_targets()
for category, organisms in suggestions.items():
    for item in organisms:
        if len(item) == 3:
            common_name, scientific_name, gene_targets = item
            all_organisms.append({
                "common_name": common_name,
                "scientific_name": scientific_name,
                "gene_targets": gene_targets,
                "label": f"{scientific_name} ({common_name})"
            })

# Ensure we have exactly 15 organisms
if len(all_organisms) < 15:
    st.error(f"Only {len(all_organisms)} organisms found, need 15")
else:
    all_organisms = all_organisms[:15]  # Take first 15

# Create the target selection interface
selected_organisms = []
selected_gene_targets = []

for i in range(15):
    organism = all_organisms[i]
    
    # Organism selection (read-only, automatically assigned)
    st.write(f"**Target {i+1}:** {organism['label']}")
    
    # Multiple gene target category selection (like autoprimer5.py)
    gene_categories = list(organism['gene_targets'].keys())
    if gene_categories:
        # Create organism-specific session state key
        organism_key = f"selected_categories_{organism['scientific_name'].replace(' ', '_')}"
        
        # Initialize default selection if not exists
        if organism_key not in st.session_state:
            # Default to first category (usually "Essential genes")
            default_categories = [gene_categories[0]] if gene_categories else []
            st.session_state[organism_key] = default_categories
        
        # Use organism-specific stored selection
        default_categories = st.session_state.get(organism_key, [])
        
        selected_categories = st.multiselect(
            f"Choose gene categories for {organism['common_name']}:",
            gene_categories,
            default=default_categories,
            help=f"Select which gene categories to target for {organism['scientific_name']}. Essential genes are recommended for reliable detection."
        )
        
        # Update organism-specific session state
        st.session_state[organism_key] = selected_categories
        
        if selected_categories:
            # Display selection summary
            total_genes = sum(len(organism['gene_targets'][cat]) for cat in selected_categories)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Selected Categories", len(selected_categories))
            with col2:
                st.metric("Total Gene Targets", total_genes)
            
            # Show selected targets
            with st.expander(f"📋 Selected Gene Targets for {organism['common_name']}", expanded=False):
                selected_genes = []
                for category in selected_categories:
                    st.write(f"**{category}:**")
                    genes = organism['gene_targets'][category]
                    for gene in genes:
                        st.write(f"  • {gene}")
                        selected_genes.append(f"{category}: {gene}")
                    st.write("")
            
            # Create entries for each selected category and gene
            for category in selected_categories:
                genes_in_category = organism['gene_targets'][category]
                for gene in genes_in_category:
                    selected_organisms.append(organism)
                    selected_gene_targets.append({
                        "organism": organism,
                        "category": category,
                        "gene": gene,
                        "label": f"{organism['scientific_name']} — {gene.split('(')[0].strip()}"
                    })
        else:
            # No categories selected, use general fallback
            selected_organisms.append(organism)
            selected_gene_targets.append({
                "organism": organism,
                "category": "General",
                "gene": "General",
                "label": f"{organism['scientific_name']} — General"
            })
    else:
        selected_organisms.append(organism)
        selected_gene_targets.append({
            "organism": organism,
            "category": "General",
            "gene": "General",
            "label": f"{organism['scientific_name']} — General"
        })
    
    # Manual sequence input option
    st.markdown("**📝 Manual Sequence Input (Optional):**")
    manual_sequence = st.text_area(
        f"Enter nucleotide sequence for {organism['common_name']}:",
        placeholder="ATCGATCGATCG... (leave empty to use NCBI sequences)",
        key=f"manual_seq_{i}",
        height=100,
        help="Enter a custom nucleotide sequence. If provided, this will override NCBI sequences for this target."
    )
    
    # Store manual sequence in session state for later use
    if manual_sequence and manual_sequence.strip():
        # Clean the sequence (remove whitespace, convert to uppercase)
        clean_sequence = ''.join(manual_sequence.split()).upper()
        # Validate that it contains only valid nucleotides
        if all(c in 'ATCGN' for c in clean_sequence):
            st.session_state[f'manual_sequence_{i}'] = clean_sequence
            st.success(f"✅ Custom sequence saved for {organism['common_name']} ({len(clean_sequence)} bp)")
        else:
            st.error("❌ Invalid sequence. Only A, T, C, G, and N are allowed.")
            st.session_state[f'manual_sequence_{i}'] = None
    else:
        st.session_state[f'manual_sequence_{i}'] = None
    
    st.markdown("---")

# Create entries for the rest of the app to use
# IMPORTANT: Ensure only one organism per target slot for multiplex PCR
entries = []
seen_organisms = set()

for i, target_info in enumerate(selected_gene_targets):
    organism_name = target_info["organism"]["scientific_name"]
    
    # Skip if we've already seen this organism (one organism per slot)
    if organism_name in seen_organisms:
        st.write(f"🔧 Debug: Skipping duplicate organism '{organism_name}' - already assigned to a slot")
        continue
    
    # Check if manual sequence is provided for this target
    manual_seq = st.session_state.get(f'manual_sequence_{i}')
    
    entries.append({
        "label": target_info["label"],
        "organism": organism_name,
        "target": target_info["gene"].split('(')[0].strip(),
        "sequence": manual_seq,  # Use manual sequence if available, otherwise None (will be fetched from NCBI)
        "path": f"auto_target_{organism_name}",
        "source": "manual" if manual_seq else "ncbi"
    })
    
    # Mark this organism as seen
    seen_organisms.add(organism_name)

# Show organism deduplication summary
st.write(f"🔧 Debug: Organism deduplication complete - {len(entries)} unique organisms selected from {len(selected_gene_targets)} total targets")
st.write(f"🔧 Debug: Selected organisms: {list(seen_organisms)}")

# Check if we have enough unique organisms for 15 slots
if len(entries) < 15:
    st.warning(f"⚠️ **Only {len(entries)} unique organisms available for 15 slots. Consider selecting more diverse organisms.**")
elif len(entries) > 15:
    st.info(f"ℹ️ **{len(entries)} unique organisms available, will use first 15 for multiplex design.**")
    entries = entries[:15]  # Limit to 15 for multiplex design

# Auto-fetch logic will be handled in the sidebar section

# If NCBI sequences are available, use them to populate entries that don't have manual sequences
if 'ncbi_sequences' in st.session_state and st.session_state['ncbi_sequences']:
    ncbi_sequences = st.session_state['ncbi_sequences']
    
    # Match NCBI sequences to our target organisms (only for entries without manual sequences)
    for entry in entries:
        # Skip if this entry already has a manual sequence
        if entry.get("source") == "manual" and entry.get("sequence"):
            continue
            
        organism_name = entry["organism"]
        target_gene = entry["target"]
        
        # Find matching sequence from NCBI results
        for ncbi_seq in ncbi_sequences:
            if (organism_name.lower() in ncbi_seq.get("organism", "").lower() or 
                ncbi_seq.get("organism", "").lower() in organism_name.lower()):
                if target_gene.lower() in ncbi_seq.get("target", "").lower() or target_gene == "General":
                    entry["sequence"] = ncbi_seq.get("sequence")
                    entry["id"] = ncbi_seq.get("id")
                    entry["source"] = "ncbi"
                    break

selected = entries

# Show sequence source summary
st.markdown("### 📊 **Sequence Source Summary**")
manual_count = sum(1 for entry in entries if entry.get("source") == "manual" and entry.get("sequence"))
ncbi_count = sum(1 for entry in entries if entry.get("source") == "ncbi" and entry.get("sequence"))
missing_count = sum(1 for entry in entries if not entry.get("sequence"))
total_targets = len(entries)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📝 Manual Sequences", manual_count)
with col2:
    st.metric("🌐 NCBI Sequences", ncbi_count)
with col3:
    st.metric("❌ Missing Sequences", missing_count)
with col4:
    st.metric("🎯 Total Targets", total_targets)

if missing_count > 0:
    st.warning(f"⚠️ {missing_count} targets still need sequences. Use manual input or fetch from NCBI.")

# Show breakdown by organism
st.markdown("### 🧬 **Target Breakdown by Organism**")
organism_summary = {}
for entry in entries:
    organism = entry.get("organism", "Unknown")
    if organism not in organism_summary:
        organism_summary[organism] = {"total": 0, "manual": 0, "ncbi": 0, "missing": 0}
    
    organism_summary[organism]["total"] += 1
    if entry.get("source") == "manual" and entry.get("sequence"):
        organism_summary[organism]["manual"] += 1
    elif entry.get("source") == "ncbi" and entry.get("sequence"):
        organism_summary[organism]["ncbi"] += 1
    else:
        organism_summary[organism]["missing"] += 1

for organism, counts in organism_summary.items():
    with st.expander(f"📋 {organism} ({counts['total']} targets)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", counts["total"])
        with col2:
            st.metric("Manual", counts["manual"])
        with col3:
            st.metric("NCBI", counts["ncbi"])
        with col4:
            st.metric("Missing", counts["missing"])

st.markdown("### 2) Primer design settings & constraints")
colA, colB, colC = st.columns(3)
with colA:
    opt_size = st.slider("Primer size (optimal)", 15, 30, 20)
    min_size = st.slider("Primer size (min)", 15, 25, 18)
    max_size = st.slider("Primer size (max)", 20, 35, 25)
with colB:
    anneal_tm = st.slider("Primer anneal Tm target (°C)", 50.0, 72.0, 60.0, 0.5)
    product_len_min, product_len_max = st.slider("Desired product length (bp)", 60, 240, (100, 170))
with colC:
    tail_penalty = st.slider("Tail length penalty (°C per nt, scoring)", 0.00, 0.10, 0.02, 0.01)
    delta_penalty = st.slider("ΔTm to slot weight (×)", 0.5, 3.0, 1.0, 0.1)

st.markdown("### 3) Auto-assign to **3 channels × 5 Tm slots** and design")

# Define the temperature grid for 3 channels × 5 slots
grid = {
    "FAM": [79.5, 83.0, 86.5, 90.0, 93.5],    # Channel 1: 79.5°C to 93.5°C
    "HEX": [80.0, 83.5, 87.0, 90.5, 94.0],    # Channel 2: 80.0°C to 94.0°C  
    "Cy5": [80.5, 84.0, 87.5, 91.0, 94.5]     # Channel 3: 80.5°C to 94.5°C
}

slot_list: List[Tuple[str,int,float]] = []  # (dye, slot_index, slot_tm)
for dye in ["FAM","HEX","Cy5"]:
    for k in range(5):
        slot_list.append((dye, k, grid[dye][k]))

# Primer3 design for multiple candidates per target

def design_candidates(seq: str, n_return: int = 6) -> List[Candidate]:
    cands: List[Candidate] = []
    if not seq or len(seq) < min_size + 30:
        return cands
    if HAVE_PRIMER3:
        params = dict(
            SEQUENCE_TEMPLATE=seq,
            PRIMER_OPT_SIZE=opt_size,
            PRIMER_MIN_SIZE=min_size,
            PRIMER_MAX_SIZE=max_size,
            PRIMER_NUM_RETURN=n_return,
            PRIMER_PRODUCT_SIZE_RANGE=[[product_len_min, product_len_max]],
            PRIMER_OPT_TM=anneal_tm,
            PRIMER_MIN_TM=anneal_tm-2.5,
            PRIMER_MAX_TM=anneal_tm+2.5,
            PRIMER_MAX_POLY_X=4,
            PRIMER_GC_CLAMP=1,
        )
        try:
            res = primer3.bindings.designPrimers({"SEQUENCE_TEMPLATE": seq}, params)
            n = int(res.get("PRIMER_PAIR_NUM_RETURNED", 0))
            for idx in range(n):
                fwd = res.get(f"PRIMER_LEFT_{idx}_SEQUENCE")
                rev = res.get(f"PRIMER_RIGHT_{idx}_SEQUENCE")
                if not (fwd and rev):
                    continue
                core = seq
                if len(seq) > (len(fwd)+len(rev)):
                    core = seq[len(fwd):-(len(rev))]
                amp = f"{fwd}{core}{rev}"
                cands.append(Candidate(fwd=fwd, rev=rev, prod_len=len(amp), base_tm=rough_amp_tm(amp, monovalent_mM, free_Mg_mM)))
        except Exception:
            pass
    if not cands:
        # naive fallback: terminal slices
        fwd = seq[:opt_size]
        rev = Seq(seq[-opt_size:]).reverse_complement() if HAVE_BIO else seq[-opt_size:][::-1]
        core = seq[opt_size:-opt_size] if len(seq)>(2*opt_size) else ""
        amp = f"{fwd}{core}{rev}"
        cands.append(Candidate(fwd=str(fwd), rev=str(rev), prod_len=len(amp), base_tm=rough_amp_tm(amp, monovalent_mM, free_Mg_mM)))
    return cands

# Evaluate best choice of (candidate, tails) for a given slot

def best_choice_for_slot(seq: str, slot_tm: float) -> Tuple[Optional[Candidate], Optional[AssignmentChoice]]:
    cands = design_candidates(seq)
    best: Optional[AssignmentChoice] = None
    best_cand: Optional[Candidate] = None
    for cand in cands:
        core = seq
        if len(seq) > (len(cand.fwd)+len(cand.rev)):
            core = seq[len(cand.fwd):-(len(cand.rev))]
        for tF in GC_TAILS + AT_TAILS:
            for tR in GC_TAILS + AT_TAILS:
                amp2 = f"{tF}{cand.fwd}{core}{cand.rev}{tR}"
                tm2 = rough_amp_tm(amp2, monovalent_mM, free_Mg_mM)
                cost = delta_penalty*abs(tm2 - slot_tm) + tail_penalty*(len(tF)+len(tR))
                choice = AssignmentChoice(candidate=cand, f_tail=tF, r_tail=tR, tm_after=tm2, cost=cost)
                if (best is None) or (choice.cost < best.cost):
                    best = choice
                    best_cand = cand
    return best_cand, best

# Build cost table for assignment (targets × slots)
total_targets = len(selected)
run = st.button("Auto‑design & assign 15‑plex")

results: List[PrimerPair] = []
if run:
    # Check if sequences are available
    sequences_available = any(entry and entry.get("sequence") for entry in selected)
    
    if not sequences_available:
        st.warning("⚠️ **No sequences available yet!**")
        st.info("📋 **Next steps:**")
        st.write("1. Enter your NCBI email in the sidebar")
        st.write("2. Click 'Fetch Sequences for All 15 Targets' to get sequences from NCBI")
        st.write("3. Once sequences are loaded, you can run the primer design")
        st.stop()
    
    # Filter to only targets with sequences
    target_infos = []
    missing_sequences = []
    for i, entry in enumerate(selected):
        if not entry or not entry.get("sequence"):
            missing_sequences.append(f"Target {i+1}: {entry.get('label', 'Unknown') if entry else 'No entry'}")
            continue
        target_infos.append(entry)
    
    if not target_infos:
        st.error("❌ **No targets have sequences available!**")
        st.info("💡 **Solution:** Click 'Fetch Sequences for All 15 Targets' in the sidebar to get sequences from NCBI")
        st.stop()
    
    # Limit to exactly 15 targets for multiplex design
    if len(target_infos) > 15:
        st.warning(f"⚠️ **Found {len(target_infos)} targets with sequences. Using first 15 for multiplex design.**")
        target_infos = target_infos[:15]
    elif len(target_infos) < 15:
        st.warning(f"⚠️ **Only {len(target_infos)} targets have sequences. Proceeding with available targets.**")
    
    if missing_sequences:
        st.info(f"📋 **{len(missing_sequences)} targets missing sequences (will be skipped):**")
        for missing in missing_sequences[:5]:  # Show first 5
            st.write(f"• {missing}")
        if len(missing_sequences) > 5:
            st.write(f"• ... and {len(missing_sequences) - 5} more")
    
    st.info(f"🎯 **Designing primers for {len(target_infos)} targets**")

    # Generate multiple primer candidates per target for better optimization
    st.info("🔬 **Generating multiple primer candidates per target...**")
    
    slot_list_local = slot_list.copy()
    if debug_mode:
        st.write(f"🔧 Debug: slot_list_local has {len(slot_list_local)} slots: {slot_list_local[:3]}...")
    all_candidates_per_target = []
    
    for i, entry in enumerate(target_infos):
        if entry is None:
            all_candidates_per_target.append([])
            continue
            
        # Generate many candidates for this target
        candidates = design_candidates(entry["sequence"])
        
        # Show progress for first few targets
        if i < 5:
            st.write(f"Target {i+1} ({entry['organism']}): Generated {len(candidates)} primer candidates")
            if len(candidates) == 0:
                st.warning(f"⚠️ No primer candidates generated for {entry['organism']} - sequence may be too short or invalid")
        elif i == 5:
            st.write("... (generating candidates for remaining targets)")
        
        # Create assignment choices for all candidates × all slots
        target_choices = []
        if debug_mode and i == 0:  # Debug for first target only
            st.write(f"🔧 Debug: Processing {len(candidates)} candidates for {len(slot_list_local)} slots")
        for cand_idx, cand in enumerate(candidates):
            for slot_idx, (dye, slot_idx_val, slot_tm) in enumerate(slot_list_local):
                if debug_mode and i == 0 and cand_idx == 0 and slot_idx < 5:  # Debug first candidate, first 5 slots
                    st.write(f"🔧 Debug: Candidate {cand_idx}, Slot {slot_idx} -> slot_idx_val {slot_idx_val} (dye: {dye}, tm: {slot_tm})")
                # Calculate cost for this candidate-slot combination
                core = entry["sequence"]
                if len(core) > (len(cand.fwd)+len(cand.rev)):
                    core = core[len(cand.fwd):-(len(cand.rev))]
                else:
                    core = ""
                
                # Try different tail combinations
                for tF in GC_TAILS + AT_TAILS:
                    for tR in GC_TAILS + AT_TAILS:
                        amp = f"{tF}{cand.fwd}{core}{cand.rev}{tR}"
                        tm = rough_amp_tm(amp, monovalent_mM, free_Mg_mM)
                        cost = delta_penalty*abs(tm - slot_tm) + tail_penalty*(len(tF)+len(tR))
                        
                        choice = AssignmentChoice(
                            candidate=cand, 
                            f_tail=tF, 
                            r_tail=tR, 
                            tm_after=tm, 
                            cost=cost
                        )
                        target_choices.append((choice, slot_idx))  # Use global slot index, not slot_idx_val
                        
                        # Debug: Show cost calculation for first few candidates
                        if debug_mode and i == 0 and cand_idx == 0 and slot_idx < 3:
                            st.write(f"🔧 Debug: Target {i}, Candidate {cand_idx}, Slot {slot_idx} (Tm={slot_tm:.1f}°C): amp_tm={tm:.1f}°C, cost={cost:.2f}")
        
        # Sort by cost and keep top candidates
        target_choices.sort(key=lambda x: x[0].cost)
        # Limit to top 20 candidates per target to ensure diversity across targets
        all_candidates_per_target.append(target_choices[:20])
    
    st.info(f"✅ **Generated {sum(len(candidates) for candidates in all_candidates_per_target)} total candidate-slot combinations**")

    # Ensure we have exactly 15 targets for multiplex design
    if len(all_candidates_per_target) < 15:
        # Pad with empty targets if we have fewer than 15
        while len(all_candidates_per_target) < 15:
            all_candidates_per_target.append([])
        while len(target_infos) < 15:
            target_infos.append(None)
    elif len(all_candidates_per_target) > 15:
        # Truncate if we have more than 15
        all_candidates_per_target = all_candidates_per_target[:15]
        target_infos = target_infos[:15]
    
    nT = 15  # Always work with exactly 15 targets
    nS = len(slot_list_local)
    
    if nS != 15:
        st.error("Internal error: expected 15 slots.")
        st.stop()

    # Greedy assignment with multiple candidates per target
    st.info("🎯 **Optimizing target-slot assignments...**")
    
    assigned_slots = [-1]*nT
    taken = [False]*nS
    flat = []
    
    # Create flat list of all candidate-slot combinations
    for i in range(nT):
        for choice, slot_idx in all_candidates_per_target[i]:
            flat.append((choice.cost, i, slot_idx, choice))
    
    if not flat:
        st.error("No viable primer/slot combinations were found. Try widening product length or primer Tm range.")
        st.stop()
    
    # Cross-dimer checker function
    def has_bad_3prime_dimer(a: str, b: str) -> bool:
        a3 = a[-5:]
        b3 = b[-5:]
        comp = str.maketrans("ACGT", "TGCA")
        return a3 == b3.translate(comp)[::-1]
    
    # ROUND-ROBIN TARGET BALANCING: Ensure diversity across targets
    # Sort candidates within each target first, then use round-robin selection
    target_candidates = {}
    for cost, target_idx, slot_idx, choice in flat:
        if target_idx not in target_candidates:
            target_candidates[target_idx] = []
        target_candidates[target_idx].append((cost, target_idx, slot_idx, choice))
    
    # Sort candidates within each target by cost
    for target_idx in target_candidates:
        target_candidates[target_idx].sort(key=lambda x: x[0])
    
    # Round-robin selection: take best candidate from each target in turn
    balanced_flat = []
    max_rounds = max(len(candidates) for candidates in target_candidates.values()) if target_candidates else 0
    
    for round_num in range(max_rounds):
        for target_idx in sorted(target_candidates.keys()):
            if round_num < len(target_candidates[target_idx]):
                balanced_flat.append(target_candidates[target_idx][round_num])
    
    # Use the balanced flat list instead of the original
    flat = balanced_flat
    st.write(f"🔧 Debug: Round-robin target balancing applied - {len(flat)} candidates selected from {len(flat)} total, {len(target_candidates)} targets")
    
    # Debug first few entries after sorting
    if debug_mode:
        for i in range(min(5, len(flat))):
            cost, target_idx, slot_idx, choice = flat[i]
            st.write(f"🔧 Debug: Flat entry {i+1}: target {target_idx}, slot {slot_idx}, cost {cost:.2f}")
    
    # Always show first 10 entries to understand the distribution
    st.write("🔧 Debug: First 10 entries in sorted flat list:")
    for i in range(min(10, len(flat))):
        cost, target_idx, slot_idx, choice = flat[i]
        st.write(f"  Entry {i+1}: target {target_idx}, slot {slot_idx}, cost {cost:.2f}")
    
    # Show slot distribution in first 20 entries
    slot_counts = {}
    for cost, target_idx, slot_idx, choice in flat[:20]:
        slot_counts[slot_idx] = slot_counts.get(slot_idx, 0) + 1
    st.write(f"🔧 Debug: Slot distribution in first 20 entries: {dict(sorted(slot_counts.items()))}")
    
    # Show target distribution in first 20 entries
    target_counts = {}
    for cost, target_idx, slot_idx, choice in flat[:20]:
        target_counts[target_idx] = target_counts.get(target_idx, 0) + 1
    st.write(f"🔧 Debug: Target distribution in first 20 entries: {dict(sorted(target_counts.items()))}")
    
    # Show temperature analysis for first 20 entries
    st.write("🔧 Debug: Temperature analysis for first 20 entries:")
    for i, (cost, target_idx, slot_idx, choice) in enumerate(flat[:20]):
        slot_tm = slot_list[slot_idx][2]  # Get slot temperature
        st.write(f"  Entry {i+1}: target {target_idx}, slot {slot_idx} (Tm={slot_tm:.1f}°C), amp_tm={choice.tm_after:.1f}°C, cost={cost:.2f}")
    
    # Debug: show some information about the flat list
    if debug_mode:
        st.write(f"🔧 Debug: Flat list has {len(flat)} combinations")
        if len(flat) > 0:
            st.write(f"🔧 Debug: Best cost: {flat[0][0]:.2f}, Worst cost: {flat[-1][0]:.2f}")
            st.write(f"🔧 Debug: First few combinations: {flat[:3]}")
            
            # Show slot distribution in first 20 entries
            slot_counts = {}
            target_counts = {}
            for cost, target_idx, slot_idx, choice in flat[:20]:
                slot_counts[slot_idx] = slot_counts.get(slot_idx, 0) + 1
                target_counts[target_idx] = target_counts.get(target_idx, 0) + 1
            st.write(f"🔧 Debug: Slot distribution in first 20 entries: {dict(sorted(slot_counts.items()))}")
            st.write(f"🔧 Debug: Target distribution in first 20 entries: {dict(sorted(target_counts.items()))}")
            
            # Show target diversity in first 50 entries
            unique_targets = set()
            for cost, target_idx, slot_idx, choice in flat[:50]:
                unique_targets.add(target_idx)
            st.write(f"🔧 Debug: Unique targets in first 50 entries: {sorted(unique_targets)} (total: {len(unique_targets)})")
        
        # Debug: check target_infos structure
        valid_targets = sum(1 for t in target_infos if t is not None)
        st.write(f"🔧 Debug: {valid_targets} valid targets out of {len(target_infos)} total targets")
        for i, target in enumerate(target_infos[:5]):
            if target is not None:
                st.write(f"🔧 Debug: Target {i}: {target['organism']} - {target.get('target', 'Unknown')}")
            else:
                st.write(f"🔧 Debug: Target {i}: None")
    
    # Greedy assignment with optional cross-dimer consideration
    assigned_choices = {}  # target_idx -> choice
    conflicts_avoided = 0
    assignments_made = 0
    debug_count = 0  # Limit debug output
    max_iterations = min(len(flat), 1000)  # Safety limit to prevent infinite loops
    iteration_count = 0
    
    # Early termination: if we have fewer valid targets than slots, we can stop early
    valid_targets = sum(1 for t in target_infos if t is not None)
    max_possible_assignments = min(valid_targets, nS)
    
    # Always show some basic info about the assignment process
    st.write(f"🔧 Debug: Starting assignment with {valid_targets} valid targets, {nS} slots, {len(flat)} total combinations")
    
    for cost, target_idx, slot_idx, choice in flat:
        iteration_count += 1
        if iteration_count > max_iterations:
            st.warning(f"⚠️ **Reached maximum iteration limit ({max_iterations}). Stopping assignment.**")
            break
            
        # Early termination if we've assigned all possible targets
        if assignments_made >= max_possible_assignments:
            st.write(f"🔧 Debug: Early termination - assigned {assignments_made} targets, max possible: {max_possible_assignments}")
            break
            
        # Debug: show why assignments are being skipped (limit output)
        if debug_mode and debug_count < 10:  # Only show first 10 attempts
            target_already_assigned = assigned_slots[target_idx] != -1
            slot_already_taken = taken[slot_idx]
            st.write(f"🔧 Debug: Trying target {target_idx}, slot {slot_idx} - Target assigned: {target_already_assigned}, Slot taken: {slot_already_taken}")
            debug_count += 1
        
        # Skip if target is already assigned or slot is taken
        if assigned_slots[target_idx] != -1 or taken[slot_idx]:
            # Debug: show why we're skipping (first 20 skips)
            if iteration_count <= 20:
                reason = []
                if assigned_slots[target_idx] != -1:
                    reason.append(f"target {target_idx} already assigned to slot {assigned_slots[target_idx]}")
                if taken[slot_idx]:
                    reason.append(f"slot {slot_idx} already taken")
                st.write(f"🔧 Debug: Skip iteration {iteration_count} - {', '.join(reason)}")
            continue
            
        # Skip if target doesn't exist (None)
        if target_idx >= len(target_infos) or target_infos[target_idx] is None:
            if iteration_count <= 20:
                st.write(f"🔧 Debug: Skip iteration {iteration_count} - target {target_idx} doesn't exist")
            continue
            
        # Debug: show when we find a viable assignment
        if assignments_made < 5:  # Show first 5 assignments
            st.write(f"🔧 Debug: Found viable assignment - target {target_idx}, slot {slot_idx}, cost {cost:.2f}")
        
        # Check for cross-dimer conflicts with already assigned primers
        has_conflict = False
        for assigned_target, assigned_choice in assigned_choices.items():
            if has_bad_3prime_dimer(choice.f_tail + choice.candidate.fwd, 
                                  assigned_choice.f_tail + assigned_choice.candidate.fwd) or \
               has_bad_3prime_dimer(choice.candidate.rev + choice.r_tail, 
                                  assigned_choice.candidate.rev + assigned_choice.r_tail):
                has_conflict = True
                conflicts_avoided += 1
                # Debug: show cross-dimer conflicts (first 10)
                if conflicts_avoided <= 10:
                    st.write(f"🔧 Debug: Cross-dimer conflict - target {target_idx} conflicts with assigned target {assigned_target}")
                break
        
        if not has_conflict:
            assigned_slots[target_idx] = slot_idx
            taken[slot_idx] = True
            assigned_choices[target_idx] = choice
            assignments_made += 1
            # Store the choice for later use
            if not hasattr(choice, '_assigned_target'):
                choice._assigned_target = target_idx
                choice._assigned_slot = slot_idx
            
            # Debug: show first few assignments
            if debug_mode and assignments_made <= 3:
                st.write(f"🔧 Debug: Assigned target {target_idx} to slot {slot_idx} (cost: {cost:.2f})")
                
            # Check if all targets are assigned
            if all(a != -1 for a in assigned_slots):
                break
    
    # If we have very few assignments due to conflicts, try a more lenient approach
    if assignments_made < 5 and len(flat) > 0:
        st.warning(f"⚠️ **Cross-dimer conflicts prevented {conflicts_avoided} assignments. Trying more lenient assignment...**")
        
        # Reset and try without cross-dimer checking
        assigned_slots = [-1]*nT
        taken = [False]*nS
        assigned_choices = {}
        assignments_made = 0
        iteration_count = 0
        
        for cost, target_idx, slot_idx, choice in flat:
            iteration_count += 1
            if iteration_count > max_iterations:
                st.warning(f"⚠️ **Reached maximum iteration limit ({max_iterations}) in lenient assignment. Stopping.**")
                break
                
            # Early termination if we've assigned all possible targets
            if assignments_made >= max_possible_assignments:
                break
                
            # Skip if target is already assigned or slot is taken
            if assigned_slots[target_idx] != -1 or taken[slot_idx]:
                continue
                
            # Skip if target doesn't exist (None)
            if target_idx >= len(target_infos) or target_infos[target_idx] is None:
                continue
                
            assigned_slots[target_idx] = slot_idx
            taken[slot_idx] = True
            assigned_choices[target_idx] = choice
            assignments_made += 1
            # Store the choice for later use
            if not hasattr(choice, '_assigned_target'):
                choice._assigned_target = target_idx
                choice._assigned_slot = slot_idx
            
            # Debug: show first few assignments
            if debug_mode and assignments_made <= 3:
                st.write(f"🔧 Debug: Assigned target {target_idx} to slot {slot_idx} (cost: {cost:.2f})")
                
            # Check if all targets are assigned
            if all(a != -1 for a in assigned_slots):
                break

    # Count successful assignments
    successful_assignments = sum(1 for a in assigned_slots if a != -1)
    
    # Debug: show assignment status
    st.write(f"🔧 Debug: Assignment complete - {successful_assignments} targets assigned, {assignments_made} assignments made, {conflicts_avoided} conflicts avoided")
    st.write(f"🔧 Debug: Assigned slots: {[i for i, slot in enumerate(assigned_slots) if slot != -1]}")
    st.write(f"🔧 Debug: Taken slots: {[i for i, taken_flag in enumerate(taken) if taken_flag]}")
    
    if successful_assignments == 0:
        st.error("Could not assign any targets to slots with current constraints.")
        st.stop()
    elif successful_assignments < 15:
        st.warning(f"⚠️ **Successfully assigned {successful_assignments} out of 15 targets.**")
        st.info("💡 **Tip:** Try adjusting primer design parameters or fetch more sequences to improve assignment.")
    
    # Show optimization statistics
    if successful_assignments > 0:
        total_candidates = sum(len(candidates) for candidates in all_candidates_per_target)
        st.info(f"📊 **Optimization Stats:** Evaluated {total_candidates} candidate-slot combinations, selected {successful_assignments} optimal assignments")
        
        # Show assignment details
        if 'conflicts_avoided' in locals():
            st.info(f"🔍 **Assignment Details:** Avoided {conflicts_avoided} cross-dimer conflicts, made {assignments_made} assignments")
        
        # Show which targets were successfully assigned
        assigned_targets = []
        for i, slot in enumerate(assigned_slots):
            if slot != -1 and i < len(target_infos) and target_infos[i] is not None:
                assigned_targets.append(f"Target {i+1}: {target_infos[i]['organism']}")
        
        if assigned_targets:
            with st.expander("✅ Successfully Assigned Targets", expanded=False):
                for target in assigned_targets:
                    st.write(f"• {target}")

    assigned_pairs: List[PrimerPair] = []
    for i, entry in enumerate(target_infos):
        if entry is None:  # Skip empty targets
            continue
        j = assigned_slots[i]
        if j == -1:  # Skip unassigned targets
            continue
        
        # Get the assigned choice for this target
        assigned_choice = assigned_choices.get(i)
        if assigned_choice is None:  # Skip targets without viable primers
            continue
            
        dye, slot_idx, slot_tm = slot_list_local[j]
        cand = assigned_choice.candidate
        core = entry["sequence"]
        if len(core) > (len(cand.fwd)+len(cand.rev)):
            core_frag = core[len(cand.fwd):-(len(cand.rev))]
        else:
            core_frag = ""
        amp_len = len(assigned_choice.f_tail) + len(cand.fwd) + len(core_frag) + len(cand.rev) + len(assigned_choice.r_tail)
        issues: List[str] = []
        # quick within-set cross-dimer marking added later
        assigned_pairs.append(
            PrimerPair(
                fwd=assigned_choice.f_tail + cand.fwd,
                rev=cand.rev + assigned_choice.r_tail,
                product_len=amp_len,
                product_tm=assigned_choice.tm_after,
                channel=dye,
                slot_tm=slot_tm,
                slot_index=slot_idx,
                organism=entry["organism"],
                target_name=entry["target"],
                tails=(assigned_choice.f_tail, assigned_choice.r_tail),
                issues=issues,
            )
        )

    # cross-dimer warnings
    for a in range(len(assigned_pairs)):
        for b in range(a+1, len(assigned_pairs)):
            if has_bad_3prime_dimer(assigned_pairs[a].fwd, assigned_pairs[b].fwd) or \
               has_bad_3prime_dimer(assigned_pairs[a].rev, assigned_pairs[b].rev):
                assigned_pairs[a].issues.append(f"Potential 3' cross-dimer with {assigned_pairs[b].organism} ({assigned_pairs[b].channel} s{assigned_pairs[b].slot_index+1})")
                assigned_pairs[b].issues.append(f"Potential 3' cross-dimer with {assigned_pairs[a].organism} ({assigned_pairs[a].channel} s{assigned_pairs[a].slot_index+1})")

    df = pd.DataFrame([
        {
            "Channel": r.channel,
            "Slot": r.slot_index+1,
            "Slot_Tm_°C": round(r.slot_tm,1),
            "Pred_Product_Tm_°C": round(r.product_tm,2),
            "ΔTm_to_Slot": round(r.product_tm - r.slot_tm, 2),
            "Organism": r.organism,
            "Target": r.target_name,
            "Forward_Primer": r.fwd,
            "Reverse_Primer": r.rev,
            "Amplicon_bp": r.product_len,
            "Tails": "/".join([r.tails[0] or "-", r.tails[1] or "-"]),
            "Issues": "; ".join(sorted(set(r.issues))) if r.issues else ""
        } for r in assigned_pairs
    ]).sort_values(["Channel","Slot"]).reset_index(drop=True)

    st.success(f"✅ **Auto-assignment complete!** Successfully designed primers for {len(assigned_pairs)} targets.")
    st.dataframe(df, use_container_width=True)

    out = io.BytesIO()
    df.to_csv(out, index=False)
    st.download_button("Download manifest (CSV)", data=out.getvalue(), file_name="autoprimer5_multiplex_manifest.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "This version loads organisms/targets from autoprimer5.py, extracts embedded sequences with user-provided key hints, "
    "and auto-assigns 15 targets to 3×5 Tm slots. No NCBI/network calls required."
)
