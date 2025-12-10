import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Tuple, Optional
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as MT
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import requests
import json
import time
import random
from dataclasses import dataclass
from Bio.SeqUtils.MeltingTemp import Tm_NN
from Bio.Align import PairwiseAligner
import plotly.express as px
import primer3
import re
from Bio import Entrez, SeqIO
import io
import pickle
import gzip
from datetime import datetime, timedelta
from functools import wraps
from urllib.error import HTTPError
from Bio.Blast import NCBIWWW, NCBIXML
from pathlib import Path
import base64
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def get_related_organisms(target_organism, max_organisms=100):
    """Get comprehensive list of related organisms for specificity testing (100+ organisms)"""
    organism_lower = target_organism.lower()
    
    # Comprehensive organism database with 100+ organisms across all major taxonomic groups
    comprehensive_organisms = {
        # Fungi (40+ organisms)
        'fungi': [
            'Fusarium oxysporum', 'Fusarium solani', 'Fusarium graminearum', 'Fusarium verticillioides',
            'Aspergillus niger', 'Aspergillus flavus', 'Aspergillus fumigatus', 'Aspergillus terreus',
            'Penicillium chrysogenum', 'Penicillium expansum', 'Penicillium digitatum', 'Penicillium italicum',
            'Botrytis cinerea', 'Sclerotinia sclerotiorum', 'Alternaria alternata', 'Alternaria solani',
            'Rhizoctonia solani', 'Pythium ultimum', 'Phytophthora infestans', 'Phytophthora capsici',
            'Trichoderma harzianum', 'Trichoderma viride', 'Trichoderma reesei', 'Trichoderma atroviride',
            'Erysiphe necator', 'Puccinia graminis', 'Ustilago maydis', 'Magnaporthe oryzae',
            'Colletotrichum gloeosporioides', 'Verticillium dahliae', 'Verticillium albo-atrum',
            'Cercospora beticola', 'Septoria tritici', 'Mycosphaerella graminicola', 'Cladosporium fulvum',
            'Monilinia fructicola', 'Monilinia laxa', 'Monilinia fructigena', 'Sclerotium rolfsii',
            'Armillaria mellea', 'Ganoderma lucidum', 'Pleurotus ostreatus', 'Agaricus bisporus'
        ],
        
        # Bacteria (30+ organisms)
        'bacteria': [
            'Pseudomonas syringae', 'Pseudomonas aeruginosa', 'Pseudomonas fluorescens', 'Pseudomonas putida',
            'Xanthomonas campestris', 'Xanthomonas oryzae', 'Xanthomonas axonopodis', 'Xanthomonas citri',
            'Erwinia amylovora', 'Erwinia carotovora', 'Erwinia chrysanthemi', 'Erwinia pyrifoliae',
            'Ralstonia solanacearum', 'Agrobacterium tumefaciens', 'Rhizobium leguminosarum', 'Sinorhizobium meliloti',
            'Bacillus subtilis', 'Bacillus thuringiensis', 'Bacillus cereus', 'Bacillus anthracis',
            'Staphylococcus aureus', 'Streptococcus pneumoniae', 'Escherichia coli', 'Salmonella enterica',
            'Listeria monocytogenes', 'Clostridium botulinum', 'Mycobacterium tuberculosis', 'Corynebacterium diphtheriae',
            'Streptomyces coelicolor', 'Actinomyces israelii', 'Propionibacterium acnes', 'Lactobacillus acidophilus'
        ],
        
        # Viruses (25+ organisms)
        'viruses': [
            'Tobacco mosaic virus', 'Tomato mosaic virus', 'Cucumber mosaic virus', 'Potato virus X',
            'Potato virus Y', 'Beet curly top virus', 'Arabis mosaic virus', 'Alfalfa mosaic virus',
            'Lettuce chlorosis virus', 'Cannabis cryptic virus', 'Hop latent viroid', 'Potato spindle tuber viroid',
            'Influenza A virus', 'Coronavirus', 'Rhinovirus', 'Adenovirus', 'Respiratory syncytial virus',
            'MERS-CoV', 'Human coronavirus 229E', 'Human coronavirus OC43', 'Parainfluenza virus',
            'Measles virus', 'Mumps virus', 'Rubella virus', 'Varicella-zoster virus', 'Herpes simplex virus'
        ],
        
        # Arthropods (25+ organisms)
        'arthropods': [
            'Tetranychus urticae', 'Panonychus ulmi', 'Oligonychus ilicis', 'Aculops lycopersici',
            'Bemisia tabaci', 'Trialeurodes vaporariorum', 'Aphis gossypii', 'Myzus persicae',
            'Diaphorina citri', 'Frankliniella occidentalis', 'Thrips tabaci', 'Thrips palmi',
            'Scirtothrips dorsalis', 'Polyphagotarsonemus latus', 'Tarsonemus pallidus',
            'Drosophila melanogaster', 'Musca domestica', 'Lucilia sericata', 'Calliphora vicina',
            'Tribolium castaneum', 'Sitophilus oryzae', 'Plodia interpunctella', 'Ephestia kuehniella',
            'Helicoverpa armigera', 'Spodoptera frugiperda', 'Plutella xylostella', 'Pieris rapae'
        ],
        
        # Plants (20+ organisms)
        'plants': [
            'Arabidopsis thaliana', 'Oryza sativa', 'Zea mays', 'Triticum aestivum',
            'Hordeum vulgare', 'Solanum lycopersicum', 'Solanum tuberosum', 'Capsicum annuum',
            'Nicotiana tabacum', 'Gossypium hirsutum', 'Glycine max', 'Phaseolus vulgaris',
            'Vitis vinifera', 'Malus domestica', 'Prunus persica', 'Citrus sinensis',
            'Brassica oleracea', 'Daucus carota', 'Lactuca sativa', 'Spinacia oleracea',
            'Helianthus annuus', 'Beta vulgaris', 'Saccharum officinarum', 'Manihot esculenta'
        ],
        
        # Nematodes (10+ organisms)
        'nematodes': [
            'Caenorhabditis elegans', 'Meloidogyne incognita', 'Meloidogyne javanica', 'Meloidogyne hapla',
            'Heterodera glycines', 'Globodera rostochiensis', 'Pratylenchus penetrans', 'Radopholus similis',
            'Ditylenchus dipsaci', 'Aphelenchoides fragariae', 'Bursaphelenchus xylophilus', 'Steinernema carpocapsae'
        ]
    }
    
    # Try to find specific matches first
    specific_matches = []
    for category, organisms in comprehensive_organisms.items():
        for organism in organisms:
            organism_lower_check = organism.lower()
            # Check for genus or species matches
            if any(part in organism_lower_check for part in organism_lower.split()):
                specific_matches.append(organism)
    
    # If we found specific matches, return them plus related organisms
    if specific_matches:
        result = specific_matches[:]
        # Add organisms from the same category
        for category, organisms in comprehensive_organisms.items():
            if any(match.lower() in [org.lower() for org in organisms] for match in specific_matches):
                result.extend([org for org in organisms if org not in result])
                break
        return result[:max_organisms]
    
    # If no specific matches, return a diverse set from all categories
    diverse_set = []
    for category, organisms in comprehensive_organisms.items():
        diverse_set.extend(organisms[:max_organisms//len(comprehensive_organisms)])
    
    # Add more organisms to reach the target number
    remaining_needed = max_organisms - len(diverse_set)
    if remaining_needed > 0:
        all_organisms = []
        for organisms in comprehensive_organisms.values():
            all_organisms.extend(organisms)
        
        # Add random selection from remaining organisms
        remaining_organisms = [org for org in all_organisms if org not in diverse_set]
        if remaining_organisms:
            diverse_set.extend(random.sample(remaining_organisms, min(remaining_needed, len(remaining_organisms))))
    
    return diverse_set[:max_organisms]

def check_session_state_validity():
    """Check if session state has valid data"""
    # Ensure all required keys exist with safe defaults
    primers = OptimizedSessionManager.get_primers_optimized()
    sequence = st.session_state.get('current_sequence', '')
    seq_info = st.session_state.get('sequence_info', {})
    
    has_primers = bool(primers)
    has_sequence = bool(sequence)
    has_seq_info = bool(seq_info)
    
    return {
        'has_primers': has_primers,
        'has_sequence': has_sequence,
        'has_seq_info': has_seq_info,
        'primer_count': len(primers),
        'sequence_length': len(sequence)
    }

def get_gene_priority(category):
    """Get priority level for gene category"""
    category_lower = category.lower()
    
    # High priority categories
    if any(keyword in category_lower for keyword in ['essential', 'housekeeping', 'actin', 'tubulin', 'rna polymerase', 'ribosomal']):
        return "High"
    
    # Medium priority categories
    elif any(keyword in category_lower for keyword in ['pathogenicity', 'virulence', 'effector', 'resistance', 'detoxification', 'rnai pathway']):
        return "Medium"
    
    # Low priority categories
    else:
        return "Low"

def get_gene_use_recommendation(category):
    """Get usage recommendation for gene category"""
    category_lower = category.lower()
    
    if 'essential' in category_lower or 'housekeeping' in category_lower:
        return "Recommended for reliable detection and quantification. These genes are highly conserved and expressed constitutively."
    
    elif 'pathogenicity' in category_lower or 'virulence' in category_lower:
        return "Useful for detecting pathogenic strains and studying disease mechanisms. May not be present in all isolates."
    
    elif 'rnai pathway' in category_lower:
        return "Critical for gene silencing studies and RNAi-based control strategies. Essential for understanding RNA interference machinery and developing dsRNA-based treatments."
    
    elif 'resistance' in category_lower:
        return "Important for monitoring resistance development and screening for resistant strains. Critical for management decisions."
    
    elif 'detoxification' in category_lower:
        return "Useful for studying pesticide resistance and detoxification mechanisms. Important for resistance management."
    
    elif 'secondary metabolite' in category_lower:
        return "Useful for detecting toxin-producing strains and studying secondary metabolism. May be strain-specific."
    
    elif 'cell wall' in category_lower:
        return "Good for structural studies and cell wall targeting. Generally well-conserved across strains."
    
    elif 'development' in category_lower:
        return "Useful for studying life cycle stages and development processes. May be stage-specific."
    
    else:
        return "General purpose gene category. Consider specific research objectives when selecting targets."

def get_species_prefix(scientific_name):
    """Generate species prefix for gene names (e.g., 'Botrytis cinerea' -> 'Bc')"""
    parts = scientific_name.split()
    if len(parts) >= 2:
        return parts[0][0].upper() + parts[1][0].lower()
    elif len(parts) == 1:
        return parts[0][:2].capitalize()
    return "Sp"  # Generic prefix

def generate_rnai_genes(species_prefix, organism_type="fungus"):
    """Generate RNAi pathway genes based on organism type"""
    
    if organism_type == "fungus":
        return [
            f"{species_prefix}DCL1 (dicer-like 1)",
            f"{species_prefix}DCL2 (dicer-like 2)",
            f"{species_prefix}AGO1 (argonaute 1)",
            f"{species_prefix}AGO2 (argonaute 2)",
            f"{species_prefix}RDRP1 (RNA-dependent RNA polymerase 1)",
            f"{species_prefix}RDRP2 (RNA-dependent RNA polymerase 2)",
            f"{species_prefix}QDE2 (RecQ helicase)",
            f"{species_prefix}SAD1 (RNAi deficient 1)",
            f"{species_prefix}SMS2 (suppressor of meiotic silencing)",
            f"{species_prefix}RRP44 (exoribonuclease)"
        ]
    
    elif organism_type == "virus":
        return [
            f"{species_prefix}VSR1 (viral suppressor of RNAi 1)",
            f"{species_prefix}VSR2 (viral suppressor of RNAi 2)",
            f"{species_prefix}P19 (silencing suppressor)",
            f"{species_prefix}2b (silencing suppressor)",
            f"{species_prefix}HC-Pro (helper component protease)",
            f"{species_prefix}P1 (silencing suppressor)"
        ]
    
    elif organism_type == "insect":
        return [
            f"{species_prefix}DCR1 (dicer 1)",
            f"{species_prefix}DCR2 (dicer 2)",
            f"{species_prefix}AGO1 (argonaute 1)",
            f"{species_prefix}AGO2 (argonaute 2)",
            f"{species_prefix}AGO3 (argonaute 3)",
            f"{species_prefix}PIWI (piwi protein)",
            f"{species_prefix}AUB (aubergine)",
            f"{species_prefix}R2D2 (dsRNA binding protein)",
            f"{species_prefix}LOQUACIOUS (dsRNA binding protein)",
            f"{species_prefix}DROSHA (RNase III enzyme)"
        ]
    
    elif organism_type == "plant":
        return [
            f"{species_prefix}DCL1 (dicer-like 1)",
            f"{species_prefix}DCL2 (dicer-like 2)",
            f"{species_prefix}DCL3 (dicer-like 3)",
            f"{species_prefix}DCL4 (dicer-like 4)",
            f"{species_prefix}AGO1 (argonaute 1)",
            f"{species_prefix}AGO4 (argonaute 4)",
            f"{species_prefix}AGO7 (argonaute 7)",
            f"{species_prefix}RDR1 (RNA-dependent RNA polymerase 1)",
            f"{species_prefix}RDR2 (RNA-dependent RNA polymerase 2)",
            f"{species_prefix}RDR6 (RNA-dependent RNA polymerase 6)"
        ]
    
    elif organism_type == "bacteria":
        return [
            f"{species_prefix}CAS1 (CRISPR-associated protein 1)",
            f"{species_prefix}CAS2 (CRISPR-associated protein 2)",
            f"{species_prefix}CAS3 (CRISPR-associated protein 3)",
            f"{species_prefix}CAS9 (CRISPR-associated protein 9)",
            f"{species_prefix}CSPA (cold shock protein A)",
            f"{species_prefix}RNaseE (ribonuclease E)",
            f"{species_prefix}RNaseIII (ribonuclease III)",
            f"{species_prefix}sRNA1 (small regulatory RNA 1)"
        ]
    
    elif organism_type == "oomycete":
        return [
            f"{species_prefix}DCL1 (dicer-like 1)",
            f"{species_prefix}DCL2 (dicer-like 2)",
            f"{species_prefix}AGO1 (argonaute 1)",
            f"{species_prefix}RRP6 (exoribonuclease)",
            f"{species_prefix}RRP44 (exoribonuclease)",
            f"{species_prefix}RDRP1 (RNA-dependent RNA polymerase)"
        ]
    
    elif organism_type == "nematode":
        return [
            f"{species_prefix}DCR1 (dicer 1)",
            f"{species_prefix}RDE1 (RNAi deficient 1)",
            f"{species_prefix}RDE4 (RNAi deficient 4)",
            f"{species_prefix}RRF1 (RNA-directed RNA polymerase 1)",
            f"{species_prefix}RRF3 (RNA-directed RNA polymerase 3)",
            f"{species_prefix}ALG1 (argonaute-like gene 1)",
            f"{species_prefix}ALG2 (argonaute-like gene 2)",
            f"{species_prefix}EGO1 (enhancer of glp-1)",
            f"{species_prefix}MUT7 (mutator 7)",
            f"{species_prefix}SID1 (systemic RNAi deficient 1)"
        ]
    
    else:
        # Generic RNAi genes for unknown organism types
        return [
            f"{species_prefix}DCL1 (dicer-like 1)",
            f"{species_prefix}AGO1 (argonaute 1)",
            f"{species_prefix}RDRP1 (RNA-dependent RNA polymerase)"
        ]

def determine_organism_type(category, subcategory, scientific_name):
    """Determine organism type based on category and taxonomy"""
    
    category_lower = category.lower()
    subcategory_lower = subcategory.lower()
    name_lower = scientific_name.lower()
    
    if "fungi" in category_lower or "fungal" in category_lower:
        return "fungus"
    elif "virus" in category_lower or "viral" in category_lower:
        return "virus"
    elif "insect" in category_lower or "arthropod" in category_lower or "mite" in subcategory_lower or "thrips" in subcategory_lower:
        return "insect"
    elif "bacteria" in category_lower or "bacterial" in category_lower:
        return "bacteria"
    elif "plant" in category_lower:
        return "plant"
    elif "oomycete" in category_lower or "pythium" in name_lower or "phytophthora" in name_lower:
        return "oomycete"
    elif "nematode" in category_lower or "caenorhabditis" in name_lower or "meloidogyne" in name_lower:
        return "nematode"
    else:
        return "unknown"

def enhanced_gene_search_variations(clean_gene):
    """Generate search variations including comprehensive RNAi terms"""
    gene_variations = [clean_gene]
    
    # Existing gene variations
    if clean_gene.upper() in ['ACT1', 'ACTIN']:
        gene_variations.extend(['actin', 'ACT', 'Actin'])
    elif clean_gene.upper() in ['TUB1', 'TUBULIN']:
        gene_variations.extend(['tubulin', 'TUB', 'Tubulin'])
    elif clean_gene.upper() in ['EF1A', 'ELONGATION']:
        gene_variations.extend(['elongation factor', 'EF1', 'EF-1'])
    elif 'RPL' in clean_gene.upper():
        gene_variations.extend([clean_gene.replace('RPL', 'ribosomal protein L'), 'ribosomal protein'])
    elif 'RPS' in clean_gene.upper():
        gene_variations.extend([clean_gene.replace('RPS', 'ribosomal protein S'), 'ribosomal protein'])
    
    # RNAi gene variations - Dicer genes
    elif any(dcl in clean_gene.upper() for dcl in ['DCL1', 'DCR1']):
        gene_variations.extend([
            'dicer-like 1', 'dicer like 1', 'DCL1', 'DCR1', 'dicer1', 'dicer 1',
            'RNase III', 'ribonuclease III', 'endoribonuclease'
        ])
    elif any(dcl in clean_gene.upper() for dcl in ['DCL2', 'DCR2']):
        gene_variations.extend([
            'dicer-like 2', 'dicer like 2', 'DCL2', 'DCR2', 'dicer2', 'dicer 2',
            'RNase III', 'ribonuclease III'
        ])
    elif any(dcl in clean_gene.upper() for dcl in ['DCL3']):
        gene_variations.extend(['dicer-like 3', 'dicer like 3', 'DCL3', 'dicer3'])
    elif any(dcl in clean_gene.upper() for dcl in ['DCL4']):
        gene_variations.extend(['dicer-like 4', 'dicer like 4', 'DCL4', 'dicer4'])
    
    # Argonaute genes
    elif 'AGO1' in clean_gene.upper():
        gene_variations.extend([
            'argonaute 1', 'argonaute1', 'AGO1', 'RISC', 'slicer',
            'RNA-induced silencing complex'
        ])
    elif 'AGO2' in clean_gene.upper():
        gene_variations.extend(['argonaute 2', 'argonaute2', 'AGO2', 'RISC'])
    elif 'AGO3' in clean_gene.upper():
        gene_variations.extend(['argonaute 3', 'argonaute3', 'AGO3'])
    elif 'AGO4' in clean_gene.upper():
        gene_variations.extend(['argonaute 4', 'argonaute4', 'AGO4'])
    elif 'AGO7' in clean_gene.upper():
        gene_variations.extend(['argonaute 7', 'argonaute7', 'AGO7'])
    
    # RNA-dependent RNA polymerase genes
    elif any(rdrp in clean_gene.upper() for rdrp in ['RDRP1', 'RDR1', 'RRF1']):
        gene_variations.extend([
            'RNA-dependent RNA polymerase 1', 'RdRP1', 'RDR1', 'RRF1',
            'RNA polymerase', 'RNA-directed RNA polymerase'
        ])
    elif any(rdrp in clean_gene.upper() for rdrp in ['RDRP2', 'RDR2', 'RRF2']):
        gene_variations.extend([
            'RNA-dependent RNA polymerase 2', 'RdRP2', 'RDR2', 'RRF2'
        ])
    elif any(rdrp in clean_gene.upper() for rdrp in ['RDRP3', 'RDR6', 'RRF3']):
        gene_variations.extend([
            'RNA-dependent RNA polymerase 6', 'RdRP6', 'RDR6', 'RRF3'
        ])
    
    # Helicase genes
    elif 'QDE2' in clean_gene.upper():
        gene_variations.extend([
            'RecQ helicase', 'QDE2', 'helicase', 'DNA helicase',
            'ATP-dependent helicase'
        ])
    
    # PIWI genes
    elif 'PIWI' in clean_gene.upper():
        gene_variations.extend(['piwi protein', 'PIWI', 'P-element induced wimpy'])
    elif 'AUB' in clean_gene.upper():
        gene_variations.extend(['aubergine', 'AUB', 'piwi protein'])
    
    # Other RNAi components
    elif 'R2D2' in clean_gene.upper():
        gene_variations.extend(['R2D2', 'dsRNA binding protein', 'double-stranded RNA binding'])
    elif 'LOQUACIOUS' in clean_gene.upper():
        gene_variations.extend(['loquacious', 'LOQS', 'dsRNA binding protein'])
    elif 'DROSHA' in clean_gene.upper():
        gene_variations.extend(['drosha', 'RNase III enzyme', 'microprocessor'])
    
    # RNAi deficient genes
    elif any(rde in clean_gene.upper() for rde in ['RDE1', 'RDE4']):
        gene_variations.extend([
            'RNAi deficient', 'RDE1', 'RDE4', 'RNA interference deficient'
        ])
    elif 'SAD1' in clean_gene.upper():
        gene_variations.extend(['RNAi deficient 1', 'SAD1', 'suppressor'])
    elif 'SMS2' in clean_gene.upper():
        gene_variations.extend(['suppressor of meiotic silencing', 'SMS2'])
    
    # Viral suppressors
    elif 'VSR' in clean_gene.upper():
        gene_variations.extend([
            'viral suppressor of RNAi', 'VSR', 'silencing suppressor',
            'RNA silencing suppressor'
        ])
    elif 'P19' in clean_gene.upper():
        gene_variations.extend(['P19', 'silencing suppressor', 'p19 protein'])
    elif 'HC-PRO' in clean_gene.upper():
        gene_variations.extend([
            'helper component protease', 'HC-Pro', 'helper component proteinase',
            'silencing suppressor'
        ])
    
    # CRISPR genes (for bacteria)
    elif 'CAS' in clean_gene.upper():
        gene_variations.extend([
            'CRISPR-associated protein', 'CAS', 'CRISPR protein',
            'clustered regularly interspaced short palindromic repeats'
        ])
    
    # Exoribonuclease genes
    elif any(rrp in clean_gene.upper() for rrp in ['RRP44', 'RRP6']):
        gene_variations.extend([
            'exoribonuclease', 'RRP44', 'RRP6', '3 to 5 exoribonuclease',
            'RNA processing'
        ])
    
    # Systemic RNAi genes
    elif 'SID1' in clean_gene.upper():
        gene_variations.extend([
            'systemic RNAi deficient 1', 'SID1', 'dsRNA transporter',
            'RNA interference systemic'
        ])
    elif 'EGO1' in clean_gene.upper():
        gene_variations.extend(['enhancer of glp-1', 'EGO1', 'RNA polymerase'])
    elif 'MUT7' in clean_gene.upper():
        gene_variations.extend(['mutator 7', 'MUT7', 'RNAi pathway'])
    
    return gene_variations

def add_rnai_genes_to_all_organisms():
    """Add RNAi pathway genes to all organisms in the system"""
    
    # Get the original organism suggestions
    original_suggestions = get_organism_suggestions_with_gene_targets_original()
    
    # Create a deep copy to modify
    enhanced_suggestions = {}
    
    for category, subcategories in original_suggestions.items():
        enhanced_suggestions[category] = {}
        
        for subcategory, organisms in subcategories.items():
            enhanced_suggestions[category][subcategory] = []
            
            for organism_item in organisms:
                if len(organism_item) == 3:
                    common_name, scientific_name, gene_targets = organism_item
                    
                    # Determine organism type
                    organism_type = determine_organism_type(category, subcategory, scientific_name)
                    
                    # Generate species prefix
                    species_prefix = get_species_prefix(scientific_name)
                    
                    # Generate RNAi genes for this organism type
                    rnai_genes = generate_rnai_genes(species_prefix, organism_type)
                    
                    # Add RNAi genes to gene targets
                    enhanced_gene_targets = gene_targets.copy()
                    enhanced_gene_targets["RNAi pathway genes"] = rnai_genes
                    
                    # Add the enhanced organism to the list
                    enhanced_suggestions[category][subcategory].append(
                        (common_name, scientific_name, enhanced_gene_targets)
                    )
                    
                else:
                    # Handle old format organisms (without gene targets)
                    common_name, scientific_name = organism_item
                    
                    # Determine organism type
                    organism_type = determine_organism_type(category, subcategory, scientific_name)
                    
                    # Generate species prefix
                    species_prefix = get_species_prefix(scientific_name)
                    
                    # Create basic gene targets with RNAi genes
                    basic_gene_targets = {
                        "Essential genes": ["16S rRNA", "18S rRNA", "ACT1", "TUB1", "EF1A"],
                        "RNAi pathway genes": generate_rnai_genes(species_prefix, organism_type)
                    }
                    
                    # Add the enhanced organism to the list
                    enhanced_suggestions[category][subcategory].append(
                        (common_name, scientific_name, basic_gene_targets)
                    )
    
    return enhanced_suggestions

def get_organism_suggestions_with_gene_targets():
    """Get agricultural pest and pathogen suggestions with comprehensive RNAi pathway genes"""
    
    # Call the function that adds RNAi genes to all organisms
    return add_rnai_genes_to_all_organisms()

def get_organism_suggestions_with_gene_targets_original():
    """Get agricultural pest and pathogen suggestions organized by category with comprehensive gene targets"""
    return {
        "🍄 Fungal Pathogens": {
            "Fusarium species": [
                ("Fusarium wilt", "Fusarium oxysporum", {
                    "Essential genes": ["ACT1 (actin)", "TUB2 (tubulin)", "EF1A (elongation factor)", "RPB2 (RNA polymerase)", "LSU (large subunit rRNA)"],
                    "Pathogenicity genes": ["SIX1-14 (secreted in xylem)", "FTF1 (transcription factor)", "FMK1 (MAPK)", "SGE1 (cutinase)", "PEL1 (pectate lyase)"],
                    "Secondary metabolite genes": ["FUM1 (fumonisin biosynthesis)", "TRI5 (trichothecene biosynthesis)", "PKS4 (polyketide synthase)", "BIK1 (bikaverin)", "FUS1 (fusarin)"],
                    "Cell wall genes": ["CHI1 (chitinase)", "GEL1 (β-1,3-glucanase)", "CHS1 (chitin synthase)", "FKS1 (β-1,3-glucan synthase)", "PMI1 (mannose-6-phosphate isomerase)"],
                    "Resistance targets": ["CYP51 (sterol 14α-demethylase)", "SDH (succinate dehydrogenase)", "QoI (cytochrome bc1)", "MBC (β-tubulin)", "DMI (sterol biosynthesis)"]
                }),
                ("Fusarium head blight", "Fusarium graminearum", {
                    "Essential genes": ["ACT1 (actin)", "TUB1 (α-tubulin)", "TUB2 (β-tubulin)", "EF1A (elongation factor)", "RPB2 (RNA polymerase II)"],
                    "Trichothecene pathway": ["TRI1-TRI16 (trichothecene cluster)", "TRI101 (trichothecene resistance)", "TRI201 (regulatory)", "TRI301 (transport)", "TRI401 (efflux pump)"],
                    "DON biosynthesis": ["TRI4 (trichodiene oxygenase)", "TRI11 (isotrichodermin C-15 hydroxylase)", "TRI3 (15-O-acetyltransferase)", "TRI7 (C-4 hydroxylase)", "TRI13 (isotrichodermol dehydrogenase)"],
                    "Pathogenicity factors": ["MGV1 (major facilitator)", "TRI12 (efflux pump)", "FgCHS1 (chitin synthase)", "FgPKS4 (polyketide synthase)", "FgLAE1 (global regulator)"],
                    "Stress response": ["HOG1 (osmoregulation)", "FgOS2 (osmotic stress)", "FgSLT2 (cell wall integrity)", "ATF1 (oxidative stress)", "FgAP1 (AP-1 transcription factor)"]
                }),
                ("Fusarium crown rot", "Fusarium culmorum", {
                    "Housekeeping genes": ["ACT1", "TUB2", "EF1A", "RPB2", "LSU rRNA"],
                    "Pathogenicity": ["CUT1 (cutinase)", "PEL2 (pectinase)", "XYL1 (xylanase)", "CEL1 (cellulase)", "AMY1 (amylase)"],
                    "Secondary metabolites": ["ZEA1-2 (zearalenone)", "TRI cluster", "AUR1 (aurofusarin)", "CUL1 (culmorin)", "BUT1 (butenolide)"],
                    "Survival genes": ["HSP70 (heat shock)", "SOD1 (superoxide dismutase)", "CAT1 (catalase)", "GPX1 (glutathione peroxidase)", "TRX1 (thioredoxin)"]
                }),
                ("Fusarium root rot", "Fusarium solani", {
                    "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "HSP70"],
                    "Pathogenicity factors": ["FSOL1-10 (F. solani specific)", "CUT1-5 (cutinases)", "PEL1-3 (pectate lyases)", "XYL1-2 (xylanases)", "CEL1-2 (cellulases)"],
                    "Secondary metabolites": ["FUM1-3 (fumonisin)", "TRI1-16 (trichothecene)", "ZEA1-2 (zearalenone)", "FUS1-5 (fusarin)", "BIK1-3 (bikaverin)"],
                    "Resistance mechanisms": ["CYP51A1-B1", "SDH1-4", "ABC1-20", "MFS1-15", "GST1-10"]
                }),
                ("Fusarium ear rot", "Fusarium proliferatum", {
                    "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "ITS1-2"],
                    "Fumonisin biosynthesis": ["FUM1-21 (fumonisin cluster)", "FUM8 (polyketide synthase)", "FUM6 (aminotransferase)", "FUM3 (C-5 oxygenase)", "FUM19 (transporter)"],
                    "Pathogenicity": ["FPR1-10 (F. proliferatum specific)", "CUT1-3", "PEL1-2", "XYL1", "CEL1"],
                    "Host interaction": ["HOST1-5 (host-specific)", "ADH1-3 (adhesion)", "INV1-2 (invasion)", "COL1-3 (colonization)"]
                })
            ],
            "Other fungi": [
                ("Gray mold", "Botrytis cinerea", {
                    "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "HSP70"],
                    "Pathogenicity factors": ["BCG1 (α-galactosidase)", "BMP1 (metalloprotease)", "BCP1 (cerato-platanin)", "BOA1 (botrydial)", "BCR1 (ABC transporter)"],
                    "Cell wall degrading": ["BcPG1-6 (polygalacturonases)", "BcPME1 (pectin methylesterase)", "BcXYL1 (xylanase)", "BcCEL1 (cellulase)", "BcCUT1 (cutinase)"],
                    "Secondary metabolites": ["BOT1-5 (botrydial cluster)", "DHN1 (1,8-dihydroxynaphthalene)", "PKS1-13 (polyketide synthases)", "NPS1-6 (nonribosomal peptide synthetases)"],
                    "Resistance mechanisms": ["ABC1-50 (ABC transporters)", "MFS1-20 (major facilitator superfamily)", "CYP1-20 (cytochrome P450s)", "GST1-10 (glutathione S-transferases)"]
                }),
                ("White mold", "Sclerotinia sclerotiorum", {
                    "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "CAL1"],
                    "Pathogenicity factors": ["SSPG1 (polygalacturonase)", "SsPME1 (pectin methylesterase)", "SsCUT1 (cutinase)", "SsNEP1 (necrosis-inducing protein)", "SsCP1 (cysteine protease)"],
                    "Oxalic acid pathway": ["OAH1 (oxaloacetate hydrolase)", "OXA1 (oxalate oxidase)", "OXD1 (oxalate decarboxylase)", "OMT1 (oxalate metabolism)", "OXT1 (oxalate transporter)"],
                    "Sclerotia formation": ["SMK1 (sclerotial development)", "SLT1 (sclerotial maturation)", "SCL1 (sclerotial pigmentation)", "MBF1 (melanin biosynthesis)", "LAC1 (laccase)"]
                }),
                ("Late blight", "Phytophthora infestans", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPB1", "COX1 (cytochrome c oxidase)"],
                    "Effector genes": ["AVR1-11 (avirulence genes)", "PexRD1-36 (Phytophthora expressed)", "Pi02860-Pi17316 (candidate effectors)", "RXLR1-100 (effector motif)", "CRN1-50 (crinkler effectors)"],
                    "RXLR effectors": ["PiAVR3a", "PiAVR4", "PiAVRblb1", "PiAVRblb2", "PiAVRvnt1", "Pi02860", "Pi03192", "Pi04314", "Pi07569", "Pi09585"],
                    "Cell wall degrading": ["PiCBEL1-15 (cellulose-binding elicitor lectin)", "PiGH12-1 (endoglucanase)", "PiPL1-20 (pectate lyases)", "PiXEG1 (xyloglucanase)", "PiPG1-5 (polygalacturonases)"],
                    "Pathogenicity factors": ["PiINF1 (infestin)", "PiNPP1 (necrosis-inducing protein)", "PiEPI1-10 (epidermis-specific)", "PiHAM34 (haustorial membrane)", "PiMCF1 (mycelium-cyst transition)"]
                }),
                ("Powdery mildew", "Golovinomyces ambrosiae", {
                    "Essential genes": ["ACT1", "TUB2", "EF1A", "RPB2", "ITS1-2"],
                    "Pathogenicity": ["GAH1 (haustorium formation)", "GAC1 (conidiophore development)", "GAS1 (spore germination)", "GAP1 (penetration)", "GAA1 (appressorium formation)"],
                    "Effectors": ["CSEP1-100 (candidate secreted effector proteins)", "GAE1-50 (G. ambrosiae effectors)", "AVR1-10 (avirulence candidates)", "HAU1-20 (haustorial expressed)"],
                    "Sterol biosynthesis": ["CYP51A1", "CYP51B1", "ERG1 (squalene epoxidase)", "ERG7 (lanosterol synthase)", "ERG11 (sterol 14α-demethylase)"]
                }),
                ("Rust disease", "Puccinia graminis", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPB1", "COI"],
                    "Pathogenicity factors": ["PG1-50 (P. graminis specific)", "AVR1-20 (avirulence genes)", "EFF1-30 (effector proteins)", "SEC1-10 (secreted proteins)", "HST1-5 (host-specific toxins)"],
                    "Life cycle genes": ["TEL1-5 (teliospore formation)", "BAS1-3 (basidiospore)", "PYC1-2 (pycniospore)", "AEC1-3 (aeciospore)", "URE1-5 (urediniospore)"],
                    "Host interaction": ["HOST1-10 (host recognition)", "PEN1-5 (penetration)", "COL1-8 (colonization)", "SPR1-5 (sporulation)"]
                }),
                ("Smut disease", "Ustilago maydis", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPB1", "HSP70"],
                    "Pathogenicity factors": ["UM1-100 (U. maydis specific)", "EFF1-50 (effector proteins)", "SEC1-20 (secreted proteins)", "CWP1-10 (cell wall proteins)", "ENZ1-15 (enzymes)"],
                    "Mating and development": ["MAT1-2 (mating type)", "B1-6 (b locus)", "A1-4 (a locus)", "CLP1 (clamp connection)", "DIC1-3 (dikaryon formation)"],
                    "Host specificity": ["HOST1-8 (host recognition)", "MAI1-5 (maize interaction)", "TIS1-3 (tissue specificity)", "SIZ1-2 (size control)"]
                })
            ]
        },
        
        "🐛 Insect Pests": {
            "Mites": [
                ("Two-spotted spider mite", "Tetranychus urticae", {
                    "Essential genes": ["ACT1 (actin)", "TUB1 (tubulin)", "EF1A (elongation factor)", "RPL32 (ribosomal protein L32)", "RPS3 (ribosomal protein S3)"],
                    "Detoxification genes": ["CYP1-100 (cytochrome P450s)", "GST1-30 (glutathione S-transferases)", "EST1-20 (esterases)", "UGT1-15 (UDP-glucuronosyltransferases)", "ABC1-50 (ABC transporters)"],
                    "Acaricide resistance": ["AChE (acetylcholinesterase)", "VGSC (voltage-gated sodium channel)", "RDL (GABA receptor)", "nAChR (nicotinic acetylcholine receptor)", "GluCl (glutamate-gated chloride channel)"],
                    "Development genes": ["JH1-3 (juvenile hormone)", "ECR (ecdysone receptor)", "USP (ultraspiracle)", "E74 (ecdysone response)", "BR-C (broad complex)"],
                    "Reproduction genes": ["VIT1-6 (vitellogenin)", "VTG1-3 (vitellogenin)", "CHR (chorion)", "EGG1-5 (egg development)", "EMB1-10 (embryogenesis)"]
                }),
                ("European red mite", "Panonychus ulmi", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPL32", "COI (cytochrome oxidase I)"],
                    "Resistance genes": ["CYP1-50", "GST1-20", "EST1-15", "P450-1-25", "MFO1-10 (mixed function oxidases)"],
                    "Cold tolerance": ["HSP70 (heat shock protein)", "AFP1-3 (antifreeze proteins)", "TRE1 (trehalose)", "GLY1-2 (glycerol)", "CRY1-2 (cryoprotectants)"],
                    "Diapause genes": ["DIA1-5 (diapause-associated)", "CLK (clock)", "PER (period)", "TIM (timeless)", "CYC (cycle)"]
                }),
                ("Broad mite", "Polyphagotarsonemus latus", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPL32", "COI"],
                    "Host range genes": ["HOST1-10 (host specificity)", "DET1-5 (detoxification)", "ADH1-3 (adhesion)", "PEN1-2 (penetration)", "COL1-3 (colonization)"],
                    "Resistance mechanisms": ["CYP1-30", "GST1-15", "EST1-10", "ABC1-20", "MFS1-10"],
                    "Development": ["JH1-2", "ECR", "USP", "E74", "BR-C"]
                }),
                ("Russet mite", "Aculops lycopersici", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPL32", "COI"],
                    "Tomato adaptation": ["TOM1-5 (tomato-specific)", "LYC1-3 (lycopersicon)", "SOL1-2 (solanum)", "ADH1-2 (adhesion)", "PEN1 (penetration)"],
                    "Resistance genes": ["CYP1-25", "GST1-12", "EST1-8", "ABC1-15", "MFS1-8"],
                    "Feeding behavior": ["FED1-5 (feeding)", "SAL1-3 (salivary)", "GUT1-3 (gut-specific)", "DIG1-2 (digestion)"]
                })
            ],
            "Sucking insects": [
                ("Silverleaf whitefly", "Bemisia tabaci", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "16S rRNA"],
                    "Insecticide resistance": ["CYP6CM1", "CYP4C64", "CYP4G61", "CYP4G70", "GST1-15", "EST1-10", "ABC1-30", "nAChR (nicotinic receptor)", "VGSC (sodium channel)"],
                    "Biotype markers": ["mtCOI (mitochondrial COI)", "ITS1", "16S rRNA", "28S rRNA", "RAPD markers"],
                    "Virus transmission": ["GroEL (endosymbiont)", "HSP70", "cyclophilin", "importin-α", "karyopherin", "VP1-4 (viral proteins)"],
                    "Endosymbiont genes": ["Portiera (P-endosymbiont)", "Hamiltonella", "Arsenophonus", "Cardinium", "Wolbachia", "Rickettsia"]
                }),
                ("Greenhouse whitefly", "Trialeurodes vaporariorum", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "COII"],
                    "Development markers": ["JH (juvenile hormone)", "ECR (ecdysone receptor)", "CHI1-3 (chitinase)", "CHS1-2 (chitin synthase)", "TRE1-2 (trehalase)"],
                    "Host plant interaction": ["SUC1-3 (sucrase)", "APH1-2 (aphid-like stylet)", "SAL1-3 (salivary)", "GUT1-5 (gut-specific)", "PHE1-3 (phenoloxidase)"]
                }),
                ("Green peach aphid", "Myzus persicae", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "COII"],
                    "Insecticide resistance": ["MACE (modified acetylcholinesterase)", "kdr (knockdown resistance)", "RDL (GABA receptor)", "CYP6CY3", "CYP4CJ1", "GST1-10", "EST1-8", "ABC1-20"],
                    "Morph determination": ["APH1 (apterous)", "WIN1 (wingless)", "VG1 (vestigial)", "DSX1 (doublesex)", "FRU1 (fruitless)"],
                    "Virus transmission": ["PLRV (potato leafroll virus)", "PVY (potato virus Y)", "CMV (cucumber mosaic virus)", "receptor proteins", "helper factors"],
                    "Endosymbiont": ["Buchnera aphidicola", "trpA-E (tryptophan synthesis)", "aroA-Q (aromatic amino acids)", "ilv (isoleucine-valine)", "leu (leucine)", "phe (phenylalanine)"]
                }),
                ("Cotton aphid", "Aphis gossypii", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "16S rRNA"],
                    "Host specialization": ["HSP1-10 (host selection)", "DET1-5 (detoxification)", "FED1-3 (feeding behavior)", "GOT1-5 (gossypol tolerance)", "TAN1-3 (tannin tolerance)"],
                    "Polyphenism": ["WIN1-5 (wing development)", "ALT1-3 (alternate morph)", "ENV1-5 (environmental response)", "DEN1-3 (density-dependent)", "PHO1-3 (photoperiod)"]
                }),
                ("Asian citrus psyllid", "Diaphorina citri", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "16S rRNA"],
                    "CLas transmission": ["CLas1-10 (Candidatus Liberibacter asiaticus)", "HSP1-5 (heat shock)", "SEC1-5 (secretion)", "TRP1-3 (transporter)", "BIND1-3 (binding)"],
                    "Citrus adaptation": ["CIT1-5 (citrus-specific)", "RUT1-3 (rutaceae)", "FLAV1-2 (flavonoid)", "TERP1-2 (terpene)", "OIL1-2 (essential oil)"],
                    "Development": ["JH1-2", "ECR", "USP", "E74", "BR-C"]
                })
            ],
            "Thrips": [
                ("Western flower thrips", "Frankliniella occidentalis", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "18S rRNA"],
                    "Insecticide resistance": ["CYP1-30", "GST1-15", "EST1-10", "AChE", "VGSC", "RDL", "GluCl", "nAChR"],
                    "Virus transmission": ["TSWV (tomato spotted wilt virus)", "INSV (impatiens necrotic spot virus)", "receptor1-3", "helper1-2", "vector1-5"],
                    "Host preference": ["ORN1-10 (olfactory)", "GRN1-5 (gustatory)", "CHE1-8 (chemoreception)", "HOST1-5 (host selection)", "FED1-3 (feeding)"],
                    "Development": ["JHE (juvenile hormone esterase)", "JH1-3", "ECR", "USP", "E74", "BR-C", "CHI1-2", "CHS1-2"]
                }),
                ("Onion thrips", "Thrips tabaci", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "COI", "COII"],
                    "Arrhenotoky": ["SEX1-5 (sex determination)", "HAP1-3 (haplodiploidy)", "CSD (complementary sex determiner)", "FEM1-2 (feminizer)", "DSX"],
                    "Onion adaptation": ["ALL1-5 (alliin tolerance)", "SUL1-3 (sulfur metabolism)", "ONI1-5 (onion-specific)", "LAC1-2 (lachrymatory factor)", "THI1-3 (thiosulfinate)"]
                })
            ]
        },

        "🦠 Bacterial Pathogens": {
            "Erwinia species": [
                ("Fire blight", "Erwinia amylovora", {
                    "Essential genes": ["16S rRNA", "23S rRNA", "gyrA", "gyrB", "rpoB", "rpoD"],
                    "Pathogenicity factors": ["hrpA-W (type III secretion)", "dspA/E (disease-specific)", "amsA-K (amylovoran synthesis)", "rcsA-C (regulation)", "avrRpt2EA"],
                    "Type III effectors": ["eop1-4 (effector of pathogenicity)", "hopA1EA", "hopC1EA", "avrE1", "dspA/E", "eop2", "eop3", "eop4"],
                    "Exopolysaccharide": ["amsA-K (amylovoran)", "galE", "galF", "ugpA-E", "wza-c", "gmd", "fcl", "rmlA-D"],
                    "Virulence regulation": ["rcsA-C", "kdpD/E", "ompR/envZ", "phoP/Q", "rpoS", "fur", "crp", "ihfA/B"]
                })
            ],
            "Pseudomonas species": [
                ("Bacterial speck", "Pseudomonas syringae", {
                    "Essential genes": ["16S rRNA", "gyrA", "gyrB", "rpoB", "rpoD"],
                    "Type III secretion": ["hrpA-U", "hrcC", "hrcJ", "hrcN", "hrcQ-U", "hrpZ", "hrpA", "hrpG", "hrpL"],
                    "Effector proteins": ["avrE1", "hopA1", "hopM1", "hopZ1-3", "avrPto", "avrPtoB", "avrRpm1", "avrRpt2", "hopF2", "hopG1"],
                    "Toxin production": ["coronatine biosynthesis", "cmaA-T", "cfa1-9", "syringomycin", "syrA-E", "syringopeptin", "sypA-C"],
                    "Ice nucleation": ["inaA-Z", "ice nucleation proteins", "ina genes", "frost injury", "epiphytic survival"]
                })
            ],
            "Xanthomonas species": [
                ("Bacterial blight", "Xanthomonas campestris", {
                    "Essential genes": ["16S rRNA", "gyrA", "gyrB", "rpoB", "atpD"],
                    "Xanthan production": ["gumB-M", "xanA-B", "xanthan gum", "EPS production", "biofilm formation"],
                    "Type III system": ["hrpA-F", "hrcC", "hrcJ", "hrcN", "hrcQ-U", "hpaA-F", "hpaP"],
                    "TAL effectors": ["avrBs3 family", "PthA1-4", "TAL1-20", "transcription activator-like", "DNA-binding repeats"],
                    "Pathogenicity factors": ["rpfA-G (regulation)", "clp (cellulase)", "prt (protease)", "man (mannanase)", "xps (secretion)"]
                })
            ],
            "Ralstonia species": [
                ("Bacterial wilt", "Ralstonia solanacearum", {
                    "Essential genes": ["16S rRNA", "gyrA", "gyrB", "rpoB", "egl"],
                    "Type III secretion": ["hrpA-Y", "hrcC", "hrcJ", "hrcN", "hrcQ-U", "popA-F", "ripA-Z"],
                    "Rip effectors": ["ripA1-Z9", "popP1-2", "popF1-4", "avrA", "ripG1-7", "ripP1-2", "ripE1", "ripF1"],
                    "EPS production": ["epsA-R", "exopolysaccharide", "biofilm", "wilt induction", "vascular plugging"],
                    "Phylotypes": ["phylotype I-IV", "sequevars", "biovars 1-5", "race 1-5", "geographic distribution"]
                })
            ],
            "Common bacteria": [
                ("Crown gall", "Agrobacterium tumefaciens", {
                    "Essential genes": ["16S rRNA", "gyrA", "gyrB", "rpoB", "atpD"],
                    "Ti plasmid genes": ["virA-G (virulence)", "tms1-2 (tumor morphology)", "tmr (tumor morphology)", "tss (tumor size)", "ipt (isopentenyl transferase)"],
                    "T-DNA transfer": ["T-DNA border sequences", "overdrive sequences", "virD1-4 (T-DNA processing)", "virE1-2 (T-DNA protection)", "virB1-11 (T4SS)"],
                    "Opine catabolism": ["occ (octopine)", "nop (nopaline)", "agr (agropine)", "man (mannopine)", "suc (succinamopine)"],
                    "Plant interaction": ["chvA-B (chromosomal virulence)", "exoC (exopolysaccharide)", "cel (cellulose)", "att (attachment)", "biofilm formation"]
                })
            ]
        },

        "🦠 Plant Viruses": {
            "Tobamoviruses": [
                ("Tobacco mosaic virus", "Tobacco mosaic virus", {
                    "Structural genes": ["MP (movement protein)", "CP (coat protein)", "Rep (replicase)", "helicase domain", "polymerase domain"],
                    "Movement": ["MP (30K movement protein)", "cell-to-cell movement", "plasmodesmata gating", "viral transport", "cytoskeleton interaction"],
                    "Replication": ["RdRp (RNA-dependent RNA polymerase)", "126K protein", "183K protein", "methyltransferase", "helicase"],
                    "Host interaction": ["elicitor recognition", "hypersensitive response", "systemic acquired resistance", "pathogenesis-related proteins"],
                    "Resistance genes": ["N gene", "Tm-1", "Tm-2", "Tm-22", "resistance-breaking strains"]
                }),
                ("Tomato mosaic virus", "Tomato mosaic virus", {
                    "Essential genes": ["CP (coat protein)", "MP (movement protein)", "Rep (replicase)", "126K protein", "183K protein"],
                    "Pathogenicity": ["virulence determinants", "host range", "symptom expression", "systemic movement"],
                    "Resistance breaking": ["resistance-breaking strains", "Tm-2 resistance", "point mutations", "strain classification"]
                })
            ],
            "Other viruses": [
                ("Beet curly top virus", "Beet curly top virus", {
                    "Essential genes": ["C1 (replication initiator)", "C2 (transcription activator)", "C3 (replication enhancer)", "C4 (pathogenicity)", "V1 (coat protein)", "V2 (movement protein)"],
                    "Geminivirus features": ["circular ssDNA", "bipartite genome", "rolling circle replication", "transcription activation", "silencing suppression"],
                    "Host range": ["beet", "tomato", "pepper", "bean", "spinach", "squash", "cucumber"],
                    "Vector transmission": ["beet leafhopper", "Circulifer tenellus", "persistent transmission", "propagative transmission"],
                    "Pathogenicity factors": ["C4 (symptom determinant)", "C2 (transcription activator)", "V2 (movement protein)", "silencing suppression", "host range determination"]
                }),
                ("Arabis mosaic virus", "Arabis mosaic virus", {
                    "Essential genes": ["RNA1 (replication)", "RNA2 (movement)", "CP (coat protein)", "MP (movement protein)", "Rep (replicase)"],
                    "Nepovirus features": ["bipartite RNA genome", "polyprotein processing", "3' poly(A) tail", "5' VPg", "icosahedral particles"],
                    "Host range": ["arabis", "strawberry", "raspberry", "grapevine", "tobacco", "cucumber"],
                    "Vector transmission": ["nematode vectors", "Xiphinema diversicaudatum", "X. coxi", "soil transmission", "seed transmission"],
                    "Pathogenicity": ["systemic infection", "mosaic symptoms", "stunting", "yield reduction", "latent infection"]
                }),
                ("Alfalfa mosaic virus", "Alfalfa mosaic virus", {
                    "Essential genes": ["RNA1 (P1, P2)", "RNA2 (P3)", "RNA3 (MP, CP)", "P1 (replicase)", "P2 (helicase)", "P3 (polymerase)", "MP (movement protein)", "CP (coat protein)"],
                    "Alfamovirus features": ["tripartite RNA genome", "bacilliform particles", "coat protein requirement", "genome activation", "replication"],
                    "Host range": ["alfalfa", "tobacco", "tomato", "pepper", "bean", "cucumber", "lettuce"],
                    "Vector transmission": ["aphid transmission", "non-persistent", "stylet-borne", "Myzus persicae", "Aphis gossypii"],
                    "Pathogenicity": ["mosaic symptoms", "yellowing", "stunting", "systemic infection", "yield reduction"]
                }),
                ("Cannabis cryptic virus", "Cannabis cryptic virus", {
                    "Essential genes": ["RNA1 (RdRp)", "RNA2 (CP)", "RdRp (RNA-dependent RNA polymerase)", "CP (coat protein)", "MP (movement protein)"],
                    "Cryptic virus features": ["persistent infection", "latent infection", "no symptoms", "vertical transmission", "seed transmission"],
                    "Host specificity": ["Cannabis sativa", "hemp", "marijuana", "endophytic", "systemic infection"],
                    "Transmission": ["seed transmission", "pollen transmission", "no vector", "vertical transmission", "graft transmission"],
                    "Molecular features": ["dsRNA genome", "icosahedral particles", "persistent infection", "no cell-to-cell movement", "replication in cytoplasm"]
                }),
                ("Lettuce chlorosis virus", "Lettuce chlorosis virus", {
                    "Essential genes": ["RNA1 (replication)", "RNA2 (movement)", "P1 (replicase)", "P2 (helicase)", "P3 (polymerase)", "MP (movement protein)", "CP (coat protein)"],
                    "Crinivirus features": ["bipartite RNA genome", "whitefly transmission", "phloem-limited", "long flexuous particles", "genome activation"],
                    "Host range": ["lettuce", "tomato", "pepper", "cucumber", "melon", "squash", "bean"],
                    "Vector transmission": ["whitefly transmission", "Bemisia tabaci", "Trialeurodes vaporariorum", "semi-persistent", "circulative"],
                    "Pathogenicity": ["chlorosis", "yellowing", "stunting", "phloem necrosis", "yield reduction"]
                }),
                ("Potato virus Y", "Potato virus Y", {
                    "Essential genes": ["P1", "HC-Pro", "P3", "6K1", "CI", "6K2", "VPg", "NIa-Pro", "NIb", "CP"],
                    "Silencing suppression": ["HC-Pro (helper component proteinase)", "P1", "RNA silencing suppression", "siRNA binding", "RISC complex"],
                    "Strain differentiation": ["PVY-O (ordinary)", "PVY-N (necrotic)", "PVY-C", "PVY-Z", "PVYNTN", "PVYN-Wi"],
                    "Vector transmission": ["aphid transmission", "stylet-borne", "non-persistent", "helper component", "virion retention"],
                    "Recombination": ["recombinant strains", "PVYNTN", "breakpoints", "fitness advantage", "emergence"]
                }),
                ("Cucumber mosaic virus", "Cucumber mosaic virus", {
                    "Essential genes": ["1a", "2a", "2b", "3a", "MP", "CP"],
                    "Satellite RNA": ["satRNA", "symptom modulation", "D-satRNA", "Y-satRNA", "WL1-satRNA"],
                    "Subgroups": ["subgroup I", "subgroup II", "subgroup IA", "subgroup IB", "phylogenetic classification"],
                    "Host range": ["wide host range", "over 1000 species", "monocots", "dicots", "woody plants"],
                    "Symptom determinants": ["2b protein", "satellite RNA", "strain-specific", "host-dependent", "environmental factors"]
                })
            ]
        },

        "🦠 Oomycetes": {
            "Water molds": [
                ("Pythium root rot", "Pythium ultimum", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPB1", "COX1"],
                    "Pathogenicity factors": ["PyCBEL1-10 (cellulose-binding elicitor lectin)", "PyPL1-8 (pectate lyases)", "PyCUT1-3 (cutinases)", "PyPRO1-5 (proteases)", "PyLIP1-3 (lipases)"],
                    "Zoospore motility": ["FLA1-20 (flagellar proteins)", "DYN1-5 (dynein)", "KIN1-8 (kinesin)", "TUB1-3 (tubulins)", "MOT1-10 (motility)"],
                    "Oospore formation": ["OOS1-15 (oospore genes)", "SEX1-5 (sexual reproduction)", "MAT1-3 (mating type)", "GER1-5 (germination)", "DOR1-3 (dormancy)"],
                    "Cell wall synthesis": ["CEL1-5 (cellulose synthase)", "CHI1-3 (chitinase)", "GEL1-3 (β-1,3-glucanase)", "CHS1 (chitin synthase)", "CAL1 (callose synthase)"]
                }),
                ("Pythium damping-off", "Pythium myriotylum", {
                    "Essential genes": ["ACT1", "TUB1", "EF1A", "RPB1", "COX1"],
                    "Pathogenicity factors": ["PmCBEL1-8 (cellulose-binding elicitor lectin)", "PmPL1-6 (pectate lyases)", "PmCUT1-2 (cutinases)", "PmPRO1-4 (proteases)", "PmLIP1-2 (lipases)"],
                    "Damping-off factors": ["DAMP1-5 (damping-off specific)", "ROOT1-3 (root infection)", "SEED1-2 (seed rot)", "STEM1-2 (stem rot)", "CROWN1 (crown rot)"],
                    "Host range": ["HOST1-10 (host specificity)", "CROP1-5 (crop-specific)", "SOL1-3 (solanaceous)", "CUC1-2 (cucurbit)", "LEG1-2 (legume)"],
                    "Environmental adaptation": ["TEMP1-3 (temperature)", "MOIST1-2 (moisture)", "pH1-2 (pH tolerance)", "SALT1 (salt tolerance)", "ANA1 (anaerobic)"]
                })
            ]
        },

        "🧬 Viroids": {
            "RNA pathogens": [
                ("Hop latent viroid", "Hop latent viroid", {
                    "Structural domains": ["P (pathogenicity)", "C (central)", "V (variable)", "T1 (terminal left)", "T2 (terminal right)"],
                    "Secondary structure": ["rod-like structure", "base-pairing", "hairpin loops", "bulges", "thermodynamic stability"],
                    "Variants": ["HLVd variants", "sequence diversity", "geographic strains", "hop cultivar adaptation"],
                    "Pathogenicity": ["latent infection", "hop stunt", "yield reduction", "brewing quality", "bitter compound"],
                    "Detection": ["RT-PCR", "northern blot", "dot blot", "in situ hybridization", "high-throughput sequencing"]
                })
            ]
        }
    }

def get_organism_suggestions():
    """Get agricultural pest and pathogen suggestions organized by category with comprehensive gene targets"""
    # Import the enhanced function
    return get_organism_suggestions_with_gene_targets()

def search_organism_with_gene_targets(organism_name, email, api_key=None, max_sequences=10):
    """Enhanced organism search that includes gene target information"""
    
    # Get gene targets for the organism
    suggestions = get_organism_suggestions_with_gene_targets()
    organism_targets = None
    
    # Normalize the input organism name
    organism_name_lower = organism_name.lower().strip()
    
    # Find matching organism and extract gene targets
    for category, subcategories in suggestions.items():
        for subcategory, organisms in subcategories.items():
            for item in organisms:
                if len(item) == 3:  # New format with gene targets
                    common_name, scientific_name, gene_targets = item
                else:  # Old format without gene targets
                    common_name, scientific_name = item
                    gene_targets = {"Essential genes": ["16S rRNA", "18S rRNA", "ACT1", "TUB1", "EF1A"]}
                
                # Improved matching logic
                scientific_lower = scientific_name.lower().strip()
                common_lower = common_name.lower().strip()
                
                # Check for exact matches or partial matches
                if (organism_name_lower == scientific_lower or 
                    organism_name_lower == common_lower or
                    organism_name_lower in scientific_lower or 
                    scientific_lower in organism_name_lower or
                    organism_name_lower in common_lower or
                    common_lower in organism_name_lower):
                    
                    organism_targets = {
                        'organism': scientific_name,
                        'common_name': common_name,
                        'category': category,
                        'subcategory': subcategory,
                        'gene_targets': gene_targets
                    }
                    break
            if organism_targets:
                break
        if organism_targets:
            break
    
    return organism_targets

def display_gene_targets_interface(organism_targets):
    """Display gene targets interface in Streamlit with improved state management"""
    
    if organism_targets:
        st.success(f"📋 **Gene Targets Available for {organism_targets['common_name']} ({organism_targets['organism']})**")
        
        with st.expander("🧬 View Available Gene Targets", expanded=True):
            st.write(f"**Category:** {organism_targets['category']} → {organism_targets['subcategory']}")
            
            # Display gene targets by category
            for gene_category, genes in organism_targets['gene_targets'].items():
                st.write(f"**{gene_category}:**")
                for gene in genes:
                    st.write(f"  • {gene}")
                st.write("")
            
            # Gene target selection
            st.subheader("🎯 Select Gene Targets for Primer Design")
            
            # Priority-based selection
            essential_cats = [cat for cat in organism_targets['gene_targets'].keys() if 'essential' in cat.lower() or 'housekeeping' in cat.lower()]
            pathogen_cats = [cat for cat in organism_targets['gene_targets'].keys() if any(x in cat.lower() for x in ['pathogen', 'virulence', 'effector'])]
            resistance_cats = [cat for cat in organism_targets['gene_targets'].keys() if 'resistance' in cat.lower()]
            
            # Initialize default categories only once per organism
            organism_key = f"gene_categories_{organism_targets['organism']}"
            if organism_key not in st.session_state:
                default_categories = []
                if essential_cats:
                    default_categories.extend(essential_cats[:2])  # First 2 essential categories
                if pathogen_cats:
                    default_categories.extend(pathogen_cats[:1])   # First pathogenicity category
                if resistance_cats:
                    default_categories.extend(resistance_cats[:1]) # First resistance category
                
                # Fallback if no defaults found
                if not default_categories:
                    default_categories = [list(organism_targets['gene_targets'].keys())[0]]
                
                st.session_state[organism_key] = default_categories
            
            # Use organism-specific stored selection
            selected_categories = st.multiselect(
                "Choose gene categories to target:",
                list(organism_targets['gene_targets'].keys()),
                default=st.session_state.get(organism_key, []),
                help="Select which gene categories to focus on for primer design. Essential genes are recommended for reliable detection.",
                key=f"gene_category_selector_{organism_targets['organism']}"
            )
            
            # Update organism-specific session state only if selection changed
            if selected_categories != st.session_state.get(organism_key, []):
                st.session_state[organism_key] = selected_categories
            
            if selected_categories:
                # Display selection summary
                total_genes = sum(len(organism_targets['gene_targets'][cat]) for cat in selected_categories)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Selected Categories", len(selected_categories))
                with col2:
                    st.metric("Total Gene Targets", total_genes)
                with col3:
                    priority_score = sum(1 if get_gene_priority(cat) == "High" else 0.5 if get_gene_priority(cat) == "Medium" else 0.1 for cat in selected_categories)
                    st.metric("Priority Score", f"{priority_score:.1f}")
                
                # Show selected targets
                with st.expander("📋 Selected Gene Targets", expanded=False):
                    selected_genes = []
                    for category in selected_categories:
                        st.write(f"**{category}:** ({get_gene_priority(category)} Priority)")
                        genes = organism_targets['gene_targets'][category]
                        for gene in genes:
                            st.write(f"  • {gene}")
                            selected_genes.append(f"{category}: {gene}")
                        st.write(f"*{get_gene_use_recommendation(category)}*")
                        st.write("")
                
                # Store in session state - always update when selections change
                st.session_state.selected_gene_targets = {
                    'organism_info': organism_targets,
                    'selected_categories': selected_categories,
                    'selected_genes': selected_genes,
                    'total_targets': total_genes
                }
                
                return True
            else:
                # Clear session state when no categories are selected
                if 'selected_gene_targets' in st.session_state:
                    del st.session_state.selected_gene_targets
    
    return False

def display_results_with_gene_context():
    """Display primer results with gene target context - workflow aware"""
    
    # Check design mode
    design_mode = st.session_state.get('sequence_info', {}).get('design_mode', 'unknown')
    
    if design_mode == 'gene_targeted' and 'selected_gene_targets' in st.session_state:
        target_info = st.session_state.get('selected_gene_targets', {})
        
        st.subheader("🎯 Gene-Targeted Design Context")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Target Organism", target_info['organism_info']['common_name'])
        with col2:
            st.metric("Gene Categories", len(target_info['selected_categories']))
        with col3:
            st.metric("Total Gene Targets", target_info['total_targets'])
        with col4:
            high_priority = sum(1 for cat in target_info['selected_categories'] if get_gene_priority(cat) == "High")
            st.metric("High Priority Categories", high_priority)
        
        # Target-specific recommendations
        with st.expander("📋 Gene Target Recommendations", expanded=False):
            recommendations = []
            for category in target_info['selected_categories']:
                rec = get_gene_use_recommendation(category)
                priority = get_gene_priority(category)
                gene_count = len(target_info['organism_info']['gene_targets'][category])
                recommendations.append({
                    'Category': category,
                    'Priority': priority,
                    'Gene Count': gene_count,
                    'Recommendation': rec[:100] + "..." if len(rec) > 100 else rec
                })
            
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True)
    
    elif design_mode == 'conservation_based':
        st.subheader("🧬 Conservation-Based Design Context")
        analysis_metadata = st.session_state.get('analysis_metadata', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequences Analyzed", analysis_metadata.get('sequences_analyzed', 'N/A'))
        with col2:
            conservation_thresh = analysis_metadata.get('conservation_threshold', 0)
            st.metric("Conservation Threshold", f"{conservation_thresh:.0%}")
        with col3:
            specificity_tested = analysis_metadata.get('specificity_tested', False)
            st.metric("Specificity Tested", "Yes" if specificity_tested else "No")
        with col4:
            if specificity_tested:
                spec_thresh = analysis_metadata.get('specificity_threshold', 0)
                st.metric("Specificity Threshold", f"{spec_thresh:.0%}")
    
    elif design_mode == 'standard':
        st.subheader("⚡ Standard Design Context")
        st.info("Primers designed from single sequence using standard approach. Consider Gene-Targeted or Conservation-Based design for enhanced specificity.")

def export_with_gene_targets(primers, format_type="excel"):
    """Export primers with gene target information"""
    
    # Get organism and gene target info
    organism_info = st.session_state.get('sequence_info', {})
    gene_targets = st.session_state.get('selected_gene_targets', {})
    
    if format_type == "excel":
        return export_to_excel_with_targets(primers, gene_targets.get('organism_info'))
    else:
        # CSV export with gene target context
        data = []
        for i, primer in enumerate(primers):
            row_data = {
                'Primer_Pair': i + 1,
                'Target_Organism': gene_targets.get('organism_info', {}).get('organism', 'Unknown'),
                'Common_Name': gene_targets.get('organism_info', {}).get('common_name', 'Unknown'),
                'Organism_Category': gene_targets.get('organism_info', {}).get('category', 'Unknown'),
                'Selected_Gene_Categories': '; '.join(gene_targets.get('selected_categories', [])),
                'Total_Gene_Targets': gene_targets.get('total_targets', 0)
            }
            
            # Add primer-specific data
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                row_data.update({
                    'Forward_T7_Sequence': primer.forward_seq,
                    'Reverse_T7_Sequence': primer.reverse_seq,
                    'Forward_Core_Sequence': primer.core_forward_seq,
                    'Reverse_Core_Sequence': primer.core_reverse_seq,
                    'Core_Forward_Tm': round(primer.forward_tm, 2),
                    'Core_Reverse_Tm': round(primer.reverse_tm, 2),
                    'dsRNA_Size': primer.product_size,
                    'Primer_Type': 'T7_dsRNA'
                })
            else:
                row_data.update({
                    'Forward_Sequence': primer.forward_seq,
                    'Reverse_Sequence': primer.reverse_seq,
                    'Forward_Tm': round(primer.forward_tm, 2),
                    'Reverse_Tm': round(primer.reverse_tm, 2),
                    'Product_Size': primer.product_size,
                    'Primer_Type': 'Standard'
                })
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

def get_gene_priority(category):
    """Get priority level for gene category"""
    high_priority = ["Essential genes", "Pathogenicity genes", "Resistance targets", "Insecticide resistance", "Acaricide resistance"]
    medium_priority = ["Secondary metabolite genes", "Detoxification genes", "Development genes", "Effector genes", "Type III secretion", "RNAi pathway genes"]
    
    if category in high_priority:
        return "High"
    elif category in medium_priority:
        return "Medium"
    else:
        return "Low"

def get_gene_use_recommendation(category):
    """Get recommendation for gene category use"""
    recommendations = {
        "Essential genes": "Universal targets - always effective for species identification and basic primer design",
        "Pathogenicity genes": "Ideal for pathogen detection and virulence studies",
        "Resistance targets": "Critical for resistance monitoring and management strategies", 
        "Secondary metabolite genes": "Excellent for toxin detection and food safety applications",
        "Detoxification genes": "Essential for resistance monitoring in pest management",
        "Acaricide resistance": "Monitor for resistance development in mite populations",
        "Insecticide resistance": "Track resistance evolution in insect pest populations",
        "Development genes": "Target for growth regulation and lifecycle disruption",
        "Reproduction genes": "Fertility control and population management applications",
        "Effector genes": "Highly specific pathogenicity targets",
        "Cell wall degrading": "Virulence factors for disease mechanism studies",
        "Type III secretion": "Pathogenicity system targets",
        "Virulence regulation": "Regulatory control of disease development",
        "RNAi pathway genes": "Critical for gene silencing studies and RNAi-based control strategies. Essential for understanding RNA interference machinery and developing dsRNA-based treatments."
    }
    return recommendations.get(category, "General research and diagnostic applications")

def export_to_excel_with_targets(primers, organism_info):
    """Export primers to Excel with gene target information"""
    try:
        # Create main primer data
        primer_data = []
        for i, primer in enumerate(primers):
            row_data = {
                'Primer_Pair': i + 1,
                'Target_Organism': organism_info.get('organism', 'Unknown') if organism_info else 'Unknown',
                'Common_Name': organism_info.get('common_name', 'Unknown') if organism_info else 'Unknown',
                'Organism_Category': organism_info.get('category', 'Unknown') if organism_info else 'Unknown',
                'Organism_Subcategory': organism_info.get('subcategory', 'Unknown') if organism_info else 'Unknown'
            }
            
            # Add primer-specific data
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                row_data.update({
                    'Forward_T7_Sequence': primer.forward_seq,
                    'Reverse_T7_Sequence': primer.reverse_seq,
                    'Forward_Core_Sequence': primer.core_forward_seq,
                    'Reverse_Core_Sequence': primer.core_reverse_seq,
                    'Core_Forward_Tm': round(primer.forward_tm, 2),
                    'Core_Reverse_Tm': round(primer.reverse_tm, 2),
                    'dsRNA_Size': primer.product_size,
                    'Core_Forward_GC%': round(primer.gc_content_f, 2),
                    'Core_Reverse_GC%': round(primer.gc_content_r, 2),
                    'Forward_Start': primer.forward_start,
                    'Reverse_Start': primer.reverse_start,
                    'Penalty_Score': round(primer.penalty, 4),
                    'T7_Promoter': primer.t7_promoter_seq,
                    'Primer_Type': 'T7_dsRNA'
                })
            else:
                row_data.update({
                    'Forward_Sequence': primer.forward_seq,
                    'Reverse_Sequence': primer.reverse_seq,
                    'Forward_Tm': round(primer.forward_tm, 2),
                    'Reverse_Tm': round(primer.reverse_tm, 2),
                    'Product_Size': primer.product_size,
                    'Forward_GC%': round(primer.gc_content_f, 2),
                    'Reverse_GC%': round(primer.gc_content_r, 2),
                    'Forward_Start': primer.forward_start,
                    'Reverse_Start': primer.reverse_start,
                    'Penalty_Score': round(primer.penalty, 4),
                    'Primer_Type': 'Standard'
                })
            
            primer_data.append(row_data)
        
        # Create gene target data if available
        gene_target_data = []
        if organism_info and 'gene_targets' in organism_info:
            targets = organism_info['gene_targets']
            for category, genes in targets.items():
                for gene in genes:
                    gene_target_data.append({
                        'Gene_Category': category,
                        'Gene_Target': gene,
                        'Target_Organism': organism_info.get('organism', 'Unknown'),
                        'Priority': get_gene_priority(category),
                        'Recommended_Use': get_gene_use_recommendation(category)
                    })
        
        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Primer results sheet
            primer_df = pd.DataFrame(primer_data)
            primer_df.to_excel(writer, sheet_name='Primer_Results', index=False)
            
            # Gene targets sheet
            if gene_target_data:
                targets_df = pd.DataFrame(gene_target_data)
                targets_df.to_excel(writer, sheet_name='Gene_Targets', index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error exporting to Excel: {e}")
        return b""

def create_comprehensive_gene_target_export():
    """Create comprehensive gene target database export"""
    try:
        suggestions = get_organism_suggestions_with_gene_targets()
        data = []
        
        for category, subcategories in suggestions.items():
            for subcategory, organisms in subcategories.items():
                for item in organisms:
                    if len(item) == 3:
                        common_name, scientific_name, gene_targets = item
                    else:
                        common_name, scientific_name = item
                        gene_targets = {"Essential genes": ["16S rRNA", "18S rRNA", "ACT1", "TUB1", "EF1A"]}
                    
                    for gene_category, genes in gene_targets.items():
                        for gene in genes:
                            data.append({
                                'Organism_Category': category,
                                'Organism_Subcategory': subcategory,
                                'Common_Name': common_name,
                                'Scientific_Name': scientific_name,
                                'Gene_Category': gene_category,
                                'Gene_Name': gene,
                                'Priority': get_gene_priority(gene_category),
                                'Recommendation': get_gene_use_recommendation(gene_category),
                                'Target_Type': 'Pathogen' if any(x in category for x in ['🍄', '🦠']) else 'Pest'
                            })
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error creating gene target export: {e}")
        return pd.DataFrame()

def generate_gene_target_statistics():
    """Generate comprehensive statistics about available gene targets"""
    try:
        df = create_comprehensive_gene_target_export()
        
        if df.empty:
            return {}, {}, {}
        
        stats = {
            'Total Organisms': df['Scientific_Name'].nunique(),
            'Total Gene Targets': len(df),
            'Organism Categories': df['Organism_Category'].nunique(),
            'Gene Categories': df['Gene_Category'].nunique(),
            'High Priority Targets': len(df[df['Priority'] == 'High']),
            'Pathogen Targets': len(df[df['Target_Type'] == 'Pathogen']),
            'Pest Targets': len(df[df['Target_Type'] == 'Pest'])
        }
        
        category_counts = df['Organism_Category'].value_counts().to_dict()
        gene_category_counts = df['Gene_Category'].value_counts().to_dict()
        
        return stats, category_counts, gene_category_counts
    except Exception as e:
        st.error(f"Error generating statistics: {e}")
        return {}, {}, {}

# Configure Streamlit page
st.set_page_config(
    page_title="Automated Primer Designer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class PrimerPair:
    """Class to store primer pair information"""
    forward_seq: str
    reverse_seq: str
    forward_tm: float
    reverse_tm: float
    product_size: int
    gc_content_f: float
    gc_content_r: float
    forward_start: int
    reverse_start: int
    penalty: float = 0.0
    gene_target: str = "Standard Design"  # Specific gene target for this primer pair

def retry_with_backoff(max_retries=3, base_delay=0.34):
    """Decorator factory for NCBI operations with retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.3)
                        st.info(f"Retrying NCBI request in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    
                    return func(self, *args, **kwargs)
                    
                except (HTTPError, requests.exceptions.RequestException, Exception) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        st.error(f"NCBI request failed after {max_retries} attempts: {e}")
                        break
                    st.warning(f"NCBI request failed (attempt {attempt + 1}): {e}")
            
            return None
        return wrapper
    return decorator

class ResilientNCBIConnector:
    """Resilient NCBI connector with exponential backoff and retry logic"""
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.base_delay = 0.34 if not api_key else 0.1
        self.max_retries = 3
        self.timeout = 30
        
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3)
        return delay + jitter
    
    @retry_with_backoff(max_retries=3, base_delay=0.34)
    def search_sequences(self, query: str, database: str = "nucleotide", max_results: int = 100) -> List[str]:
        """Search with timeout and retry logic"""
        handle = Entrez.esearch(
            db=database, 
            term=query, 
            retmax=max_results,
            timeout=self.timeout
        )
        search_results = Entrez.read(handle)
        handle.close()
        return search_results["IdList"]
    
    @retry_with_backoff(max_retries=3, base_delay=0.34)
    def fetch_sequence(self, seq_id: str, database: str = "nucleotide") -> Optional[str]:
        """Fetch with timeout and retry logic"""
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
    
    @retry_with_backoff(max_retries=3, base_delay=0.34)
    def fetch_sequence_info(self, seq_id: str, database: str = "nucleotide") -> Dict:
        """Fetch sequence info with timeout and retry logic"""
        handle = Entrez.efetch(
            db=database, 
            id=seq_id, 
            rettype="gb", 
            retmode="text",
            timeout=self.timeout
        )
        record = SeqIO.read(handle, "genbank")
        handle.close()
        
        return {
            "id": record.id,
            "description": record.description,
            "length": len(record.seq),
            "organism": record.annotations.get("organism", "Unknown"),
            "sequence": str(record.seq)
        }
    
    def validate_api_key(self) -> bool:
        """Validate API key by making a test request"""
        try:
            test_ids = self.search_sequences("test", max_results=1)
            return True
        except:
            return False

# Keep the old class name for backward compatibility
NCBIConnector = ResilientNCBIConnector

class PrimerDesigner:
    """Main primer design class with T7 dsRNA functionality"""
    
    def __init__(self):
        self.default_params = {
            'PRIMER_OPT_SIZE': 20,
            'PRIMER_MIN_SIZE': 18,
            'PRIMER_MAX_SIZE': 25,
            'PRIMER_OPT_TM': 60.0,
            'PRIMER_MIN_TM': 57.0,
            'PRIMER_MAX_TM': 63.0,
            'PRIMER_MIN_GC': 40.0,
            'PRIMER_MAX_GC': 60.0,
            'PRIMER_MAX_POLY_X': 4,
            'PRIMER_SALT_MONOVALENT': 50.0,
            'PRIMER_DNA_CONC': 50.0,
            'PRIMER_MAX_NS_ACCEPTED': 0,
            'PRIMER_MAX_SELF_ANY': 12,
            'PRIMER_MAX_SELF_END': 8,
            'PRIMER_PAIR_MAX_COMPL_ANY': 12,
            'PRIMER_PAIR_MAX_COMPL_END': 8,
            'PRIMER_PRODUCT_SIZE_RANGE': [[75, 300], [300, 600], [600, 1000]],
            'PRIMER_NUM_RETURN': 20
        }
        
        # T7 promoter sequence for dsRNA production
        self.t7_promoter = "TAATACGACTCACTATAGGG"
    
    def calculate_gc_content(self, sequence: str) -> float:
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return (gc_count / len(sequence)) * 100 if sequence else 0
    
    def reverse_complement(self, sequence: str) -> str:
        """Generate reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
        return "".join(complement.get(base, base) for base in reversed(sequence.upper()))
    
    def design_primers(self, sequence: str, target_region: Optional[Tuple[int, int]] = None,
                      custom_params: Optional[Dict] = None, add_t7_promoter: bool = False, 
                      gene_target: str = "Standard Design") -> List[PrimerPair]:
        """Design primers with optional T7 promoter for dsRNA production"""
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)
        
        seq_args = {
            'SEQUENCE_ID': 'target',
            'SEQUENCE_TEMPLATE': sequence.upper()
        }
        
        if target_region:
            start, end = target_region
            seq_args['SEQUENCE_TARGET'] = [start, end - start]
        
        try:
            primer_results = primer3.bindings.designPrimers(seq_args, params)
            primers = []
            
            num_pairs = primer_results.get('PRIMER_PAIR_NUM_RETURNED', 0)
            
            for i in range(num_pairs):
                forward_seq = primer_results[f'PRIMER_LEFT_{i}_SEQUENCE']
                reverse_seq = primer_results[f'PRIMER_RIGHT_{i}_SEQUENCE']
                forward_start = primer_results[f'PRIMER_LEFT_{i}'][0]
                reverse_start = primer_results[f'PRIMER_RIGHT_{i}'][0]
                product_size = primer_results[f'PRIMER_PAIR_{i}_PRODUCT_SIZE']
                penalty = primer_results[f'PRIMER_PAIR_{i}_PENALTY']
                
                # Add T7 promoter if requested (for dsRNA production)
                if add_t7_promoter:
                    # Add T7 to both forward and reverse primers for bidirectional transcription
                    forward_seq_t7 = self.t7_promoter + forward_seq
                    reverse_seq_t7 = self.t7_promoter + reverse_seq
                    
                    # Calculate Tm for core primer sequence (without T7)
                    forward_tm = Tm_NN(forward_seq)
                    reverse_tm = Tm_NN(reverse_seq)
                    
                    # Store both original and T7-modified sequences
                    primer_pair = PrimerPair(
                        forward_seq=forward_seq_t7,  # T7 + primer
                        reverse_seq=reverse_seq_t7,  # T7 + primer  
                        forward_tm=forward_tm,       # Tm of core primer
                        reverse_tm=reverse_tm,       # Tm of core primer
                        product_size=product_size,
                        gc_content_f=self.calculate_gc_content(forward_seq),  # GC of core primer
                        gc_content_r=self.calculate_gc_content(reverse_seq),  # GC of core primer
                        forward_start=forward_start,
                        reverse_start=reverse_start,
                        penalty=penalty,
                        gene_target=gene_target
                    )
                    
                    # Store additional T7 information
                    primer_pair.core_forward_seq = forward_seq
                    primer_pair.core_reverse_seq = reverse_seq
                    primer_pair.has_t7_promoter = True
                    primer_pair.t7_promoter_seq = self.t7_promoter
                    
                else:
                    # Standard primers without T7
                    forward_tm = Tm_NN(forward_seq)
                    reverse_tm = Tm_NN(reverse_seq)
                    
                    primer_pair = PrimerPair(
                        forward_seq=forward_seq,
                        reverse_seq=reverse_seq,
                        forward_tm=forward_tm,
                        reverse_tm=reverse_tm,
                        product_size=product_size,
                        gc_content_f=self.calculate_gc_content(forward_seq),
                        gc_content_r=self.calculate_gc_content(reverse_seq),
                        forward_start=forward_start,
                        reverse_start=reverse_start,
                        penalty=penalty,
                        gene_target=gene_target
                    )
                    
                    primer_pair.has_t7_promoter = False
                
                primers.append(primer_pair)
            
            return primers
            
        except Exception as e:
            st.error(f"Error in primer design: {e}")
            return []
    
    def calculate_dsrna_properties(self, primer_pair: PrimerPair, sequence: str) -> Dict:
        """Calculate properties relevant for dsRNA production"""
        if not hasattr(primer_pair, 'has_t7_promoter') or not primer_pair.has_t7_promoter:
            return {}
        
        try:
            # Extract the target region that will be transcribed
            target_start = primer_pair.forward_start
            target_end = primer_pair.reverse_start
            target_sequence = sequence[target_start:target_end + 1]
            
            # Calculate dsRNA properties
            dsrna_length = len(target_sequence)
            gc_content = self.calculate_gc_content(target_sequence)
            
            # Check for optimal dsRNA characteristics
            optimal_length = 100 <= dsrna_length <= 500  # Optimal for RNAi
            moderate_gc = 40 <= gc_content <= 60  # Avoid extreme GC content
            
            # Calculate T7 transcription efficiency factors
            # T7 prefers certain nucleotides at +1 position (G is best)
            transcription_start = target_sequence[0] if target_sequence else 'N'
            t7_efficiency = "High" if transcription_start == 'G' else "Moderate" if transcription_start in ['A', 'C'] else "Low"
            
            return {
                'dsrna_length': dsrna_length,
                'dsrna_gc_content': gc_content,
                'target_sequence': target_sequence,
                'optimal_length': optimal_length,
                'moderate_gc': moderate_gc,
                'transcription_efficiency': t7_efficiency,
                'transcription_start': transcription_start,
                'estimated_yield': "High" if optimal_length and moderate_gc else "Moderate" if optimal_length or moderate_gc else "Low"
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def design_primers_with_complementarity(self, sequence: str, target_region: Optional[Tuple[int, int]] = None,
                                          custom_params: Optional[Dict] = None, add_t7_promoter: bool = False, 
                                          gene_target: str = "Standard Design", target_type: str = "linear",
                                          include_reverse_orientation: bool = False) -> Tuple[List[PrimerPair], Dict]:
        """Design primers and return both primers and raw Primer3 results"""
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)
        
        seq_args = {
            'SEQUENCE_ID': 'target',
            'SEQUENCE_TEMPLATE': sequence.upper()
        }
        
        if target_region:
            start, end = target_region
            seq_args['SEQUENCE_TARGET'] = [start, end - start]
        
        try:
            primer_results = primer3.bindings.designPrimers(seq_args, params)
            primers = []
            
            num_pairs = primer_results.get('PRIMER_PAIR_NUM_RETURNED', 0)
            
            for i in range(num_pairs):
                forward_seq = primer_results[f'PRIMER_LEFT_{i}_SEQUENCE']
                reverse_seq = primer_results[f'PRIMER_RIGHT_{i}_SEQUENCE']
                forward_start = primer_results[f'PRIMER_LEFT_{i}'][0]
                reverse_start = primer_results[f'PRIMER_RIGHT_{i}'][0]
                product_size = primer_results[f'PRIMER_PAIR_{i}_PRODUCT_SIZE']
                penalty = primer_results[f'PRIMER_PAIR_{i}_PENALTY']
                
                if add_t7_promoter:
                    forward_seq_t7 = self.t7_promoter + forward_seq
                    reverse_seq_t7 = self.t7_promoter + reverse_seq
                    
                    forward_tm = Tm_NN(forward_seq)
                    reverse_tm = Tm_NN(reverse_seq)
                    
                    primer_pair = PrimerPair(
                        forward_seq=forward_seq_t7,
                        reverse_seq=reverse_seq_t7,
                        forward_tm=forward_tm,
                        reverse_tm=reverse_tm,
                        product_size=product_size,
                        gc_content_f=self.calculate_gc_content(forward_seq),
                        gc_content_r=self.calculate_gc_content(reverse_seq),
                        forward_start=forward_start,
                        reverse_start=reverse_start,
                        penalty=penalty,
                        gene_target=gene_target
                    )
                    
                    primer_pair.core_forward_seq = forward_seq
                    primer_pair.core_reverse_seq = reverse_seq
                    primer_pair.has_t7_promoter = True
                    primer_pair.t7_promoter_seq = self.t7_promoter
                    
                else:
                    forward_tm = Tm_NN(forward_seq)
                    reverse_tm = Tm_NN(reverse_seq)
                    
                    primer_pair = PrimerPair(
                        forward_seq=forward_seq,
                        reverse_seq=reverse_seq,
                        forward_tm=forward_tm,
                        reverse_tm=reverse_tm,
                        product_size=product_size,
                        gc_content_f=self.calculate_gc_content(forward_seq),
                        gc_content_r=self.calculate_gc_content(reverse_seq),
                        forward_start=forward_start,
                        reverse_start=reverse_start,
                        penalty=penalty,
                        gene_target=gene_target
                    )
                    
                    primer_pair.has_t7_promoter = False
                
                primers.append(primer_pair)
            
            # Generate reverse orientation primers if requested
            if include_reverse_orientation and primers:
                try:
                    reverse_primers = self.generate_reverse_orientation_primers(sequence, primers, target_type)
                    primers.extend(reverse_primers)
                    
                    # Update primer results to include reverse orientation data
                    primer_results['REVERSE_ORIENTATION_GENERATED'] = True
                    primer_results['TARGET_TYPE'] = target_type
                    primer_results['TOTAL_PRIMER_PAIRS'] = len(primers)
                    
                except Exception as e:
                    st.warning(f"Could not generate reverse orientation primers: {e}")
            
            return primers, primer_results
            
        except Exception as e:
            st.error(f"Error in primer design: {e}")
            return [], {}
    
    def determine_orientation_strategy(self, target_info):
        """Auto-suggest orientation strategy based on target characteristics"""
        
        # Auto-enable both orientations for:
        if target_info.get('topology') == 'circular':
            return {'both_orientations': True, 'reason': 'Circular topology - no inherent directionality', 'priority': 'high'}
        
        if target_info.get('length', 0) > 10000:
            return {'both_orientations': True, 'reason': 'Large genome - multiple conserved regions accessible', 'priority': 'high'}
        
        if 'viral' in target_info.get('type', '').lower():
            return {'both_orientations': True, 'reason': 'Viral target - strand flexibility for robust detection', 'priority': 'medium'}
        
        if target_info.get('application') == 'antisense':
            return {'both_orientations': True, 'reason': 'Antisense detection required', 'priority': 'high'}
        
        if target_info.get('application') == 'method_development':
            return {'both_orientations': True, 'reason': 'Method development - comparative testing valuable', 'priority': 'medium'}
        
        # Auto-disable for specific applications
        if target_info.get('application') in ['splice_junction', 'snp', 'allele_specific', 'gene_expression']:
            return {'both_orientations': False, 'reason': 'Orientation-specific application', 'priority': 'high'}
        
        # Default: ask user
        return {'both_orientations': None, 'reason': 'User choice recommended', 'priority': 'low'}
    
    def generate_reverse_orientation_primers(self, sequence, primary_primers, target_type="circular"):
        """Generate primers for reverse orientation around circular or large targets"""
        reverse_orientation_primers = []
        
        for primer_pair in primary_primers:
            # For circular targets, swap and reverse complement
            if target_type == "circular":
                new_forward = self.reverse_complement(primer_pair.reverse_seq)
                new_reverse = self.reverse_complement(primer_pair.forward_seq)
                
                # Calculate new positions for circular target
                seq_len = len(sequence)
                new_forward_start = (seq_len - primer_pair.reverse_start) % seq_len
                new_reverse_start = (seq_len - primer_pair.forward_start) % seq_len
                
            else:
                # For linear targets, just reverse complement both primers
                new_forward = self.reverse_complement(primer_pair.forward_seq)
                new_reverse = self.reverse_complement(primer_pair.reverse_seq)
                
                # For linear targets, positions are calculated from the end
                seq_len = len(sequence)
                new_forward_start = seq_len - primer_pair.reverse_start - len(primer_pair.reverse_seq)
                new_reverse_start = seq_len - primer_pair.forward_start - len(primer_pair.forward_seq)
            
            # Recalculate properties for new orientation
            forward_tm = Tm_NN(new_forward)
            reverse_tm = Tm_NN(new_reverse)
            
            reverse_pair = PrimerPair(
                forward_seq=new_forward,
                reverse_seq=new_reverse,
                forward_tm=forward_tm,
                reverse_tm=reverse_tm,
                product_size=primer_pair.product_size,  # Same for circular
                gc_content_f=self.calculate_gc_content(new_forward),
                gc_content_r=self.calculate_gc_content(new_reverse),
                forward_start=new_forward_start,
                reverse_start=new_reverse_start,
                penalty=primer_pair.penalty,  # Keep same penalty
                gene_target=f"{primer_pair.gene_target} (Reverse Orientation)"
            )
            
            # Add orientation metadata
            reverse_pair.orientation = 'reverse'
            reverse_pair.original_orientation = 'forward'
            reverse_pair.target_type = target_type
            
            # Copy T7 promoter info if present
            if hasattr(primer_pair, 'has_t7_promoter') and primer_pair.has_t7_promoter:
                reverse_pair.has_t7_promoter = True
                reverse_pair.t7_promoter_seq = primer_pair.t7_promoter_seq
                reverse_pair.core_forward_seq = new_forward
                reverse_pair.core_reverse_seq = new_reverse
            
            reverse_orientation_primers.append(reverse_pair)
        
        return reverse_orientation_primers
    
    def compare_with_literature(self, primer_pairs, target_name="HLVd"):
        """Compare designed primers with known literature primers"""
        literature_db = {
            "HLVd": {
                "Eastwell_Nelson_2007": {
                    "forward": "ATACAACTCTTGAGCGCCGA",
                    "reverse": "CCACCGGGTAGTTTCCCAACT",
                    "reference": "Eastwell & Nelson (2007) Plant Disease"
                },
                "Verhoeven_2010": {
                    "forward": "GAGCGCCGAGACTCTTGAT",
                    "reverse": "CCACCGGGTAGTTTCCCAACT", 
                    "reference": "Verhoeven et al. (2010) Archives of Virology"
                }
            },
            "CEVd": {
                "Semancik_1997": {
                    "forward": "GAGCGCCGAGACTCTTGAT",
                    "reverse": "CCACCGGGTAGTTTCCCAACT",
                    "reference": "Semancik et al. (1997) Virology"
                }
            },
            "ASBVd": {
                "Di_Serio_2001": {
                    "forward": "GAGCGCCGAGACTCTTGAT",
                    "reverse": "CCACCGGGTAGTTTCCCAACT",
                    "reference": "Di Serio et al. (2001) Journal of Virology"
                }
            }
        }
        
        if target_name in literature_db:
            st.subheader(f"📚 Literature Comparison ({target_name})")
            
            for ref, primers in literature_db[target_name].items():
                st.write(f"**Reference:** {primers['reference']}")
                
                # Check for matches
                matches = []
                for i, pair in enumerate(primer_pairs, 1):
                    pair_matches = []
                    
                    # Check forward primer matches
                    if pair.forward_seq == primers['forward']:
                        pair_matches.append("Forward = Literature Forward")
                    if pair.reverse_seq == primers['reverse']:
                        pair_matches.append("Reverse = Literature Reverse")
                    
                    # Check reverse complement matches
                    if pair.reverse_seq == self.reverse_complement(primers['forward']):
                        pair_matches.append("Reverse = Literature Forward RC ⭐")
                    if pair.forward_seq == self.reverse_complement(primers['reverse']):
                        pair_matches.append("Forward = Literature Reverse RC ⭐")
                    
                    if pair_matches:
                        matches.append(f"Pair {i}: {', '.join(pair_matches)}")
                
                if matches:
                    st.success("✅ **Matches Found:**")
                    for match in matches:
                        st.write(f"  • {match}")
                else:
                    st.info("ℹ️ No exact matches found with literature primers")
                
                st.write("---")
        else:
            st.info(f"ℹ️ No literature database available for {target_name}")


class Primer3ComplementarityAnalyzer:
    """Extracts and visualizes complementarity data directly from Primer3 results"""
    
    def __init__(self):
        self.primer3_results = None
    
    def store_primer3_results(self, primer3_results):
        """Store the raw Primer3 results for analysis"""
        self.primer3_results = primer3_results
    
    def extract_complementarity_data(self, primer_index=0):
        """Extract complementarity data for a specific primer pair from Primer3 results"""
        if not self.primer3_results:
            return None
        
        try:
            # Extract Primer3 complementarity values
            data = {
                'forward_self_any': self.primer3_results.get(f'PRIMER_LEFT_{primer_index}_SELF_ANY_TH', 0.0),
                'forward_self_end': self.primer3_results.get(f'PRIMER_LEFT_{primer_index}_SELF_END_TH', 0.0),
                'reverse_self_any': self.primer3_results.get(f'PRIMER_RIGHT_{primer_index}_SELF_ANY_TH', 0.0),
                'reverse_self_end': self.primer3_results.get(f'PRIMER_RIGHT_{primer_index}_SELF_END_TH', 0.0),
                'pair_compl_any': self.primer3_results.get(f'PRIMER_PAIR_{primer_index}_COMPL_ANY_TH', 0.0),
                'pair_compl_end': self.primer3_results.get(f'PRIMER_PAIR_{primer_index}_COMPL_END_TH', 0.0),
                
                # Also get the non-thermodynamic scores (if available)
                'forward_self_any_score': self.primer3_results.get(f'PRIMER_LEFT_{primer_index}_SELF_ANY', 0.0),
                'forward_self_end_score': self.primer3_results.get(f'PRIMER_LEFT_{primer_index}_SELF_END', 0.0),
                'reverse_self_any_score': self.primer3_results.get(f'PRIMER_RIGHT_{primer_index}_SELF_ANY', 0.0),
                'reverse_self_end_score': self.primer3_results.get(f'PRIMER_RIGHT_{primer_index}_SELF_END', 0.0),
                'pair_compl_any_score': self.primer3_results.get(f'PRIMER_PAIR_{primer_index}_COMPL_ANY', 0.0),
                'pair_compl_end_score': self.primer3_results.get(f'PRIMER_PAIR_{primer_index}_COMPL_END', 0.0),
                
                # Get penalty score
                'penalty': self.primer3_results.get(f'PRIMER_PAIR_{primer_index}_PENALTY', 0.0)
            }
            
            return data
        except Exception as e:
            st.error(f"Error extracting Primer3 complementarity data: {e}")
            return None
    
    def create_complementarity_summary_chart(self, primers_data):
        """Create summary chart of complementarity across all primers"""
        if not primers_data:
            return None
        
        # Prepare data for plotting
        primer_numbers = []
        forward_self_any = []
        forward_self_end = []
        reverse_self_any = []
        reverse_self_end = []
        pair_compl_any = []
        pair_compl_end = []
        
        for i, data in enumerate(primers_data):
            if data:
                primer_numbers.append(i + 1)
                forward_self_any.append(data['forward_self_any'])
                forward_self_end.append(data['forward_self_end'])
                reverse_self_any.append(data['reverse_self_any'])
                reverse_self_end.append(data['reverse_self_end'])
                pair_compl_any.append(data['pair_compl_any'])
                pair_compl_end.append(data['pair_compl_end'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Self-Complementarity (Any Position)', 
                'Self-Complementarity (3\' End)',
                'Primer-Primer Complementarity (Any)', 
                'Primer-Primer Complementarity (3\' End)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Self-Any
        fig.add_trace(
            go.Bar(x=primer_numbers, y=forward_self_any, name='Forward', 
                   marker_color='lightblue', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=primer_numbers, y=reverse_self_any, name='Reverse', 
                   marker_color='lightcoral', opacity=0.7),
            row=1, col=1
        )
        
        # Self-End
        fig.add_trace(
            go.Bar(x=primer_numbers, y=forward_self_end, name='Forward', 
                   marker_color='blue', opacity=0.7, showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=primer_numbers, y=reverse_self_end, name='Reverse', 
                   marker_color='red', opacity=0.7, showlegend=False),
            row=1, col=2
        )
        
        # Pair-Any
        fig.add_trace(
            go.Bar(x=primer_numbers, y=pair_compl_any, name='Pair Compl Any', 
                   marker_color='green', opacity=0.7),
            row=2, col=1
        )
        
        # Pair-End
        fig.add_trace(
            go.Bar(x=primer_numbers, y=pair_compl_end, name='Pair Compl End', 
                   marker_color='darkgreen', opacity=0.7, showlegend=False),
            row=2, col=2
        )
        
        # Add threshold lines (typical Primer3 defaults)
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_hline(y=47.0, line_dash="dash", line_color="red", 
                             annotation_text="Typical threshold (47°C)", row=row, col=col)
        
        fig.update_layout(
            height=600,
            title_text="Primer3 Complementarity Analysis (ΔG in °C)",
            showlegend=True
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="ΔG (°C)", row=1, col=1)
        fig.update_yaxes(title_text="ΔG (°C)", row=1, col=2)
        fig.update_yaxes(title_text="ΔG (°C)", row=2, col=1)
        fig.update_yaxes(title_text="ΔG (°C)", row=2, col=2)
        
        # Update x-axes labels
        fig.update_xaxes(title_text="Primer Pair", row=2, col=1)
        fig.update_xaxes(title_text="Primer Pair", row=2, col=2)
        
        return fig
    
    def create_detailed_analysis_table(self, data, primer_index):
        """Create detailed table for a specific primer pair"""
        if not data:
            return None
        
        analysis_data = [
            {
                'Analysis Type': 'Forward Self-Any (Hairpins)',
                'ΔG (°C)': f"{data['forward_self_any']:.1f}",
                'Score': f"{data['forward_self_any_score']:.1f}",
                'Risk Level': self._assess_risk(data['forward_self_any'], 'self'),
                'Description': 'Internal complementarity in forward primer'
            },
            {
                'Analysis Type': 'Forward Self-End (3\' Dimers)',
                'ΔG (°C)': f"{data['forward_self_end']:.1f}",
                'Score': f"{data['forward_self_end_score']:.1f}",
                'Risk Level': self._assess_risk(data['forward_self_end'], 'end'),
                'Description': '3\' end complementarity in forward primer'
            },
            {
                'Analysis Type': 'Reverse Self-Any (Hairpins)',
                'ΔG (°C)': f"{data['reverse_self_any']:.1f}",
                'Score': f"{data['reverse_self_any_score']:.1f}",
                'Risk Level': self._assess_risk(data['reverse_self_any'], 'self'),
                'Description': 'Internal complementarity in reverse primer'
            },
            {
                'Analysis Type': 'Reverse Self-End (3\' Dimers)',
                'ΔG (°C)': f"{data['reverse_self_end']:.1f}",
                'Score': f"{data['reverse_self_end_score']:.1f}",
                'Risk Level': self._assess_risk(data['reverse_self_end'], 'end'),
                'Description': '3\' end complementarity in reverse primer'
            },
            {
                'Analysis Type': 'Primer-Primer Any',
                'ΔG (°C)': f"{data['pair_compl_any']:.1f}",
                'Score': f"{data['pair_compl_any_score']:.1f}",
                'Risk Level': self._assess_risk(data['pair_compl_any'], 'pair'),
                'Description': 'Complementarity between primer pair'
            },
            {
                'Analysis Type': 'Primer-Primer End',
                'ΔG (°C)': f"{data['pair_compl_end']:.1f}",
                'Score': f"{data['pair_compl_end_score']:.1f}",
                'Risk Level': self._assess_risk(data['pair_compl_end'], 'pair_end'),
                'Description': '3\' end complementarity between primers'
            }
        ]
        
        return pd.DataFrame(analysis_data)
    
    def _assess_risk(self, delta_g, analysis_type):
        """Assess risk level based on ΔG values and analysis type"""
        # These thresholds are based on typical Primer3 defaults
        if analysis_type == 'end' or analysis_type == 'pair_end':
            # 3' end complementarity is more critical
            if delta_g > 47.0:
                return "🔴 High Risk"
            elif delta_g > 35.0:
                return "🟡 Medium Risk"
            else:
                return "🟢 Low Risk"
        else:
            # General complementarity
            if delta_g > 47.0:
                return "🔴 High Risk"
            elif delta_g > 40.0:
                return "🟡 Medium Risk"
            else:
                return "🟢 Low Risk"


class EnhancedSpecificityAnalyzer:
    """Enhanced specificity testing with proper alignment and thermodynamic considerations"""
    
    def __init__(self, ncbi_connector):
        self.ncbi = ncbi_connector
        self.aligner = PairwiseAligner()
        self.aligner.match_score = 2
        self.aligner.mismatch_score = -1
        self.aligner.open_gap_score = -2
        self.aligner.extend_gap_score = -0.5
    
    def calculate_binding_specificity(self, primer_seq: str, target_seq: str, 
                                    temp: float = 60.0, salt: float = 50.0) -> Dict:
        """Calculate thermodynamic binding specificity"""
        try:
            from Bio.SeqUtils.MeltingTemp import Tm_NN
            
            # Find best alignment
            alignments = self.aligner.align(primer_seq, target_seq)
            best_alignment = max(alignments, key=lambda x: x.score)
            
            # Calculate identity and similarity
            matches = sum(1 for a, b in zip(str(best_alignment[0]), str(best_alignment[1])) if a == b and a != '-')
            identity = matches / len(primer_seq) * 100
            
            # Calculate melting temperature for best match region
            aligned_primer = str(best_alignment[0]).replace('-', '')
            if len(aligned_primer) >= 10:  # Minimum length for Tm calculation
                tm_aligned = Tm_NN(aligned_primer, Na=salt, Tris=10.0)
            else:
                tm_aligned = 0.0
            
            # Calculate binding probability at PCR temperature
            if tm_aligned > 0:
                binding_prob = 1.0 if tm_aligned > temp else np.exp(-(temp - tm_aligned) / 10.0)
            else:
                binding_prob = 0.0
            
            return {
                'identity': identity,
                'alignment_score': best_alignment.score,
                'tm_aligned': tm_aligned,
                'binding_probability': binding_prob,
                'is_specific': identity < 70 or binding_prob < 0.1,
                'alignment': str(best_alignment)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'identity': 0.0,
                'is_specific': True
            }
    
    def comprehensive_specificity_test(self, primer_pair: PrimerPair, 
                                     target_sequence: str, 
                                     comparison_organisms: List[str],
                                     target_organism: str = None,
                                     pcr_temp: float = 60.0) -> Dict:
        """Enhanced specificity testing with thermodynamic considerations"""
        
        results = {
            'forward_results': {},
            'reverse_results': {},
            'overall_specificity': True,
            'risk_organisms': []
        }
        
        # Extract primer sequences
        if hasattr(primer_pair, 'has_t7_promoter') and primer_pair.has_t7_promoter:
            forward_seq = primer_pair.core_forward_seq
            reverse_seq = primer_pair.core_reverse_seq
        else:
            forward_seq = primer_pair.forward_seq
            reverse_seq = primer_pair.reverse_seq
        
        # Filter out target organism from comparison list
        filtered_organisms = []
        if target_organism:
            target_organism_lower = target_organism.lower().strip()
            for organism in comparison_organisms:
                organism_lower = organism.lower().strip()
                # Skip if it's the same organism (exact match or genus match)
                if (organism_lower == target_organism_lower or 
                    organism_lower.split()[0] == target_organism_lower.split()[0]):
                    continue
                filtered_organisms.append(organism)
        else:
            filtered_organisms = comparison_organisms
        
        # Test against each organism (excluding target organism)
        for organism in filtered_organisms:
            try:
                # Fetch sequences for organism
                seq_ids = self.ncbi.search_sequences(f'"{organism}"[organism]', max_results=3)
                
                if not seq_ids:
                    continue
                
                organism_results = {'forward': [], 'reverse': []}
                
                for seq_id in seq_ids[:2]:  # Test top 2 sequences
                    comp_sequence = self.ncbi.fetch_sequence(seq_id)
                    if not comp_sequence or len(comp_sequence) < 100:
                        continue
                    
                    # Test forward primer
                    forward_result = self.calculate_binding_specificity(
                        forward_seq, comp_sequence, pcr_temp
                    )
                    organism_results['forward'].append(forward_result)
                    
                    # Test reverse primer
                    reverse_result = self.calculate_binding_specificity(
                        reverse_seq, comp_sequence, pcr_temp
                    )
                    organism_results['reverse'].append(reverse_result)
                
                # Determine worst-case scenario for this organism
                if organism_results['forward'] and organism_results['reverse']:
                    max_forward_risk = max(r.get('binding_probability', 0) for r in organism_results['forward'])
                    max_reverse_risk = max(r.get('binding_probability', 0) for r in organism_results['reverse'])
                    
                    # Both primers must bind for amplification
                    amplification_risk = max_forward_risk * max_reverse_risk
                    
                    if amplification_risk > 0.1:  # 10% threshold
                        results['risk_organisms'].append({
                            'organism': organism,
                            'amplification_risk': amplification_risk,
                            'forward_risk': max_forward_risk,
                            'reverse_risk': max_reverse_risk
                        })
                        results['overall_specificity'] = False
                
                results['forward_results'][organism] = organism_results['forward']
                results['reverse_results'][organism] = organism_results['reverse']
                
            except Exception as e:
                st.warning(f"Specificity test failed for {organism}: {e}")
                continue
        
        return results

class SequenceManager:
    """Manages sequence selection and analysis workflow"""
    
    def __init__(self, ncbi_connector):
        self.ncbi = ncbi_connector
    
    def fetch_organism_sequences(self, organism_name, max_sequences=10):
        """Fetch multiple sequences for an organism with metadata"""
        try:
            search_query = f'"{organism_name}"[organism]'
            seq_ids = self.ncbi.search_sequences(
                search_query, 
                database="nucleotide", 
                max_results=max_sequences
            )
            
            sequences = []
            for seq_id in seq_ids:
                seq_info = self.ncbi.fetch_sequence_info(seq_id)
                if seq_info and seq_info.get('sequence'):
                    # Clean sequence
                    clean_seq = re.sub(r'[^ATGCatgc]', '', seq_info['sequence'].upper())
                    
                    if len(clean_seq) >= 100:  # Minimum length filter
                        sequences.append({
                            'id': seq_id,
                            'description': seq_info.get('description', ''),
                            'organism': seq_info.get('organism', ''),
                            'length': len(clean_seq),
                            'sequence': clean_seq[:10000],  # Limit to 10kb for performance
                            'full_length': len(clean_seq)
                        })
            
            return sequences
            
        except Exception as e:
            st.error(f"Error fetching sequences: {e}")
            return []
    
    def fetch_complete_genomes(self, organism_name, max_sequences=10):
        """Fetch complete genomes/sequences for conservation analysis"""
        try:
            # Search for complete genomes and complete sequences only
            complete_genome_query = f'"{organism_name}"[organism] AND (complete genome[title] OR complete sequence[title] OR whole genome[title] OR chromosome[title] OR genome[title])'
            
            seq_ids = self.ncbi.search_sequences(
                complete_genome_query, 
                database="nucleotide", 
                max_results=max_sequences * 2  # Get more results to filter
            )
            
            sequences = []
            for seq_id in seq_ids:
                seq_info = self.ncbi.fetch_sequence_info(seq_id)
                if seq_info and seq_info.get('sequence'):
                    description = seq_info.get('description', '').lower()
                    
                    # Filter for complete genomes/sequences only
                    is_complete = any(keyword in description for keyword in [
                        'complete genome', 'complete sequence', 'whole genome', 
                        'chromosome', 'genome sequence', 'complete'
                    ])
                    
                    # Also check sequence length - complete genomes are typically much longer
                    clean_seq = re.sub(r'[^ATGCatgc]', '', seq_info['sequence'].upper())
                    is_long_enough = len(clean_seq) >= 100000  # At least 100kb for complete genomes
                    
                    if is_complete and is_long_enough:
                        sequences.append({
                            'id': seq_id,
                            'description': seq_info.get('description', ''),
                            'organism': seq_info.get('organism', ''),
                            'length': len(clean_seq),
                            'sequence': clean_seq[:50000],  # Use larger sample for conservation analysis
                            'full_length': len(clean_seq),
                            'type': 'complete_genome'
                        })
                        
                        if len(sequences) >= max_sequences:
                            break
            
            # If we didn't find enough complete genomes, try a broader search but still filter
            if len(sequences) < max_sequences:
                st.warning(f"Found only {len(sequences)} complete genomes. Searching for additional complete sequences...")
                
                # Broader search for complete sequences
                broader_query = f'"{organism_name}"[organism] AND complete[title]'
                additional_ids = self.ncbi.search_sequences(
                    broader_query, 
                    database="nucleotide", 
                    max_results=max_sequences
                )
                
                for seq_id in additional_ids:
                    if len(sequences) >= max_sequences:
                        break
                        
                    seq_info = self.ncbi.fetch_sequence_info(seq_id)
                    if seq_info and seq_info.get('sequence'):
                        description = seq_info.get('description', '').lower()
                        clean_seq = re.sub(r'[^ATGCatgc]', '', seq_info['sequence'].upper())
                        
                        # More lenient filtering for complete sequences
                        is_complete = 'complete' in description and len(clean_seq) >= 10000  # At least 10kb
                        
                        if is_complete and not any(seq['id'] == seq_id for seq in sequences):
                            sequences.append({
                                'id': seq_id,
                                'description': seq_info.get('description', ''),
                                'organism': seq_info.get('organism', ''),
                                'length': len(clean_seq),
                                'sequence': clean_seq[:50000],
                                'full_length': len(clean_seq),
                                'type': 'complete_sequence'
                            })
            
            return sequences
            
        except Exception as e:
            st.error(f"Error fetching complete genomes: {e}")
            return []

class OptimizedSequenceManager:
    """Memory-optimized sequence manager with caching and size limits"""
    
    def __init__(self, ncbi_connector, max_sequence_size: int = 50000):
        self.ncbi = ncbi_connector
        self.max_sequence_size = max_sequence_size
        self.sequence_cache = {}
    
    def fetch_organism_sequences_optimized(self, organism_name: str, max_sequences: int = 10) -> List[Dict]:
        """Memory-optimized sequence fetching with caching"""
        
        cache_key = f"{organism_name}_{max_sequences}"
        if cache_key in self.sequence_cache:
            return self.sequence_cache[cache_key]
        
        sequences = []
        
        try:
            # Use more specific search to reduce irrelevant results
            search_query = f'"{organism_name}"[organism] AND (complete genome[title] OR complete sequence[title])'
            seq_ids = self.ncbi.search_sequences(search_query, max_results=max_sequences * 2)
            
            # Process sequences in batches to manage memory
            batch_size = 3
            for i in range(0, min(len(seq_ids), max_sequences), batch_size):
                batch_ids = seq_ids[i:i + batch_size]
                
                for seq_id in batch_ids:
                    # Fetch metadata first to filter by size
                    seq_info = self.ncbi.fetch_sequence_info(seq_id)
                    if not seq_info:
                        continue
                    
                    sequence_length = seq_info.get('length', 0)
                    
                    # Skip sequences that are too large or too small
                    if sequence_length < 1000 or sequence_length > 10000000:
                        continue
                    
                    # Fetch and truncate sequence if needed
                    full_sequence = self.ncbi.fetch_sequence(seq_id)
                    if not full_sequence:
                        continue
                    
                    # Clean and truncate sequence
                    clean_seq = re.sub(r'[^ATGCatgc]', '', full_sequence.upper())
                    
                    if len(clean_seq) > self.max_sequence_size:
                        # Take middle portion for better gene representation
                        start = (len(clean_seq) - self.max_sequence_size) // 2
                        clean_seq = clean_seq[start:start + self.max_sequence_size]
                        seq_info['truncated'] = True
                        seq_info['original_length'] = len(full_sequence)
                    
                    sequences.append({
                        'id': seq_id,
                        'description': seq_info.get('description', ''),
                        'organism': seq_info.get('organism', ''),
                        'length': len(clean_seq),
                        'sequence': clean_seq,
                        'metadata': seq_info
                    })
                    
                    if len(sequences) >= max_sequences:
                        break
                
                # Memory cleanup between batches
                if len(sequences) >= max_sequences:
                    break
        
        except Exception as e:
            st.error(f"Error in optimized sequence fetching: {e}")
            return []
        
        # Cache results but limit cache size
        if len(self.sequence_cache) > 10:  # Max 10 cached queries
            oldest_key = next(iter(self.sequence_cache))
            del self.sequence_cache[oldest_key]
        
        self.sequence_cache[cache_key] = sequences
        return sequences

class OptimizedSessionManager:
    """Optimized session state management with compression and cleanup"""
    
    @staticmethod
    def compress_session_data(data):
        """Compress large session data"""
        if isinstance(data, list) and len(data) > 10:
            # Compress large lists (like primer results)
            pickled = pickle.dumps(data)
            compressed = gzip.compress(pickled)
            return {
                '_compressed': True,
                '_data': compressed,
                '_size': len(data),
                '_timestamp': datetime.now().isoformat()
            }
        return data
    
    @staticmethod
    def decompress_session_data(data):
        """Decompress session data"""
        if isinstance(data, dict) and data.get('_compressed'):
            try:
                compressed = data['_data']
                pickled = gzip.decompress(compressed)
                return pickle.loads(pickled)
            except:
                return []
        return data
    
    @staticmethod
    def cleanup_session_state():
        """Remove stale session data"""
        keys_to_remove = []
        current_time = datetime.now()
        
        for key, value in st.session_state.items():
            # Remove data older than 1 hour
            if isinstance(value, dict) and '_timestamp' in value:
                timestamp = datetime.fromisoformat(value['_timestamp'])
                if current_time - timestamp > timedelta(hours=1):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
    
    @staticmethod
    def store_primers_optimized(primers: List[PrimerPair]):
        """Store primers with compression"""
        compressed_primers = OptimizedSessionManager.compress_session_data(primers)
        st.session_state.primers_designed = compressed_primers
    
    @staticmethod
    def get_primers_optimized() -> List[PrimerPair]:
        """Retrieve primers with decompression"""
        compressed_primers = st.session_state.get('primers_designed', [])
        return OptimizedSessionManager.decompress_session_data(compressed_primers)

class TargetSpecificFilterManager:
    """Manages target-specific filtering for expression and beneficial species"""
    
    def __init__(self):
        self.expression_data = self._load_expression_database()
        self.essential_genes = self._load_essential_genes()
        self.host_range_data = self._load_host_range_data()
    
    def _load_expression_database(self) -> Dict:
        """Load tissue/condition-specific expression data"""
        # This would load from a curated database
        return {
            'Diaphorina citri': {
                'gut_specific': ['SLC26A', 'ABCA3', 'ABCG1', 'digestive_enzymes'],
                'infection_responsive': ['immune_genes', 'stress_response'],
                'essential_housekeeping': ['ACT1', 'TUB1', 'EF1A', 'RPL32']
            }
        }
    
    def _load_essential_genes(self) -> Dict:
        """Load essential gene annotations"""
        return {
            'universal_essential': ['ACT1', 'TUB1', 'EF1A', 'RPB2', 'RPL32'],
            'insect_essential': ['chitin_synthase', 'ecdysone_receptor', 'juvenile_hormone'],
            'pathogen_essential': ['cell_wall_synthesis', 'DNA_replication', 'protein_synthesis']
        }
    
    def filter_genes_by_expression(self, organism: str, gene_list: List[str], 
                                 filter_type: str) -> List[str]:
        """Filter genes by expression specificity"""
        if organism not in self.expression_data:
            return gene_list
        
        organism_data = self.expression_data[organism]
        
        if filter_type == 'gut_specific':
            specific_genes = organism_data.get('gut_specific', [])
            return [gene for gene in gene_list if any(spec in gene.lower() for spec in specific_genes)]
        
        elif filter_type == 'infection_responsive':
            responsive_genes = organism_data.get('infection_responsive', [])
            return [gene for gene in gene_list if any(resp in gene.lower() for resp in responsive_genes)]
        
        elif filter_type == 'essential_only':
            essential = organism_data.get('essential_housekeeping', [])
            return [gene for gene in gene_list if any(ess in gene for ess in essential)]
        
        return gene_list
    
    def check_beneficial_species_presence(self, gene_name: str, 
                                        beneficial_species: List[str]) -> Dict:
        """Check if gene is present in beneficial species"""
        results = {}
        
        for species in beneficial_species:
            # This would query databases or perform BLAST searches
            # Simplified implementation:
            if 'universal' in gene_name.lower() or 'conserved' in gene_name.lower():
                results[species] = {'present': True, 'confidence': 'high'}
            else:
                results[species] = {'present': False, 'confidence': 'medium'}
        
        return results

class siRNAOptimizer:
    """Optimizes siRNA sequences within dsRNA primers"""
    
    def __init__(self):
        self.sirna_length = 21
        self.guide_rules = {
            'asymmetry_rule': True,
            'gc_content_range': (30, 52),
            'avoid_poly_runs': 4,
            'thermodynamic_asymmetry': True
        }
    
    def calculate_sirna_score(self, sirna_seq: str) -> Dict:
        """Calculate siRNA efficiency score based on established rules"""
        score = 0
        details = {}
        
        # Reynolds rules (simplified)
        if len(sirna_seq) >= 19:
            # Position-specific scoring
            if sirna_seq[18] in ['A', 'U']:  # Position 19 (1-indexed)
                score += 3
                details['pos19_AU'] = True
            
            if sirna_seq[2] in ['A', 'U']:  # Position 3
                score += 2
                details['pos3_AU'] = True
            
            if sirna_seq[9] in ['A', 'U']:  # Position 10
                score += 1
                details['pos10_AU'] = True
            
            # GC content
            gc_content = (sirna_seq.count('G') + sirna_seq.count('C')) / len(sirna_seq) * 100
            if 30 <= gc_content <= 52:
                score += 2
                details['optimal_gc'] = True
            
            details['gc_content'] = gc_content
        
        # Thermodynamic asymmetry (simplified)
        if len(sirna_seq) >= 21:
            sense_5_end = sirna_seq[:4]
            antisense_5_end = sirna_seq[-4:]
            
            # Count AU at 5' ends
            sense_au = sense_5_end.count('A') + sense_5_end.count('U')
            antisense_au = antisense_5_end.count('A') + antisense_5_end.count('U')
            
            if sense_au > antisense_au:
                score += 1
                details['asymmetry'] = True
        
        details['total_score'] = score
        details['efficiency_prediction'] = 'High' if score >= 6 else 'Medium' if score >= 3 else 'Low'
        
        return details
    
    def find_optimal_sirnas(self, dsrna_sequence: str, top_n: int = 5) -> List[Dict]:
        """Find optimal siRNA sequences within dsRNA"""
        sirnas = []
        
        # Scan through sequence with sliding window
        for i in range(len(dsrna_sequence) - self.sirna_length + 1):
            sirna_seq = dsrna_sequence[i:i + self.sirna_length]
            
            # Skip sequences with poly runs
            if any(base * 4 in sirna_seq for base in 'ATGC'):
                continue
            
            score_data = self.calculate_sirna_score(sirna_seq)
            
            sirnas.append({
                'sequence': sirna_seq,
                'position': i,
                'score': score_data['total_score'],
                'efficiency': score_data['efficiency_prediction'],
                'details': score_data
            })
        
        # Sort by score and return top candidates
        sirnas.sort(key=lambda x: x['score'], reverse=True)
        return sirnas[:top_n]

# Streamlit App Functions
def init_session_state():
    """Initialize session state variables"""
    session_vars = {
        'primers_designed': [],
        'current_sequence': "",
        'sequence_info': {},
        'search_results': None,
        'database_used': None,
        'comprehensive_analysis_results': None,
        't7_results': None,
        't7_dsrna_enabled': False,
        't7_settings': {},
        # ADD THESE NEW VARIABLES:
        'found_sequences': [],
        'target_organism': '',
        'conserved_regions': [],
        'conservation_sequences': [],
        'specificity_results': {},
        'analysis_metadata': {},
        # GENE TARGET VARIABLES:
        'selected_gene_targets': {},
        'gene_target_stats': {},
        # ADDITIONAL PERSISTENCE VARIABLES:
        'session_initialized': True,
        'last_activity': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
    
    # Ensure critical variables are never None
    if st.session_state.get('primers_designed') is None:
        st.session_state.primers_designed = []
    if st.session_state.get('sequence_info') is None:
        st.session_state.sequence_info = {}
    if st.session_state.get('specificity_results') is None:
        st.session_state.specificity_results = {}
    
    # Cleanup stale session data
    OptimizedSessionManager.cleanup_session_state()

def check_session_state_validity():
    """Check if session state has valid primer data"""
    try:
        # Get primers data, handling both compressed and uncompressed formats
        primers_data = st.session_state.get('primers_designed', [])
        
        # Handle compressed data from OptimizedSessionManager
        if isinstance(primers_data, dict) and primers_data.get('_compressed'):
            primers = OptimizedSessionManager.decompress_session_data(primers_data)
            primer_count = len(primers) if primers else 0
        elif isinstance(primers_data, list):
            primers = primers_data
            primer_count = len(primers)
        else:
            primers = []
            primer_count = 0
        
        # Check sequence data
        sequence = st.session_state.get('current_sequence', '')
        sequence_length = len(sequence) if sequence else 0
        
        # Check sequence info
        sequence_info = st.session_state.get('sequence_info', {})
        has_seq_info = bool(sequence_info)
        
        return {
            'has_primers': primer_count > 0,
            'has_sequence': sequence_length > 0,
            'has_seq_info': has_seq_info,
            'primer_count': primer_count,
            'sequence_length': sequence_length
        }
    except Exception as e:
        return {
            'has_primers': False,
            'has_sequence': False,
            'has_seq_info': False,
            'primer_count': 0,
            'sequence_length': 0
        }


def create_primer_visualization(primers: List[PrimerPair]):
    """Create interactive visualizations for primer pairs"""
    if not primers:
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Melting Temperatures', 'GC Content Distribution', 
                           'Product Sizes', 'Penalty Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        primer_nums = list(range(1, len(primers) + 1))
        forward_tms = [p.forward_tm for p in primers]
        reverse_tms = [p.reverse_tm for p in primers]
        forward_gcs = [p.gc_content_f for p in primers]
        reverse_gcs = [p.gc_content_r for p in primers]
        product_sizes = [p.product_size for p in primers]
        penalties = [p.penalty for p in primers]
        
        fig.add_trace(
            go.Scatter(x=primer_nums, y=forward_tms, name='Forward Tm', 
                      line=dict(color='blue'), mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=primer_nums, y=reverse_tms, name='Reverse Tm', 
                      line=dict(color='red'), mode='lines+markers'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=primer_nums, y=forward_gcs, name='Forward GC%', 
                   marker_color='lightblue', opacity=0.7),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=primer_nums, y=reverse_gcs, name='Reverse GC%', 
                   marker_color='lightcoral', opacity=0.7),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=primer_nums, y=product_sizes, name='Product Size', 
                      line=dict(color='green'), mode='lines+markers'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=primer_nums, y=penalties, name='Penalty Score', 
                   marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Primer Pair Analysis")
        fig.update_xaxes(title_text="Primer Pair Number")
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="GC Content (%)", row=1, col=2)
        fig.update_yaxes(title_text="Product Size (bp)", row=2, col=1)
        fig.update_yaxes(title_text="Penalty Score", row=2, col=2)
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def create_enhanced_sequence_diagram(sequence: str, primers: List[PrimerPair], selected_primer: int = 0):
    """Create an enhanced sequence diagram with better scaling and interactivity"""
    if not primers or selected_primer >= len(primers) or not sequence:
        return None
    
    try:
        primer = primers[selected_primer]
        seq_len = len(sequence)
        
        # Calculate appropriate padding and scaling
        padding = max(50, seq_len * 0.05)  # 5% padding or minimum 50bp
        x_range = [-padding, seq_len + padding]
        
        fig = go.Figure()
        
        # Main sequence line with better scaling
        fig.add_shape(
            type="rect",
            x0=0, y0=0.4, x1=seq_len, y1=0.6,
            fillcolor="lightgray",
            line=dict(color="black", width=2),
        )
        
        # Determine primer sequences and lengths
        if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
            forward_seq = primer.core_forward_seq
            reverse_seq = primer.core_reverse_seq
            forward_len = len(forward_seq)
            reverse_len = len(reverse_seq)
            t7_promoter = primer.t7_promoter_seq
        else:
            forward_seq = primer.forward_seq
            reverse_seq = primer.reverse_seq
            forward_len = len(forward_seq)
            reverse_len = len(reverse_seq)
            t7_promoter = None
        
        # Forward primer
        fig.add_shape(
            type="rect",
            x0=primer.forward_start, y0=0.6, 
            x1=primer.forward_start + forward_len, y1=0.8,
            fillcolor="blue",
            line=dict(color="darkblue", width=2),
        )
        
        # Reverse primer
        fig.add_shape(
            type="rect",
            x0=primer.reverse_start - reverse_len + 1, y0=0.2,
            x1=primer.reverse_start + 1, y1=0.4,
            fillcolor="red",
            line=dict(color="darkred", width=2),
        )
        
        # T7 promoter visualization if present
        if t7_promoter:
            t7_len = len(t7_promoter)
            # T7 promoter for forward primer
            fig.add_shape(
                type="rect",
                x0=primer.forward_start - t7_len, y0=0.6, 
                x1=primer.forward_start, y1=0.8,
                fillcolor="lightblue",
                line=dict(color="blue", width=1, dash="dash"),
            )
            # T7 promoter for reverse primer
            fig.add_shape(
                type="rect",
                x0=primer.reverse_start + 1, y0=0.2,
                x1=primer.reverse_start + 1 + t7_len, y1=0.4,
                fillcolor="lightcoral",
                line=dict(color="red", width=1, dash="dash"),
            )
        
        # Product region
        product_start = primer.forward_start + forward_len
        product_end = primer.reverse_start - reverse_len + 1
        if product_end > product_start:
            fig.add_shape(
                type="rect",
                x0=product_start, y0=0.45, 
                x1=product_end, y1=0.55,
                fillcolor="green",
                line=dict(color="darkgreen", width=1),
                opacity=0.3
            )
        
        # Position markers at regular intervals
        marker_interval = max(100, seq_len // 10)  # 10 markers or every 100bp
        for pos in range(0, seq_len + 1, marker_interval):
            fig.add_vline(x=pos, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_annotation(
                x=pos, y=0.35,
                text=f"{pos:,}",
                showarrow=False,
                font=dict(size=10, color="gray"),
                yshift=-10
            )
        
        # Enhanced annotations with hover information
        fig.add_annotation(
            x=primer.forward_start + forward_len/2, y=0.7,
            text=f"Forward Primer<br>Tm: {primer.forward_tm:.1f}°C<br>Length: {forward_len} bp",
            showarrow=True, arrowhead=2, arrowcolor="blue",
            font=dict(size=12, color="darkblue"),
            bgcolor="white",
            bordercolor="blue",
            borderwidth=1
        )
        
        fig.add_annotation(
            x=primer.reverse_start - reverse_len/2, y=0.3,
            text=f"Reverse Primer<br>Tm: {primer.reverse_tm:.1f}°C<br>Length: {reverse_len} bp",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(size=12, color="darkred"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1
        )
        
        # Product size annotation
        if product_end > product_start:
            product_center = (product_start + product_end) / 2
            fig.add_annotation(
                x=product_center, y=0.5,
                text=f"Product: {primer.product_size} bp",
                showarrow=False,
                font=dict(size=12, color="darkgreen", weight="bold"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
        
        # T7 promoter annotations
        if t7_promoter:
            fig.add_annotation(
                x=primer.forward_start - t7_len/2, y=0.7,
                text="T7 Promoter",
                showarrow=False,
                font=dict(size=10, color="blue"),
                bgcolor="lightblue",
                bordercolor="blue"
            )
            fig.add_annotation(
                x=primer.reverse_start + 1 + t7_len/2, y=0.3,
                text="T7 Promoter",
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="lightcoral",
                bordercolor="red"
            )
        
        fig.update_layout(
            title=f"Enhanced Primer Binding Sites - Pair {selected_primer + 1}",
            xaxis_title="Sequence Position (bp)",
            yaxis=dict(range=[0, 1], showticklabels=False),
            height=400,
            showlegend=False,
            xaxis=dict(range=x_range),
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating enhanced sequence diagram: {e}")
        return None

def create_multi_primer_comparison(sequence: str, primers: List[PrimerPair], max_primers: int = 5):
    """Create a comparison view of multiple primer pairs on the same sequence"""
    if not primers or not sequence:
        return None
    
    try:
        seq_len = len(sequence)
        padding = max(50, seq_len * 0.05)
        x_range = [-padding, seq_len + padding]
        
        fig = go.Figure()
        
        # Main sequence line
        fig.add_shape(
            type="rect",
            x0=0, y0=0.4, x1=seq_len, y1=0.6,
            fillcolor="lightgray",
            line=dict(color="black", width=2),
        )
        
        # Color palette for different primer pairs
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plot each primer pair
        for i in range(min(max_primers, len(primers))):
            primer = primers[i]
            color = colors[i % len(colors)]
            
            # Determine primer sequences and lengths
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                forward_len = len(primer.core_forward_seq)
                reverse_len = len(primer.core_reverse_seq)
            else:
                forward_len = len(primer.forward_seq)
                reverse_len = len(primer.reverse_seq)
            
            # Calculate y position for this primer pair
            y_offset = 0.1 * i
            forward_y = 0.6 + y_offset
            reverse_y = 0.4 - y_offset
            
            # Forward primer
            fig.add_shape(
                type="rect",
                x0=primer.forward_start, y0=forward_y, 
                x1=primer.forward_start + forward_len, y1=forward_y + 0.08,
                fillcolor=color,
                line=dict(color=color, width=1),
                opacity=0.7
            )
            
            # Reverse primer
            fig.add_shape(
                type="rect",
                x0=primer.reverse_start - reverse_len + 1, y0=reverse_y - 0.08,
                x1=primer.reverse_start + 1, y1=reverse_y,
                fillcolor=color,
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.7
            )
            
            # Product region
            product_start = primer.forward_start + forward_len
            product_end = primer.reverse_start - reverse_len + 1
            if product_end > product_start:
                fig.add_shape(
                    type="rect",
                    x0=product_start, y0=0.45, 
                    x1=product_end, y1=0.55,
                    fillcolor=color,
                    line=dict(color=color, width=1),
                    opacity=0.2
                )
            
            # Legend entry
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"Pair {i+1} (Product: {primer.product_size} bp)",
                showlegend=True
            ))
        
        # Position markers
        marker_interval = max(100, seq_len // 10)
        for pos in range(0, seq_len + 1, marker_interval):
            fig.add_vline(x=pos, line_dash="dot", line_color="gray", opacity=0.3)
            if pos % (marker_interval * 2) == 0:  # Show every other marker
                fig.add_annotation(
                    x=pos, y=0.35,
                    text=f"{pos:,}",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    yshift=-10
                )
        
        fig.update_layout(
            title=f"Multi-Primer Comparison ({min(max_primers, len(primers))} pairs)",
            xaxis_title="Sequence Position (bp)",
            yaxis=dict(range=[0, 1], showticklabels=False),
            height=500,
            showlegend=True,
            xaxis=dict(range=x_range),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating multi-primer comparison: {e}")
        return None

def create_primer_coverage_heatmap(sequence: str, primers: List[PrimerPair], window_size: int = 100):
    """Create a heatmap showing primer coverage density across the sequence"""
    if not primers or not sequence:
        return None
    
    try:
        seq_len = len(sequence)
        num_windows = seq_len // window_size + (1 if seq_len % window_size > 0 else 0)
        
        # Initialize coverage array
        coverage = [0] * num_windows
        
        # Calculate coverage for each window
        for primer in primers:
            # Determine primer lengths
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                forward_len = len(primer.core_forward_seq)
                reverse_len = len(primer.core_reverse_seq)
            else:
                forward_len = len(primer.forward_seq)
                reverse_len = len(primer.reverse_seq)
            
            # Forward primer coverage
            forward_start_window = primer.forward_start // window_size
            forward_end_window = (primer.forward_start + forward_len - 1) // window_size
            
            for window in range(forward_start_window, min(forward_end_window + 1, num_windows)):
                if 0 <= window < num_windows:
                    coverage[window] += 1
            
            # Reverse primer coverage
            reverse_start_window = (primer.reverse_start - reverse_len + 1) // window_size
            reverse_end_window = primer.reverse_start // window_size
            
            for window in range(reverse_start_window, min(reverse_end_window + 1, num_windows)):
                if 0 <= window < num_windows:
                    coverage[window] += 1
        
        # Create heatmap data
        window_positions = [i * window_size for i in range(num_windows)]
        max_coverage = max(coverage) if coverage else 1
        
        # Create the heatmap
        fig = go.Figure()
        
        # Add coverage bars
        fig.add_trace(go.Bar(
            x=window_positions,
            y=coverage,
            marker=dict(
                color=coverage,
                colorscale='Reds',
                cmin=0,
                cmax=max_coverage,
                colorbar=dict(title="Primer Count")
            ),
            name="Primer Coverage",
            hovertemplate="Position: %{x}-%{xend}<br>Primers: %{y}<extra></extra>"
        ))
        
        # Add sequence length indicator
        fig.add_vline(x=seq_len, line_dash="dash", line_color="black", 
                     annotation_text=f"Sequence End ({seq_len:,} bp)")
        
        # Add coverage statistics
        avg_coverage = sum(coverage) / len(coverage) if coverage else 0
        max_coverage_pos = window_positions[coverage.index(max_coverage)] if coverage else 0
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"Avg Coverage: {avg_coverage:.1f}<br>Max Coverage: {max_coverage} at {max_coverage_pos:,} bp",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"Primer Coverage Heatmap (Window Size: {window_size} bp)",
            xaxis_title="Sequence Position (bp)",
            yaxis_title="Number of Primers",
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating coverage heatmap: {e}")
        return None

def display_primer3_complementarity_analysis(primer_pair, selected_primer_index, primer3_results=None):
    """Display complementarity analysis using Primer3's actual calculations"""
    st.subheader("Primer3 Complementarity Analysis")
    st.info("Analysis using Primer3's internal complementarity calculations and thermodynamic parameters")
    
    if not primer3_results:
        st.warning("Primer3 results not available for complementarity analysis")
        return
    
    analyzer = Primer3ComplementarityAnalyzer()
    analyzer.store_primer3_results(primer3_results)
    
    # Extract data for selected primer
    comp_data = analyzer.extract_complementarity_data(selected_primer_index)
    
    if not comp_data:
        st.error("Could not extract complementarity data for this primer pair")
        return
    
    # Display overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Forward Self-Any", 
            f"{comp_data['forward_self_any']:.1f}°C",
            help="Maximum ΔG for internal complementarity in forward primer"
        )
        st.metric(
            "Forward Self-End", 
            f"{comp_data['forward_self_end']:.1f}°C",
            help="ΔG for 3' end complementarity in forward primer"
        )
    
    with col2:
        st.metric(
            "Reverse Self-Any", 
            f"{comp_data['reverse_self_any']:.1f}°C",
            help="Maximum ΔG for internal complementarity in reverse primer"
        )
        st.metric(
            "Reverse Self-End", 
            f"{comp_data['reverse_self_end']:.1f}°C",
            help="ΔG for 3' end complementarity in reverse primer"
        )
    
    with col3:
        st.metric(
            "Primer-Primer Any", 
            f"{comp_data['pair_compl_any']:.1f}°C",
            help="Maximum ΔG for complementarity between primers"
        )
        st.metric(
            "Primer-Primer End", 
            f"{comp_data['pair_compl_end']:.1f}°C",
            help="ΔG for 3' end complementarity between primers"
        )
    
    # Risk assessment
    st.subheader("Risk Assessment")
    
    max_self_any = max(comp_data['forward_self_any'], comp_data['reverse_self_any'])
    max_self_end = max(comp_data['forward_self_end'], comp_data['reverse_self_end'])
    max_pair = max(comp_data['pair_compl_any'], comp_data['pair_compl_end'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if max_self_any > 47.0:
            st.error(f"🔴 Self-Hairpins: High Risk ({max_self_any:.1f}°C)")
        elif max_self_any > 40.0:
            st.warning(f"🟡 Self-Hairpins: Medium Risk ({max_self_any:.1f}°C)")
        else:
            st.success(f"🟢 Self-Hairpins: Low Risk ({max_self_any:.1f}°C)")
    
    with col2:
        if max_self_end > 47.0:
            st.error(f"🔴 3' Self-Dimers: High Risk ({max_self_end:.1f}°C)")
        elif max_self_end > 35.0:
            st.warning(f"🟡 3' Self-Dimers: Medium Risk ({max_self_end:.1f}°C)")
        else:
            st.success(f"🟢 3' Self-Dimers: Low Risk ({max_self_end:.1f}°C)")
    
    with col3:
        if max_pair > 47.0:
            st.error(f"🔴 Primer Dimers: High Risk ({max_pair:.1f}°C)")
        elif max_pair > 40.0:
            st.warning(f"🟡 Primer Dimers: Medium Risk ({max_pair:.1f}°C)")
        else:
            st.success(f"🟢 Primer Dimers: Low Risk ({max_pair:.1f}°C)")
    
    # Detailed analysis table
    if st.checkbox("Show detailed Primer3 analysis", key=f"primer3_detail_{selected_primer_index}"):
        detail_table = analyzer.create_detailed_analysis_table(comp_data, selected_primer_index)
        if detail_table is not None:
            st.dataframe(detail_table, use_container_width=True)
    
    # Overall assessment
    overall_max = max(max_self_any, max_self_end, max_pair)
    
    if overall_max > 47.0:
        st.error("🚫 **Overall Assessment: HIGH RISK** - Significant complementarity detected")
        st.write("**Recommendation:** Consider using a different primer pair or adjusting design parameters")
    elif overall_max > 40.0:
        st.warning("⚠️ **Overall Assessment: MEDIUM RISK** - Some complementarity present")
        st.write("**Recommendation:** Monitor for primer artifacts during PCR, consider optimization")
    else:
        st.success("✅ **Overall Assessment: LOW RISK** - Minimal complementarity detected")
        st.write("**Recommendation:** Good primer pair for PCR applications")
    
    # Explanation of metrics
    with st.expander("Understanding Primer3 Complementarity Metrics"):
        st.markdown("""
        **ΔG Values (°C):** Lower (more negative) values indicate stronger binding
        
        **Self-Any:** Checks for internal complementarity that could form hairpins anywhere in the primer
        
        **Self-End:** Specifically examines complementarity at the 3' end where DNA polymerase extends
        
        **Primer-Primer Any:** Complementarity between forward and reverse primers at any position
        
        **Primer-Primer End:** 3' end complementarity between primers (most critical for dimer formation)
        
        **Risk Thresholds:**
        - 🟢 < 40°C: Low risk of artifacts
        - 🟡 40-47°C: Medium risk, monitor carefully  
        - 🔴 > 47°C: High risk, likely to cause PCR artifacts
        
        **Note:** These are the actual values calculated by Primer3 during primer design and directly contribute to the penalty scores you see.
        """)

def export_to_excel(primers: List[PrimerPair]) -> bytes:
    """Export primer results to Excel format"""
    try:
        data = []
        for i, primer in enumerate(primers):
            # Handle existing primer pairs that don't have gene_target attribute
            gene_target = getattr(primer, 'gene_target', 'Standard Design')
            
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                data.append({
                    'Primer_Pair': i + 1,
                    'Gene_Target': gene_target,
                    'Forward_T7_Sequence': primer.forward_seq,
                    'Reverse_T7_Sequence': primer.reverse_seq,
                    'Forward_Core_Sequence': primer.core_forward_seq,
                    'Reverse_Core_Sequence': primer.core_reverse_seq,
                    'Core_Forward_Tm': round(primer.forward_tm, 2),
                    'Core_Reverse_Tm': round(primer.reverse_tm, 2),
                    'dsRNA_Size': primer.product_size,
                    'Core_Forward_GC%': round(primer.gc_content_f, 2),
                    'Core_Reverse_GC%': round(primer.gc_content_r, 2),
                    'Forward_Start': primer.forward_start,
                    'Reverse_Start': primer.reverse_start,
                    'Penalty_Score': round(primer.penalty, 4),
                    'T7_Promoter': primer.t7_promoter_seq,
                    'Primer_Type': 'T7_dsRNA'
                })
            else:
                data.append({
                    'Primer_Pair': i + 1,
                    'Gene_Target': gene_target,
                    'Forward_Sequence': primer.forward_seq,
                    'Reverse_Sequence': primer.reverse_seq,
                    'Forward_Tm': round(primer.forward_tm, 2),
                    'Reverse_Tm': round(primer.reverse_tm, 2),
                    'Product_Size': primer.product_size,
                    'Forward_GC%': round(primer.gc_content_f, 2),
                    'Reverse_GC%': round(primer.gc_content_r, 2),
                    'Forward_Start': primer.forward_start,
                    'Reverse_Start': primer.reverse_start,
                    'Penalty_Score': round(primer.penalty, 4),
                    'Primer_Type': 'Standard'
                })
        
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Primer_Results', index=False)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error exporting to Excel: {e}")
        return b""



def search_organism_with_gene_targets(organism_name, email, api_key=None):
    """Search for organism and return gene targets if available"""
    try:
        # Get gene targets for the organism
        suggestions = get_organism_suggestions_with_gene_targets()
        organism_targets = None
        
        # Normalize the input organism name
        organism_name_lower = organism_name.lower().strip()
        
        
        # Find matching organism and extract gene targets
        for category, subcategories in suggestions.items():
            for subcategory, organisms in subcategories.items():
                for item in organisms:
                    if len(item) == 3:  # New format with gene targets
                        common_name, scientific_name, gene_targets = item
                    else:  # Old format without gene targets
                        common_name, scientific_name = item
                        gene_targets = {"Essential genes": ["16S rRNA", "18S rRNA", "ACT1", "TUB1", "EF1A"]}
                    
                    # Improved matching logic - more strict
                    scientific_lower = scientific_name.lower().strip()
                    common_lower = common_name.lower().strip()
                    
                    # Check for exact matches first, then more flexible partial matches
                    if (organism_name_lower == scientific_lower or 
                        organism_name_lower == common_lower or
                        # Allow partial matches for genus names (e.g., "fusarium" matches "Fusarium oxysporum")
                        (organism_name_lower in scientific_lower and len(organism_name_lower) > 2) or
                        (organism_name_lower in common_lower and len(organism_name_lower) > 2) or
                        # Allow reverse partial matches (e.g., "oxysporum" matches "Fusarium oxysporum")
                        (scientific_lower in organism_name_lower and len(scientific_lower) > 2) or
                        (common_lower in organism_name_lower and len(common_lower) > 2)):
                        
                        
                        organism_targets = {
                            'organism': scientific_name,
                            'common_name': common_name,
                            'category': category,
                            'subcategory': subcategory,
                            'gene_targets': gene_targets
                        }
                        break
                if organism_targets:
                    break
            if organism_targets:
                break
        
        if not organism_targets:
            # Try to find similar organisms for suggestions
            similar_organisms = []
            for category, subcategories in suggestions.items():
                for subcategory, organisms in subcategories.items():
                    for item in organisms:
                        if len(item) == 3:
                            common_name, scientific_name, gene_targets = item
                            # Check if any word from the input matches any word from the organism names
                            input_words = set(organism_name_lower.split())
                            scientific_words = set(scientific_name.lower().split())
                            common_words = set(common_name.lower().split())
                            
                            if input_words.intersection(scientific_words) or input_words.intersection(common_words):
                                similar_organisms.append(f"{common_name} ({scientific_name})")
            
            if similar_organisms:
                pass  # Could show suggestions here if needed
        
        return organism_targets
        
    except Exception as e:
        print(f"Error in search_organism_with_gene_targets: {e}")
        return None

def perform_gene_targeted_design(organism_name, email, api_key, max_sequences, custom_params, enable_t7_dsrna, optimal_dsrna_length, check_transcription_efficiency):
    """Enhanced error handling for gene-targeted design"""
    
    class DesignError(Exception):
        """Custom exception for design errors"""
        pass
    
    try:
        # Validate inputs
        if not email or '@' not in email:
            raise DesignError("Valid email address required for NCBI access")
        
        if not organism_name or len(organism_name.strip()) < 3:
            raise DesignError("Organism name must be at least 3 characters")
        
        # Initialize with validation
        try:
            ncbi = ResilientNCBIConnector(email, api_key)
            if api_key and not ncbi.validate_api_key():
                st.warning("API key validation failed - proceeding without API key")
                ncbi = ResilientNCBIConnector(email, None)
        except Exception as e:
            raise DesignError(f"Failed to initialize NCBI connection: {e}")
        
        # Gene target validation
        if 'selected_gene_targets' not in st.session_state:
            raise DesignError("No gene targets selected. Please select gene categories first.")
        
        gene_targets = st.session_state.selected_gene_targets
        selected_genes = gene_targets.get('selected_genes', [])
        
        if not selected_genes:
            raise DesignError("No gene targets selected. Please select at least one gene category.")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each gene with individual error handling
        gene_sequences = []
        failed_genes = []
        
        designer = PrimerDesigner()
        
        # Get user-configured limit or default to 5
        max_genes = st.session_state.get('max_genes_to_process', 5)
        genes_to_process = selected_genes[:max_genes]
        
        with st.spinner(f"Designing gene-targeted primers for {organism_name}..."):
            for i, gene_info in enumerate(genes_to_process):
                try:
                    progress_bar.progress((i + 1) / len(genes_to_process))
                    status_text.text(f"Processing gene {i+1}/{len(genes_to_process)}: {gene_info}")
                    
                    # Parse gene info: "Category: Gene name"
                    if ': ' in gene_info:
                        category, gene_name = gene_info.split(': ', 1)
                        # Clean gene name for search
                        clean_gene = gene_name.split('(')[0].strip()
                        
                        # Create gene name variations for better matching using enhanced function
                        gene_variations = enhanced_gene_search_variations(clean_gene)
                        
                        # Search for gene-specific sequences with multiple strategies
                        seq_ids = []
                        found_sequence = False
                        
                        for gene_var in gene_variations:
                            if found_sequence:
                                break
                                
                            search_queries = [
                                f'"{organism_name}"[organism] AND "{gene_var}"[gene]',
                                f'"{organism_name}"[organism] AND "{gene_var}"[title]',
                                f'"{organism_name}"[organism] AND "{gene_var}"',
                                f'"{organism_name}"[organism] AND {gene_var}',
                            ]
                            
                            for search_query in search_queries:
                                try:
                                    results = ncbi.search_sequences(search_query, database="nucleotide", max_results=2)
                                    if results:
                                        seq_ids = results
                                        found_sequence = True
                                        break
                                except Exception as e:
                                    continue
                        
                        if seq_ids:
                            for seq_id in seq_ids[:1]:  # Take first sequence per gene
                                sequence = ncbi.fetch_sequence(seq_id)
                                if sequence and len(sequence) > 100:
                                    gene_sequences.append({
                                        'gene': clean_gene,
                                        'category': category,
                                        'sequence': sequence,
                                        'id': seq_id
                                    })
                                    break
                                    
                except Exception as gene_error:
                    failed_genes.append((gene_info, str(gene_error)))
                    st.warning(f"Failed to process {gene_info}: {gene_error}")
                    continue
            
            # Report results
            if failed_genes:
                st.warning(f"Failed to process {len(failed_genes)} genes:")
                for gene, error in failed_genes:
                    st.text(f"  • {gene}: {error}")
            
            if not gene_sequences:
                # Implement fallback strategy
                st.warning("No gene-specific sequences found. Attempting fallback to general organism search...")
                try:
                    search_query = f'"{organism_name}"[organism]'
                    seq_ids = ncbi.search_sequences(search_query, database="nucleotide", max_results=min(max_sequences, 5))
                    
                    if not seq_ids:
                        raise DesignError(f"No sequences found for {organism_name}")
                    
                    # Use the first sequence
                    sequence = ncbi.fetch_sequence(seq_ids[0])
                    if not sequence:
                        raise DesignError("Failed to fetch sequence")
                    
                    # Clean and prepare sequence
                    clean_sequence = re.sub(r'[^ATGCatgc]', '', sequence.upper())
                    if len(clean_sequence) < 100:
                        raise DesignError("Sequence too short for primer design")
                    
                    # Limit sequence size for performance
                    if len(clean_sequence) > 10000:
                        clean_sequence = clean_sequence[:10000]
                        st.info("Using first 10kb of sequence for primer design")
                    
                    # Design primers for fallback sequence
                    gene_target_name = "Gene-Targeted Design (Fallback)"
                    primers, primer3_results = designer.design_primers_with_complementarity(
                        clean_sequence, 
                        custom_params=custom_params,
                        add_t7_promoter=enable_t7_dsrna,
                        gene_target=gene_target_name,
                        target_type=st.session_state.get('target_type', 'linear'),
                        include_reverse_orientation=st.session_state.get('include_reverse_orientation', False)
                    )
                    st.session_state.primer3_results = primer3_results
                    
                    if not primers:
                        raise DesignError("No suitable primers found. Try adjusting parameters.")
                    
                    # Store fallback results
                    st.session_state.current_sequence = clean_sequence
                    st.session_state.sequence_info = {
                        "id": seq_ids[0],
                        "description": f"Gene-targeted design for {organism_name} (fallback)",
                        "length": len(clean_sequence),
                        "organism": organism_name,
                        "design_mode": "gene_targeted_fallback",
                        "gene_target_info": {
                            'gene_name': 'General organism sequence',
                            'gene_category': 'Fallback',
                            'sequence_source': "General organism sequence (fallback)"
                        }
                    }
                    OptimizedSessionManager.store_primers_optimized(primers)
                    
                    if enable_t7_dsrna:
                        st.session_state.t7_dsrna_enabled = True
                        st.session_state.t7_settings = {
                            'optimal_length': optimal_dsrna_length,
                            'check_efficiency': check_transcription_efficiency
                        }
                    
                    st.success(f"✅ Successfully designed {len(primers)} primer pairs for {gene_target_name}!")
                    st.info("📊 Go to Results tab to view detailed analysis!")
                    return
                    
                except Exception as e:
                    raise DesignError(f"Fallback strategy failed: {e}")
            else:
                st.success(f"Found {len(gene_sequences)} gene-specific sequences!")
                
                # Design primers for ALL found gene targets
                all_primers = []
                all_sequences = []
                gene_target_info_list = []
                
                for i, gene_seq in enumerate(gene_sequences):
                    gene_name = gene_seq['gene']
                    gene_category = gene_seq['category']
                    gene_target_name = f"{gene_name} ({gene_category})"
                    
                    # Clean and prepare sequence
                    clean_sequence = re.sub(r'[^ATGCatgc]', '', gene_seq['sequence'].upper())
                    if len(clean_sequence) > 10000:
                        clean_sequence = clean_sequence[:10000]
                        st.info(f"Using first 10kb of {gene_name} sequence for primer design")
                    
                    # Design primers for this gene
                    gene_primers, primer3_results = designer.design_primers_with_complementarity(
                        clean_sequence, 
                        custom_params=custom_params,
                        add_t7_promoter=enable_t7_dsrna,
                        gene_target=gene_target_name,
                        target_type=st.session_state.get('target_type', 'linear'),
                        include_reverse_orientation=st.session_state.get('include_reverse_orientation', False)
                    )
                    st.session_state.primer3_results = primer3_results
                    
                    if gene_primers:
                        st.success(f"✅ Designed {len(gene_primers)} primer pairs for {gene_name}")
                        all_primers.extend(gene_primers)
                        all_sequences.append({
                            'gene': gene_name,
                            'category': gene_category,
                            'sequence': clean_sequence,
                            'length': len(clean_sequence),
                            'id': gene_seq['id']
                        })
                        gene_target_info_list.append({
                            'gene_name': gene_name,
                            'gene_category': gene_category,
                            'sequence_source': f"Gene-specific sequence: {gene_name}",
                            'primer_count': len(gene_primers)
                        })
                    else:
                        st.warning(f"⚠️ No suitable primers found for {gene_name}")
                
                if not all_primers:
                    raise DesignError("No suitable primers found for any gene targets. Try adjusting parameters.")
                
                # Store comprehensive results
                st.session_state.current_sequence = all_sequences[0]['sequence']  # Store first sequence as primary
                st.session_state.all_gene_sequences = all_sequences  # Store all sequences
                
                st.session_state.sequence_info = {
                    "id": "multi_gene_design",
                    "description": f"Multi-gene targeted design for {organism_name}",
                    "length": sum(seq['length'] for seq in all_sequences),
                    "organism": organism_name,
                    "design_mode": "multi_gene_targeted",
                    "gene_targets": gene_target_info_list,
                    "total_genes": len(gene_sequences),
                    "total_primers": len(all_primers)
                }
                OptimizedSessionManager.store_primers_optimized(all_primers)
                
                if enable_t7_dsrna:
                    st.session_state.t7_dsrna_enabled = True
                    st.session_state.t7_settings = {
                        'optimal_length': optimal_dsrna_length,
                        'check_efficiency': check_transcription_efficiency
                    }
                
                st.success(f"🎯 **Successfully designed {len(all_primers)} primer pairs across {len(gene_sequences)} gene targets!**")
                st.info("📊 Go to Results tab to view detailed analysis!")
                
    except DesignError as de:
        st.error(f"Design Error: {de}")
        return
    except Exception as e:
        st.error(f"Unexpected error in gene-targeted design: {e}")
        st.error("Please try again or contact support if the problem persists.")
        return

def display_sirna_analysis(primer_pair: PrimerPair, target_sequence: str):
    """Display siRNA analysis for dsRNA primers"""
    if hasattr(primer_pair, 'has_t7_promoter') and primer_pair.has_t7_promoter:
        st.subheader("siRNA Optimization within dsRNA")
        
        # Extract dsRNA sequence
        dsrna_start = primer_pair.forward_start
        dsrna_end = primer_pair.reverse_start
        dsrna_sequence = target_sequence[dsrna_start:dsrna_end]
        
        optimizer = siRNAOptimizer()
        optimal_sirnas = optimizer.find_optimal_sirnas(dsrna_sequence)
        
        if optimal_sirnas:
            st.write("**Top siRNA candidates within dsRNA:**")
            
            sirna_data = []
            for i, sirna in enumerate(optimal_sirnas):
                sirna_data.append({
                    'Rank': i + 1,
                    'Sequence': sirna['sequence'],
                    'Position': sirna['position'],
                    'Score': sirna['score'],
                    'Efficiency': sirna['efficiency'],
                    'GC%': f"{sirna['details'].get('gc_content', 0):.1f}%"
                })
            
            sirna_df = pd.DataFrame(sirna_data)
            st.dataframe(sirna_df, use_container_width=True)
            
            # Detailed analysis for top siRNA
            if st.checkbox("Show detailed siRNA analysis"):
                top_sirna = optimal_sirnas[0]
                st.write("**Top siRNA detailed analysis:**")
                st.code(top_sirna['sequence'])
                
                details = top_sirna['details']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Efficiency factors:**")
                    for factor, value in details.items():
                        if factor != 'total_score' and isinstance(value, bool):
                            st.write(f"• {factor}: {'✓' if value else '✗'}")
                
                with col2:
                    st.metric("Total Score", details['total_score'])
                    st.metric("Predicted Efficiency", details['efficiency_prediction'])

def display_enhanced_gene_targets_interface(organism_targets):
    """Enhanced interface with filtering options"""
    if organism_targets:
        st.success(f"Gene Targets Available for {organism_targets['common_name']}")
        
        # Add filtering options
        st.subheader("Target Filtering Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            expression_filter = st.selectbox(
                "Expression specificity:",
                ["All genes", "Gut-specific", "Infection-responsive", "Essential only"]
            )
        
        with col2:
            check_beneficial = st.checkbox("Check beneficial species", value=True)
            if check_beneficial:
                beneficial_species = st.multiselect(
                    "Beneficial species to avoid:",
                    ["Apis mellifera", "Bombus terrestris", "Coccinella septempunctata"],
                    default=["Apis mellifera"]
                )
        
        with col3:
            prioritize_essential = st.checkbox("Prioritize essential genes", value=True)
        
        # Apply filters
        filter_manager = TargetSpecificFilterManager()
        
        filtered_targets = {}
        for category, genes in organism_targets['gene_targets'].items():
            # Apply expression filter
            if expression_filter != "All genes":
                filter_type = expression_filter.lower().replace(' ', '_').replace('-', '_')
                genes = filter_manager.filter_genes_by_expression(
                    organism_targets['organism'], genes, filter_type
                )
            
            if genes:  # Only include categories with remaining genes
                filtered_targets[category] = genes
        
        # Display filtered results with safety annotations
        for category, genes in filtered_targets.items():
            with st.expander(f"{category} ({len(genes)} genes)", expanded=True):
                for gene in genes:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"• {gene}")
                    
                    with col2:
                        if check_beneficial:
                            safety_check = filter_manager.check_beneficial_species_presence(
                                gene, beneficial_species
                            )
                            safe_count = sum(1 for result in safety_check.values() if not result['present'])
                            if safe_count == len(beneficial_species):
                                st.success("✓ Safe")
                            else:
                                st.warning("⚠ Check")
                    
                    with col3:
                        if prioritize_essential and any(essential in gene.lower() for essential in filter_manager.essential_genes['universal_essential']):
                            st.info("Essential")

def perform_enhanced_specificity_testing(primers: List[PrimerPair], target_sequence: str, 
                                       comparison_organisms: List[str], email: str, api_key: str = None, 
                                       target_organism: str = None):
    """Perform enhanced specificity testing with thermodynamic considerations"""
    if not primers or not comparison_organisms:
        return {}
    
    try:
        ncbi = ResilientNCBIConnector(email, api_key)
        analyzer = EnhancedSpecificityAnalyzer(ncbi)
        
        specificity_results = {}
        
        # Inform user about target organism exclusion
        if target_organism:
            st.info(f"🎯 **Specificity Testing**: Excluding target organism '{target_organism}' from comparison to avoid self-matches")
        
        # Test each primer pair
        for i, primer_pair in enumerate(primers[:3]):  # Test first 3 primer pairs
            st.write(f"Testing specificity for primer pair {i+1}...")
            
            result = analyzer.comprehensive_specificity_test(
                primer_pair, target_sequence, comparison_organisms, target_organism
            )
            
            specificity_results[f"primer_pair_{i+1}"] = result
            
            # Display risk organisms if any
            if result.get('risk_organisms'):
                st.warning(f"⚠️ Risk organisms found for primer pair {i+1}:")
                for risk in result['risk_organisms']:
                    st.write(f"  • {risk['organism']}: {risk['amplification_risk']:.1%} risk")
        
        # Store enhanced specificity results in session state
        st.session_state.enhanced_specificity_results = specificity_results
        st.session_state.specificity_target_organism = target_organism
        
        return specificity_results
        
    except Exception as e:
        st.error(f"Enhanced specificity testing failed: {e}")
        return {}

def perform_conservation_based_design(
    organism_name, email, api_key, max_sequences,
    conservation_threshold=None, window_size=None,
    enable_specificity_testing=False, specificity_threshold=None,
    comparison_organisms="", custom_params=None, enable_t7_dsrna=False
):
    st.warning("Conservation-based design has been removed. Falling back to Standard Design.")
    return perform_standard_design(
        organism_name, email, api_key, max_sequences,
        custom_params, enable_t7_dsrna,
        optimal_dsrna_length=300,  # keep your existing default
        check_transcription_efficiency=False
    )

def perform_standard_design(organism_name, email, api_key, max_sequences, custom_params, enable_t7_dsrna, optimal_dsrna_length, check_transcription_efficiency):
    """Perform standard primer design workflow"""
    with st.spinner(f"Searching for {organism_name} genomes..."):
        try:
            # Clear any existing gene targets and conservation data
            for key in ['selected_gene_targets', 'conserved_regions', 'specificity_results', 'analysis_metadata']:
                if key in st.session_state:
                    del st.session_state[key]
            
            ncbi = NCBIConnector(email, api_key)
            designer = PrimerDesigner()
            
            search_query = f'"{organism_name}"[organism]'
            st.write(f"Searching with query: `{search_query}`")
            
            seq_ids = ncbi.search_sequences(search_query, database="nucleotide", max_results=max_sequences)
            
            if seq_ids:
                st.success(f"Found {len(seq_ids)} sequences!")
                
                seq_id = seq_ids[0]
                st.info(f"Using sequence {seq_id} for primer design...")
                
                sequence = ncbi.fetch_sequence(seq_id)
                seq_info = ncbi.fetch_sequence_info(seq_id)
                
                if sequence:
                    clean_sequence = re.sub(r'[^ATGCatgc]', '', sequence.upper())
                    
                    if len(clean_sequence) < 50:
                        st.error("Sequence too short for primer design")
                    else:
                        if len(clean_sequence) > 100000:
                            st.warning(f"Large sequence ({len(clean_sequence):,} bp). Using first 100kb.")
                            clean_sequence = clean_sequence[:100000]
                        
                        st.session_state.current_sequence = clean_sequence
                        st.session_state.sequence_info = seq_info or {
                            "id": seq_id,
                            "description": f"Sequence {seq_id}",
                            "length": len(clean_sequence),
                            "organism": organism_name,
                            "design_mode": "standard"
                        }
                        
                        st.write("Designing primers...")
                        primers, primer3_results = designer.design_primers_with_complementarity(
                            clean_sequence, 
                            custom_params=custom_params,
                            add_t7_promoter=enable_t7_dsrna,
                            gene_target="Standard Design",
                            target_type=st.session_state.get('target_type', 'linear'),
                            include_reverse_orientation=st.session_state.get('include_reverse_orientation', False)
                        )
                        st.session_state.primer3_results = primer3_results
                        OptimizedSessionManager.store_primers_optimized(primers)
                        
                        if enable_t7_dsrna:
                            st.session_state.t7_dsrna_enabled = True
                            st.session_state.t7_settings = {
                                'optimal_length': optimal_dsrna_length,
                                'check_efficiency': check_transcription_efficiency
                            }
                        else:
                            st.session_state.t7_dsrna_enabled = False
                        
                        if primers:
                            st.success(f"✅ Successfully designed {len(primers)} primer pairs!")
                            
                            # Step 2: Specificity testing for standard primers
                            st.write("🎯 **Step 2: Testing primer specificity...**")
                            
                            # Get related organisms for specificity testing (100+ organisms)
                            related_organisms = get_related_organisms(organism_name, max_organisms=100)
                            st.write(f"Testing against {len(related_organisms)} related organisms for comprehensive specificity analysis...")
                            st.info(f"🎯 **Specificity Testing**: Excluding target organism '{organism_name}' from comparison to avoid self-matches")
                            
                            # Test specificity for the best primers
                            specificity_results = {}
                            analyzer = EnhancedSpecificityAnalyzer(NCBIConnector(email, api_key))
                            
                            # Add progress bar for specificity testing
                            progress_bar = st.progress(0)
                            total_tests = 3 * len(related_organisms)  # Testing top 3 primers
                            current_test = 0
                            
                            # Test the top 3 primers (best penalty scores)
                            best_primers = sorted(primers, key=lambda p: p.penalty)[:3]
                            
                            for i, primer in enumerate(best_primers):
                                # Create a test sequence from the primer binding region
                                test_sequence = clean_sequence[primer.forward_start:primer.reverse_start + 1]
                                
                                if len(test_sequence) > 50:  # Ensure we have enough sequence for testing
                                    try:
                                        primer_specificity = analyzer.test_specificity(
                                            test_sequence,
                                            related_organisms,
                                            target_organism=organism_name,
                                            max_similarity=0.7  # 70% similarity threshold
                                        )
                                        specificity_results[f"Primer Pair {i+1}"] = primer_specificity
                                        
                                        # Update progress bar
                                        current_test += len(related_organisms)
                                        progress_bar.progress(min(current_test / total_tests, 1.0))
                                        
                                    except Exception as e:
                                        st.warning(f"Could not test specificity for Primer Pair {i+1}: {e}")
                                        current_test += len(related_organisms)
                                        progress_bar.progress(min(current_test / total_tests, 1.0))
                            
                # Store specificity results and target organism info
                st.session_state.specificity_results = specificity_results
                st.session_state.specificity_target_organism = organism_name
                
                # Display specificity results
                if specificity_results:
                    st.subheader("Specificity Testing Results")
                    
                    specificity_data = []
                    for primer_name, results in specificity_results.items():
                        for organism, result in results.items():
                            if 'error' not in result:
                                # Safely access result keys with defaults
                                max_similarity = result.get('max_similarity', 0.0)
                                is_specific = result.get('is_specific', False)
                                sequences_tested = result.get('sequences_tested', 0)
                                
                                specificity_data.append({
                                    'Primer Pair': primer_name,
                                    'Test Organism': organism,
                                    'Max Similarity': f"{max_similarity:.1%}",
                                    'Specific': '✅ Yes' if is_specific else '❌ No',
                                    'Sequences Tested': sequences_tested
                                })
                    
                    if specificity_data:
                        specificity_df = pd.DataFrame(specificity_data)
                        st.dataframe(specificity_df, use_container_width=True)
                        
                        # Summary statistics
                        total_tests = len(specificity_data)
                        specific_tests = sum(1 for row in specificity_data if row['Specific'] == '✅ Yes')
                        specificity_percentage = (specific_tests / total_tests) * 100 if total_tests > 0 else 0
                        
                        if specificity_percentage >= 80:
                            st.success(f"🎯 Excellent specificity: {specific_tests}/{total_tests} tests passed ({specificity_percentage:.0f}%)")
                        elif specificity_percentage >= 60:
                            st.info(f"🎯 Good specificity: {specific_tests}/{total_tests} tests passed ({specificity_percentage:.0f}%)")
                        else:
                            st.warning(f"⚠️ Moderate specificity: {specific_tests}/{total_tests} tests passed ({specificity_percentage:.0f}%)")
                
                            preview_data = []
                            for i, primer in enumerate(primers[:5]):
                                preview_data.append({
                                    'Pair': i + 1,
                                    'Forward': primer.forward_seq[:30] + '...' if len(primer.forward_seq) > 30 else primer.forward_seq,
                                    'Reverse': primer.reverse_seq[:30] + '...' if len(primer.reverse_seq) > 30 else primer.reverse_seq,
                                    'Product Size': f"{primer.product_size} bp"
                                })
                            
                            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
                            st.info("📊 Go to other tabs to view detailed analysis with specificity results!")
                    else:
                        st.warning("No suitable primers found. Try adjusting parameters.")
                
                else:
                    st.error("Failed to fetch sequence")
            else:
                st.warning(f"No sequences found for {organism_name}")
                
        except Exception as e:
            st.error(f"Error: {e}")

def main():
    """Main Streamlit application"""
    
    # Initialize session state first
    init_session_state()
    
    # Ensure ALL critical session state variables exist with safe defaults
    required_vars = {
        't7_dsrna_enabled': False,
        'primers_designed': [],
        'sequence_info': {},
        'specificity_results': {},
        'selected_gene_targets': {},
        'analysis_metadata': {},
        'found_sequences': [],
        'target_organism': '',
        'conserved_regions': [],
        'conservation_sequences': [],
        'gene_target_stats': {},
        'search_results': None,
        'database_used': None,
        'comprehensive_analysis_results': None,
        't7_results': None,
        't7_settings': {},
        'current_sequence': '',
        'session_initialized': True,
        'last_activity': None,
        # ORGANISM SELECTION VARIABLES:
        'organism_name_input': '',
        'trigger_gene_search': False,
        'current_organism_targets': None,
        'selected_organism_name': '',
        'gene_targets_loaded': False,
        'stored_organism_name': '',
        'processing_gene_design': False,
        'selected_gene_categories': []
    }
    
    for var, default_value in required_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value
    
    st.title("AutoPrimer")
    
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # NCBI Configuration
    st.sidebar.subheader("NCBI Settings")
    st.sidebar.info("📧 **Email Required**: NCBI requires a valid email address for database access.")
    
    email = st.sidebar.text_input("Email (required for NCBI)", 
                                 placeholder="your.email@example.com",
                                 help="Required by NCBI for database access.")
    
    if st.sidebar.button("🚀 Use test email (demo@example.com)", type="secondary"):
        st.session_state.demo_email = "demo@example.com"
        st.rerun()
    
    if 'demo_email' in st.session_state:
        email = st.session_state.demo_email
        del st.session_state.demo_email
    
    api_key = st.sidebar.text_input("NCBI API Key (optional)", 
                                   type="password",
                                   help="For increased rate limits (not required)")
    
    # Primer Design Parameters
    st.sidebar.subheader("Primer Parameters")
    
    with st.sidebar.expander("Basic Parameters", expanded=True):
        opt_size = st.slider("Optimal primer size", 15, 30, 20)
        min_size = st.slider("Minimum primer size", 15, 25, 18)
        max_size = st.slider("Maximum primer size", 20, 35, 25)
        
        opt_tm = st.slider("Optimal Tm (°C)", 50.0, 70.0, 60.0, 0.5)
        min_tm = st.slider("Minimum Tm (°C)", 45.0, 65.0, 57.0, 0.5)
        max_tm = st.slider("Maximum Tm (°C)", 55.0, 75.0, 63.0, 0.5)
    
    with st.sidebar.expander("Advanced Parameters"):
        min_gc = st.slider("Minimum GC content (%)", 20.0, 50.0, 40.0, 1.0)
        max_gc = st.slider("Maximum GC content (%)", 50.0, 80.0, 60.0, 1.0)
        max_poly_x = st.slider("Max poly-X runs", 3, 6, 4)
        salt_conc = st.slider("Salt concentration (mM)", 10.0, 100.0, 50.0, 1.0)
        num_primers = st.number_input("Number of primer pairs to design", 1, 100, 30, 
                                     help="Maximum number of primer pairs to return per gene target")
        
        st.write("Product size ranges:")
        min_product = st.number_input("Minimum product size", 50, 500, 75)
        max_product = st.number_input("Maximum product size", 200, 2000, 1000)
    
    # T7 dsRNA Production Settings
    st.sidebar.subheader("🧬 dsRNA Production")
    enable_t7_dsrna = st.sidebar.checkbox(
        "Add T7 promoters for dsRNA production", 
        value=False,
        help="Adds T7 promoter sequences to both primers for bidirectional transcription and dsRNA synthesis"
    )
    
    # Initialize T7 dsRNA parameters with default values
    optimal_dsrna_length = False
    check_transcription_efficiency = False
    
    if enable_t7_dsrna:
        with st.sidebar.expander("dsRNA Parameters", expanded=True):
            st.info("**T7 dsRNA Production**\nAdds T7 promoter (TAATACGACTCACTATAGGG) to both forward and reverse primers. This enables:\n\n• Bidirectional transcription\n• Double-stranded RNA synthesis\n• RNAi applications\n• Pest management research")
            
            optimal_dsrna_length = st.checkbox(
                "Optimize for dsRNA length (100-500 bp)",
                value=True,
                help="Prioritize primer pairs that produce optimal dsRNA lengths for RNAi"
            )
            
            check_transcription_efficiency = st.checkbox(
                "Check T7 transcription efficiency",
                value=True,
                help="Analyze factors affecting T7 polymerase transcription efficiency"
            )
            
            if optimal_dsrna_length:
                min_product = max(min_product, 100)
                max_product = min(max_product, 500)
                st.write(f"**Adjusted for dsRNA:** {min_product}-{max_product} bp")
    
    custom_params = {
        'PRIMER_OPT_SIZE': opt_size,
        'PRIMER_MIN_SIZE': min_size,
        'PRIMER_MAX_SIZE': max_size,
        'PRIMER_OPT_TM': opt_tm,
        'PRIMER_MIN_TM': min_tm,
        'PRIMER_MAX_TM': max_tm,
        'PRIMER_MIN_GC': min_gc,
        'PRIMER_MAX_GC': max_gc,
        'PRIMER_MAX_POLY_X': max_poly_x,
        'PRIMER_SALT_MONOVALENT': salt_conc,
        'PRIMER_PRODUCT_SIZE_RANGE': [[min_product, max_product]],
        'PRIMER_NUM_RETURN': num_primers
    }
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Input", 
        "🔬 Results", 
        "📊 Analysis", 
        "💾 Export"
    ])
    
    state_check = check_session_state_validity()
    
    with tab1:
        st.header("Sequence Input")
        
        if state_check['has_primers']:
            st.success(f"✅ Current session: {state_check['primer_count']} primers designed for {state_check['sequence_length']:,} bp sequence")
        
        input_method = st.radio(
            "Choose input method:",
            ["Organism Name", "Direct Sequence"]
        )
        
        if input_method == "Organism Name":
            st.subheader("Search by Organism")
            
            if not email:
                st.error("❌ **Email Required**: Please enter an email address in the sidebar first to search for organisms.")
            else:
                st.info("💡 **Tip:** Enter the scientific name (e.g., 'Fusarium oxysporum') for best results.")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Check if an organism was selected from buttons or preserve current value
                default_organism = st.session_state.get('selected_organism_name', '')
                if not default_organism and st.session_state.get('stored_organism_name'):
                    # If no new selection but we have a stored organism, use that
                    default_organism = st.session_state.stored_organism_name
                
                organism_name = st.text_input("Enter organism name:", 
                                            value=default_organism,
                                            placeholder="e.g., Fusarium oxysporum, Coronavirus, Tetranychus urticae",
                                            key="organism_input")
                
                
                # Store the current organism name in session state for persistence
                if organism_name:
                    st.session_state.stored_organism_name = organism_name
                
                # Clear the selected organism after setting it, but preserve the input
                if 'selected_organism_name' in st.session_state:
                    del st.session_state['selected_organism_name']
            
            with col2:
                max_sequences = st.number_input("Max sequences to search", min_value=5, max_value=50, value=10)
            
            # ==========================================
            # NEW: WORKFLOW SELECTION
            # ==========================================
            st.subheader("🔬 Primer Design Strategy")
            
            workflow_choice = st.radio(
                "Choose your primer design approach:",
                [
                    "🎯 Gene-Targeted Design (Recommended for specific genes)",
                    "⚡ Standard Design (Single sequence, fastest)"
                ],
                help="Select based on your research goals: specific gene detection vs. speed"
            )
            
            # Initialize workflow variables
            gene_targets_workflow = "🎯 Gene-Targeted Design" in workflow_choice
            conservation_workflow = False  # Conservation workflow removed
            standard_workflow = "⚡ Standard Design" in workflow_choice
            
            # ==========================================
            # TARGET TYPE SELECTION & ORIENTATION STRATEGY
            # ==========================================
            st.subheader("🧬 Target Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_type = st.radio(
                    "Select target sequence type:",
                    ["linear", "circular"],
                    format_func=lambda x: "📏 Linear DNA/RNA" if x == "linear" else "🔄 Circular RNA/Viroid",
                    help="Choose based on your target: linear sequences or circular RNAs/viruses"
                )
                
                # Additional target characteristics
                target_application = st.selectbox(
                    "Application type:",
                    ["general_detection", "antisense", "method_development", "gene_expression", "splice_junction", "snp", "allele_specific"],
                    format_func=lambda x: {
                        "general_detection": "🔍 General Detection",
                        "antisense": "🔄 Antisense Detection", 
                        "method_development": "🧪 Method Development",
                        "gene_expression": "📊 Gene Expression",
                        "splice_junction": "✂️ Splice Junction",
                        "snp": "🧬 SNP Detection",
                        "allele_specific": "🎯 Allele-Specific"
                    }[x],
                    help="Select the primary application for your primers"
                )
            
            with col2:
                # Intelligent orientation strategy suggestion
                designer = PrimerDesigner()
                target_info = {
                    'topology': target_type,
                    'application': target_application,
                    'type': 'viral' if 'viral' in organism_name.lower() else 'general',
                    'length': 0  # Will be updated when sequence is loaded
                }
                
                strategy = designer.determine_orientation_strategy(target_info)
                
                # Display strategy recommendation
                if strategy['both_orientations'] is True:
                    st.success(f"✅ **Recommended: Both Orientations**")
                    st.write(f"**Reason:** {strategy['reason']}")
                    include_reverse_orientation = st.checkbox(
                        "Generate both orientations",
                        value=True,
                        help="Auto-recommended based on target characteristics"
                    )
                elif strategy['both_orientations'] is False:
                    st.warning(f"⚠️ **Not Recommended: Single Orientation**")
                    st.write(f"**Reason:** {strategy['reason']}")
                    include_reverse_orientation = st.checkbox(
                        "Generate both orientations (not recommended)",
                        value=False,
                        help="Not recommended for this application type"
                    )
                else:
                    st.info(f"💡 **User Choice: Both Orientations**")
                    st.write(f"**Reason:** {strategy['reason']}")
                    include_reverse_orientation = st.checkbox(
                        "Generate both orientations",
                        help="Choose based on your specific needs"
                    )
                
                # Show benefits when enabled
                if include_reverse_orientation:
                    if target_type == "circular":
                        st.info("🔄 **Circular Target Benefits:**\n• No inherent directionality\n• Same conserved region accessible from either direction\n• Backup options for difficult targets")
                    elif target_application == "antisense":
                        st.info("🔄 **Antisense Benefits:**\n• Detect both (+) and (-) strand RNA\n• Monitor viral replication\n• Comprehensive strand coverage")
                    elif target_application == "method_development":
                        st.info("🧪 **Method Development Benefits:**\n• Comparative primer testing\n• Backup options when one orientation fails\n• Optimization studies")
                    else:
                        st.info("🎯 **General Benefits:**\n• Increased detection robustness\n• Multiple targeting strategies\n• Backup primer options")
            
            # Store target configuration
            st.session_state.target_type = target_type
            st.session_state.target_application = target_application
            st.session_state.include_reverse_orientation = include_reverse_orientation
            
            # ==========================================
            # WORKFLOW 1: GENE-TARGETED DESIGN (FIXED)
            # ==========================================
            if gene_targets_workflow:
                st.info("🎯 **Gene-Targeted Design Mode**\nDesign primers for specific genes with known biological functions.")
                
                
                if organism_name and email:
                    # Simple gene target loading
                    if st.button("🔍 Load Gene Targets", type="secondary"):
                        with st.spinner(f"Loading gene targets for {organism_name}..."):
                            organism_targets = search_organism_with_gene_targets(organism_name, email, api_key)
                            
                            
                            if organism_targets:
                                st.session_state.current_organism_targets = organism_targets
                                st.success(f"Found gene targets for {organism_targets['common_name']}")
                            else:
                                st.warning("No specific gene targets found for this organism.")
                                
                                # Show some suggestions
                                st.info("💡 **Try these examples:**")
                                st.write("- Fusarium oxysporum")
                                st.write("- Fusarium graminearum") 
                                st.write("- Aspergillus niger")
                                st.write("- Botrytis cinerea")
                                st.write("- Or click on one of the organism buttons above")
                
                    # Display gene targets if available
                    if 'current_organism_targets' in st.session_state and st.session_state.current_organism_targets:
                        organism_targets = st.session_state.current_organism_targets
                        
                        # Show available gene categories
                        st.subheader("Available Gene Categories")
                        gene_categories = list(organism_targets['gene_targets'].keys())
                        
                        selected_categories = st.multiselect(
                            "Select gene categories:",
                            gene_categories,
                            default=gene_categories[:2] if len(gene_categories) >= 2 else gene_categories,
                            key="gene_category_select"
                        )
                        
                        if selected_categories:
                            # Store selection in session state
                            st.session_state.selected_gene_categories = selected_categories
                            
                            # Show selected genes
                            st.write("**Selected Gene Targets:**")
                            total_genes = 0
                            for category in selected_categories:
                                st.write(f"**{category}:**")
                                for gene in organism_targets['gene_targets'][category]:
                                    st.write(f"  • {gene}")
                                    total_genes += 1
                            
                            # Gene processing limit configuration
                            st.subheader("⚙️ Processing Configuration")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                max_genes_to_process = st.number_input(
                                    "Maximum genes to process:",
                                    min_value=1,
                                    max_value=min(50, total_genes),
                                    value=min(5, total_genes),
                                    help=f"Processing {total_genes} total genes. Higher numbers take longer but provide more comprehensive results."
                                )
                            
                            with col2:
                                if total_genes > 5:
                                    st.warning(f"⚠️ Processing {total_genes} genes may take several minutes")
                                    st.info("💡 **Tip**: Start with 5 genes for faster results, then increase if needed")
                            
                            # Store the max genes setting
                            st.session_state.max_genes_to_process = max_genes_to_process
                            
                            st.markdown("---")
                            st.markdown("### 🧬 Preview Target Sequences")
                            st.markdown("*Fetch and preview the sequences that primers will be designed around*")
                            
                            # Button to fetch and preview sequences
                            if st.button("🔍 Fetch & Preview Sequences", key="preview_sequences_btn"):
                                if not email:
                                    st.error("Email required for NCBI access")
                                else:
                                    with st.spinner("Fetching sequences from NCBI..."):
                                        try:
                                            ncbi = ResilientNCBIConnector(email, api_key)
                                            preview_sequences = []
                                            
                                            # Fetch sequences for selected genes
                                            for category in selected_categories:
                                                category_genes = organism_targets['gene_targets'][category]
                                                for gene_info in category_genes[:max_genes_to_process]:
                                                    # Extract gene name
                                                    gene_name = gene_info.split('(')[0].strip()
                                                    
                                                    # Create search queries
                                                    search_queries = [
                                                        f'"{organism_name}"[organism] AND "{gene_name}"[gene]',
                                                        f'"{organism_name}"[organism] AND "{gene_name}"[title]',
                                                        f'"{organism_name}"[organism] AND "{gene_name}"',
                                                    ]
                                                    
                                                    seq_id = None
                                                    for query in search_queries:
                                                        try:
                                                            results = ncbi.search_sequences(query, database="nucleotide", max_results=1)
                                                            if results:
                                                                seq_id = results[0]
                                                                break
                                                        except:
                                                            continue
                                                    
                                                    if seq_id:
                                                        sequence = ncbi.fetch_sequence(seq_id)
                                                        if sequence and len(sequence) > 100:
                                                            clean_seq = re.sub(r'[^ATGCatgc]', '', sequence.upper())
                                                            preview_sequences.append({
                                                                'category': category,
                                                                'gene': gene_name,
                                                                'full_gene_name': gene_info,
                                                                'sequence': clean_seq[:10000] if len(clean_seq) > 10000 else clean_seq,
                                                                'length': len(clean_seq),
                                                                'id': seq_id
                                                            })
                                            
                                            # Store in session state
                                            st.session_state.preview_sequences = preview_sequences
                                            
                                            if preview_sequences:
                                                st.success(f"✅ Fetched {len(preview_sequences)} sequences!")
                                            else:
                                                st.warning("⚠️ No sequences found. You can still proceed with primer design.")
                                        
                                        except Exception as e:
                                            st.error(f"Error fetching sequences: {e}")
                            
                            # Display preview sequences if available
                            if 'preview_sequences' in st.session_state and st.session_state.preview_sequences:
                                preview_sequences = st.session_state.preview_sequences
                                
                                # Group by category
                                sequences_by_category = {}
                                for seq_info in preview_sequences:
                                    cat = seq_info['category']
                                    if cat not in sequences_by_category:
                                        sequences_by_category[cat] = []
                                    sequences_by_category[cat].append(seq_info)
                                
                                # Display sequences grouped by category
                                for category, seq_list in sequences_by_category.items():
                                    with st.expander(f"📋 {category} ({len(seq_list)} sequences)", expanded=True):
                                        for seq_info in seq_list:
                                            st.markdown(f"**{seq_info['full_gene_name']}**")
                                            st.info(f"Sequence ID: {seq_info['id']} | Length: {seq_info['length']} bp")
                                            
                                            # Show sequence preview (first 500 bp, formatted)
                                            preview_seq = seq_info['sequence'][:500]
                                            if len(seq_info['sequence']) > 500:
                                                preview_seq += f"\n... (showing first 500 of {len(seq_info['sequence'])} bp)"
                                            
                                            formatted_seq = '\n'.join([preview_seq[i:i+60] for i in range(0, len(preview_seq), 60)])
                                            st.code(formatted_seq, language="text")
                                            
                                            if len(seq_info['sequence']) > 10000:
                                                st.info(f"ℹ️ Full sequence is {len(seq_info['sequence'])} bp. Primer3 will use the first 10kb for design.")
                                            
                                            st.markdown("---")
                            
                            st.markdown("---")
                            
                            # Design button
                            if st.button("🎯 Design Gene-Targeted Primers", type="primary", key="gene_design_final"):
                                # Set up gene target info for design
                                st.session_state.selected_gene_targets = {
                                    'organism_info': organism_targets,
                                    'selected_categories': selected_categories,
                                    'selected_genes': [f"{cat}: {gene}" for cat in selected_categories 
                                                     for gene in organism_targets['gene_targets'][cat]]
                                }
                                
                                
                                # Call the design function
                                perform_gene_targeted_design(
                                    organism_name, email, api_key, max_sequences, 
                                    custom_params, enable_t7_dsrna, 
                                    optimal_dsrna_length, check_transcription_efficiency
                                )
                            else:
                                st.info("Select at least one gene category to proceed.")
                        else:
                            if not email:
                                st.error("Email required for gene target search")
                            if not organism_name:
                                st.info("Enter an organism name above to search for gene targets.")
            
            # ==========================================
            # WORKFLOW 2: CONSERVATION-BASED DESIGN (REMOVED)
            # ==========================================
            elif conservation_workflow:
                st.warning("Conservation-based design has been removed. Running Standard Design instead.")
                if not email or not organism_name:
                    st.error("❌ Please provide email and organism name.")
                else:
                    perform_standard_design(
                        organism_name, email, api_key, max_sequences,
                        custom_params, enable_t7_dsrna, optimal_dsrna_length,
                        check_transcription_efficiency
                    )
            
            # ==========================================
            # WORKFLOW 3: STANDARD DESIGN
            # ==========================================
            else:  # standard_workflow
                st.info("⚡ **Standard Design Mode**\nQuick primer design from the first available sequence. Ideal for rapid prototyping and basic applications.")
                
                # Search button for standard design
                if st.button("⚡ Design Standard Primers", type="primary", use_container_width=True):
                    if not email or not organism_name:
                        st.error("❌ Please provide email and organism name.")
                    else:
                        perform_standard_design(organism_name, email, api_key, max_sequences, custom_params, enable_t7_dsrna, optimal_dsrna_length, check_transcription_efficiency)
            
            # Agricultural Pests & Pathogens section with improved layout
            st.markdown("---")
            st.markdown("### 🎯 Quick Select: Agricultural Pests & Pathogens")
            st.markdown("*Click any button below to automatically search for that organism*")
            
            suggestions = get_organism_suggestions()
            
            # Show summary of available organisms
            total_organisms = sum(len(orgs) for subcats in suggestions.values() for orgs in subcats.values())
            st.info(f"📊 **{total_organisms} organisms** available across {len(suggestions)} categories")
            
            # Create expandable sections for better organization
            for category, subcategories in suggestions.items():
                with st.expander(f"{category} ({sum(len(orgs) for orgs in subcategories.values())} organisms)", expanded=False):
                    for subcategory, organisms in subcategories.items():
                        st.markdown(f"**{subcategory}**")
                        
                        # Use a more compact grid layout
                        num_cols = min(len(organisms), 4)  # Max 4 columns for better readability
                        cols = st.columns(num_cols)
                        
                        for i, organism_item in enumerate(organisms):
                            # Handle both old format (common_name, latin_name) and new format (common_name, latin_name, gene_targets)
                            if len(organism_item) == 3:
                                common_name, latin_name, gene_targets = organism_item
                            else:
                                common_name, latin_name = organism_item
                            
                            with cols[i % num_cols]:
                                # Create a unique key and use callback to set organism name directly
                                button_key = f"suggest_{category}_{subcategory}_{i}_{latin_name.replace(' ', '_')}"
                                
                                if st.button(
                                    common_name, 
                                    key=button_key, 
                                    help=f"Search for {latin_name}",
                                    use_container_width=True
                                ):
                                    # Set the organism name to appear in the text input
                                    st.session_state.selected_organism_name = latin_name
                                    st.rerun()
                        
                        # Add small spacing between subcategories
                        if subcategory != list(subcategories.keys())[-1]:  # Not the last subcategory
                            st.markdown("")
            
        
        elif input_method == "Direct Sequence":
            sequence_input = st.text_area("Enter DNA sequence:", 
                                        placeholder="ATGCGATCGATCG...",
                                        height=150)
            
            st.markdown("---")
            
            # Optional target region selector
            st.markdown("### Target Region Options")
            use_target_region = st.checkbox(
                "🔹 Specify target sub-region within sequence (optional)",
                value=False,
                help="If checked, primers will be designed only within the specified region. Leave unchecked to use the entire sequence."
            )
            
            target_region = None
            target_preview_seq = ""
            
            if sequence_input:
                clean_seq_preview = re.sub(r'[^ATGCatgc]', '', sequence_input.upper())
                
                if use_target_region:
                    col_reg1, col_reg2 = st.columns(2)
                    with col_reg1:
                        region_start = st.number_input(
                            "Region start (0-based index, inclusive)",
                            min_value=0,
                            max_value=max(0, len(clean_seq_preview) - 1) if clean_seq_preview else 0,
                            value=0,
                            key="direct_seq_region_start"
                        )
                    with col_reg2:
                        region_end = st.number_input(
                            "Region end (0-based index, exclusive)",
                            min_value=1,
                            max_value=len(clean_seq_preview) if clean_seq_preview else 1,
                            value=len(clean_seq_preview) if clean_seq_preview else 1,
                            key="direct_seq_region_end"
                        )
                    
                    if region_end > region_start and region_end <= len(clean_seq_preview):
                        target_region = (region_start, region_end)
                        target_preview_seq = clean_seq_preview[region_start:region_end]
                    else:
                        st.warning("Region indices are invalid for the current sequence length.")
                        target_preview_seq = clean_seq_preview
                else:
                    target_preview_seq = clean_seq_preview
                
                # Show target sequence preview
                if target_preview_seq:
                    st.markdown("---")
                    st.markdown("### 🎯 Target Sequence Preview")
                    st.markdown("*This is the sequence that primers will be designed around:*")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Sequence Length", f"{len(target_preview_seq)} bp")
                    with col_info2:
                        if use_target_region and target_region:
                            st.metric("Target Region", f"Pos {region_start} to {region_end-1}")
                        else:
                            st.metric("Target Region", "Full sequence")
                    
                    # Show sequence in a code block with word wrapping
                    st.markdown("**Target sequence:**")
                    # Format sequence for better readability (60 chars per line)
                    formatted_seq = '\n'.join([target_preview_seq[i:i+60] for i in range(0, len(target_preview_seq), 60)])
                    st.code(formatted_seq, language="text")
                    
                    if len(target_preview_seq) < 50:
                        st.warning("⚠️ Target sequence is very short (< 50 bp). Primer design may be limited.")
                    elif len(target_preview_seq) > 10000:
                        st.info("ℹ️ Large target sequence. Primer3 will search for optimal regions within it.")
                    
                    st.markdown("---")
            else:
                # Show placeholder when no sequence is entered
                st.info("💡 Enter a DNA sequence above to see the target sequence preview")
            
            if st.button("Design Primers", type="primary"):
                if not sequence_input:
                    st.error("Please provide a DNA sequence")
                else:
                    with st.spinner("Designing primers..."):
                        try:
                            designer = PrimerDesigner()
                            
                            clean_seq = re.sub(r'[^ATGCatgc]', '', sequence_input.upper())
                            
                            st.session_state.current_sequence = clean_seq
                            st.session_state.sequence_info = {
                                "length": len(clean_seq),
                                "description": "User-provided sequence",
                                "organism": "User input",
                                "id": "user_sequence"
                            }
                            
                            primers, primer3_results = designer.design_primers_with_complementarity(
                                clean_seq, 
                                target_region=target_region,
                                custom_params=custom_params,
                                add_t7_promoter=enable_t7_dsrna,
                                gene_target="User Input Sequence",
                                target_type=st.session_state.get('target_type', 'linear'),
                                include_reverse_orientation=st.session_state.get('include_reverse_orientation', False)
                            )
                            st.session_state.primer3_results = primer3_results
                            OptimizedSessionManager.store_primers_optimized(primers)
                            
                            if enable_t7_dsrna:
                                st.session_state.t7_dsrna_enabled = True
                                st.session_state.t7_settings = {
                                    'optimal_length': optimal_dsrna_length,
                                    'check_efficiency': check_transcription_efficiency
                                }
                            else:
                                st.session_state.t7_dsrna_enabled = False
                            
                            if primers:
                                st.success(f"Successfully designed {len(primers)} primer pairs!")
                                
                                # Note about specificity testing for direct sequence input
                                st.info("ℹ️ **Note**: Specificity testing is not available for direct sequence input since no organism information is provided. For specificity testing, please use the 'Organism Name' input method.")
                                
                                st.info("📊 Go to other tabs to view detailed analysis!")
                            else:
                                st.warning("No suitable primers found with current parameters")
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    with tab2:
        st.header("Primer Design Results")
        
        # Check session state validity fresh for this tab
        current_state_check = check_session_state_validity()
        
        if not current_state_check['has_primers']:
            st.info("No primers designed yet. Please use the Input tab to design primers.")
            st.stop()
        
        primers = OptimizedSessionManager.get_primers_optimized()
        t7_enabled = st.session_state.get('t7_dsrna_enabled', False)
        
        # Display gene target context if available
        try:
            display_results_with_gene_context()
        except Exception as e:
            st.warning(f"Gene target context display unavailable: {e}")
            # Continue with basic results display even if gene context fails
        
        if t7_enabled:
            st.info("🧬 **T7 dsRNA Mode Active** - Primers include T7 promoter sequences for double-stranded RNA production")
        
        # Display orientation information if both orientations were generated
        primer3_results = st.session_state.get('primer3_results', {})
        if primer3_results.get('REVERSE_ORIENTATION_GENERATED'):
            target_type = primer3_results.get('TARGET_TYPE', 'linear')
            total_pairs = primer3_results.get('TOTAL_PRIMER_PAIRS', len(primers))
            primary_pairs = total_pairs // 2
            
            st.success(f"🔄 **Both Orientations Generated** - {primary_pairs} primary + {primary_pairs} reverse orientation primer pairs")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Primary Orientation", f"{primary_pairs} pairs")
            with col2:
                st.metric("Reverse Orientation", f"{primary_pairs} pairs")
            with col3:
                st.metric("Target Type", target_type.title())
            
            # Show orientation comparison
            with st.expander("🔄 Orientation Comparison", expanded=False):
                st.write("**Primary Orientation:** Standard forward/reverse primer pairs")
                st.write("**Reverse Orientation:** Swapped and reverse-complemented primers")
                
                if target_type == "circular":
                    st.info("🔄 **Circular Target Benefits:**\n• No inherent directionality\n• Same conserved region accessible from either direction\n• Backup options for difficult targets")
                else:
                    st.info("🎯 **Linear Target Benefits:**\n• Multiple targeting strategies\n• Increased detection robustness\n• Backup primer options")
        
        # Check if this is conservation-based analysis
        analysis_metadata = st.session_state.get('analysis_metadata', {})
        is_conservation_based = analysis_metadata.get('type') == 'conservation_based'
        
        if is_conservation_based:
            st.success("🧬 **Conservation-Based Design** - Primers designed from conserved regions across multiple sequences")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sequences Analyzed", analysis_metadata.get('sequences_analyzed', 'N/A'))
            with col2:
                conservation_thresh = analysis_metadata.get('conservation_threshold', 0)
                st.metric("Conservation Threshold", f"{conservation_thresh:.0%}")
            with col3:
                specificity_tested = analysis_metadata.get('specificity_tested', False)
                st.metric("Specificity Tested", "Yes" if specificity_tested else "No")
            with col4:
                if specificity_tested:
                    spec_thresh = analysis_metadata.get('specificity_threshold', 0)
                    st.metric("Specificity Threshold", f"{spec_thresh:.0%}")
        
        # Sequence information
        if st.session_state.sequence_info:
            st.subheader("Sequence Information")
            info = st.session_state.sequence_info
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", f"{info.get('length', 'N/A'):,} bp")
            with col2:
                st.metric("Organism", info.get('organism', 'N/A'))
            with col3:
                st.metric("ID", info.get('id', 'N/A'))
            
            if 'description' in info:
                st.write(f"**Description:** {info['description']}")
            
            
            # Show multi-gene information if available
            if info.get('design_mode') == 'multi_gene_targeted':
                st.success("🎯 **Multi-Gene Targeted Design** - Primers designed across multiple gene targets")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Genes", info.get('total_genes', 'N/A'))
                with col2:
                    st.metric("Total Primers", info.get('total_primers', 'N/A'))
                with col3:
                    avg_primers_per_gene = info.get('total_primers', 0) / info.get('total_genes', 1)
                    st.metric("Avg Primers/Gene", f"{avg_primers_per_gene:.1f}")
                with col4:
                    st.metric("Total Sequence Length", f"{info.get('length', 0):,} bp")
                
                # Show gene target breakdown
                if 'gene_targets' in info:
                    st.subheader("Gene Target Breakdown")
                    gene_breakdown_data = []
                    for target in info['gene_targets']:
                        gene_breakdown_data.append({
                            'Gene': target['gene_name'],
                            'Category': target['gene_category'],
                            'Primers Designed': target['primer_count']
                        })
                    
                    gene_breakdown_df = pd.DataFrame(gene_breakdown_data)
                    st.dataframe(gene_breakdown_df, use_container_width=True)
        
        
        # Show enhanced specificity results if available
        if hasattr(st.session_state, 'enhanced_specificity_results') and st.session_state.enhanced_specificity_results:
            st.subheader("Enhanced Specificity Testing Results")
            
            # Show target organism exclusion information
            target_organism = st.session_state.get('specificity_target_organism', 'Unknown')
            if target_organism and target_organism != 'Unknown':
                st.info(f"🎯 **Target Organism Excluded**: '{target_organism}' was excluded from specificity testing to avoid self-matches")
            
            enhanced_results = st.session_state.enhanced_specificity_results
            
            # Display results for each primer pair
            for primer_name, primer_result in enhanced_results.items():
                st.write(f"**{primer_name.replace('_', ' ').title()}**")
                
                if primer_result.get('risk_organisms'):
                    st.warning(f"⚠️ **Risk Organisms Found**:")
                    risk_data = []
                    for risk in primer_result['risk_organisms']:
                        risk_data.append({
                            'Organism': risk['organism'],
                            'Amplification Risk': f"{risk['amplification_risk']:.1%}",
                            'Forward Risk': f"{risk['forward_risk']:.1%}",
                            'Reverse Risk': f"{risk['reverse_risk']:.1%}"
                        })
                    
                    if risk_data:
                        risk_df = pd.DataFrame(risk_data)
                        st.dataframe(risk_df, use_container_width=True)
                else:
                    st.success("✅ **No risk organisms detected** - primers show good specificity")
                
                # Overall specificity assessment
                overall_specificity = primer_result.get('overall_specificity', True)
                if overall_specificity:
                    st.success("🎯 **Overall Assessment**: Primers are specific to target organism")
                else:
                    st.warning("⚠️ **Overall Assessment**: Primers may cross-react with other organisms")
                
                st.write("---")
        
        # Show legacy specificity results if available (fallback)
        elif hasattr(st.session_state, 'specificity_results') and st.session_state.specificity_results:
            st.subheader("Specificity Testing Results")
            
            # Show target organism exclusion information if available
            target_organism = st.session_state.get('specificity_target_organism', None)
            if target_organism:
                st.info(f"🎯 **Target Organism Excluded**: '{target_organism}' was excluded from specificity testing to avoid self-matches")
            
            specificity_results = st.session_state.specificity_results
            specificity_data = []
            
            for organism, result in specificity_results.items():
                if 'error' not in result and isinstance(result, dict):
                    # Safely access result keys with defaults
                    max_similarity = result.get('max_similarity', 0.0)
                    is_specific = result.get('is_specific', False)
                    sequences_tested = result.get('sequences_tested', 0)
                    
                    specificity_data.append({
                        'Organism': organism,
                        'Max Similarity': f"{max_similarity:.1%}",
                        'Specific': '✅ Yes' if is_specific else '❌ No',
                        'Sequences Tested': sequences_tested
                    })
            
            if specificity_data:
                specificity_df = pd.DataFrame(specificity_data)
                st.dataframe(specificity_df, use_container_width=True)
                
                # Summary
                total_orgs = len(specificity_data)
                specific_orgs = sum(1 for row in specificity_data if row['Specific'] == '✅ Yes')
                specificity_percentage = (specific_orgs / total_orgs) * 100 if total_orgs > 0 else 0
                
                if specificity_percentage >= 80:
                    st.success(f"🎯 Excellent specificity: {specific_orgs}/{total_orgs} organisms ({specificity_percentage:.0f}%)")
                elif specificity_percentage >= 60:
                    st.info(f"🎯 Good specificity: {specific_orgs}/{total_orgs} organisms ({specificity_percentage:.0f}%)")
                else:
                    st.warning(f"⚠️ Moderate specificity: {specific_orgs}/{total_orgs} organisms ({specificity_percentage:.0f}%)")
        
        # Primer results table
        st.subheader("Primer Pairs")
        
        # Get the current sequence to extract target regions
        current_sequence = st.session_state.get('current_sequence', '')
        
        data = []
        for i, primer in enumerate(primers):
            # Handle existing primer pairs that don't have gene_target attribute
            gene_target = getattr(primer, 'gene_target', 'Standard Design')
            
            # Extract target sequence for this primer pair
            target_sequence = ""
            if current_sequence and hasattr(primer, 'forward_start') and hasattr(primer, 'product_size'):
                try:
                    start = primer.forward_start
                    end = start + primer.product_size
                    if 0 <= start < len(current_sequence) and end <= len(current_sequence):
                        target_sequence = current_sequence[start:end]
                    else:
                        target_sequence = f"[Position {start}-{end} out of range]"
                except Exception:
                    target_sequence = "[Unable to extract]"
            elif not current_sequence:
                target_sequence = "[Sequence not available]"
            else:
                target_sequence = "[Position data unavailable]"
            
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                row = {
                    'Pair': i + 1,
                    'Gene Target': gene_target,
                    'Target Sequence': target_sequence,
                    'Forward (with T7)': primer.forward_seq,
                    'Reverse (with T7)': primer.reverse_seq,
                    'Core Forward': primer.core_forward_seq,
                    'Core Reverse': primer.core_reverse_seq,
                    'Core Tm': f"{primer.forward_tm:.1f}°C / {primer.reverse_tm:.1f}°C",
                    'dsRNA Size': f"{primer.product_size} bp",
                    'Core GC%': f"{primer.gc_content_f:.1f}% / {primer.gc_content_r:.1f}%",
                    'Penalty': f"{primer.penalty:.3f}"
                }
            else:
                row = {
                    'Pair': i + 1,
                    'Gene Target': gene_target,
                    'Target Sequence': target_sequence,
                    'Forward Sequence': primer.forward_seq,
                    'Reverse Sequence': primer.reverse_seq,
                    'Forward Tm': f"{primer.forward_tm:.1f}°C",
                    'Reverse Tm': f"{primer.reverse_tm:.1f}°C",
                    'Product Size': f"{primer.product_size} bp",
                    'Forward GC%': f"{primer.gc_content_f:.1f}%",
                    'Reverse GC%': f"{primer.gc_content_r:.1f}%",
                    'Penalty': f"{primer.penalty:.3f}"
                }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed view for selected primer
        st.subheader("Detailed View")
        selected_primer = st.selectbox("Select primer pair for details:", 
                                     range(len(primers)), 
                                     format_func=lambda x: f"Pair {x+1}")
        
        if selected_primer < len(primers):
            primer = primers[selected_primer]
            
            if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                st.info("🧬 **T7 dsRNA Primer Pair** - Includes T7 promoter for double-stranded RNA synthesis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Forward Primer (with T7)**")
                    st.code(primer.forward_seq, language="text")
                    st.write("**Core Forward Primer**")
                    st.code(primer.core_forward_seq, language="text")
                    st.write(f"- Core Position: {primer.forward_start}")
                    st.write(f"- Core Length: {len(primer.core_forward_seq)} bp")
                    st.write(f"- Core Tm: {primer.forward_tm:.2f}°C")
                    st.write(f"- Core GC Content: {primer.gc_content_f:.1f}%")
                    st.write(f"- Full Length (with T7): {len(primer.forward_seq)} bp")
                
                with col2:
                    st.write("**Reverse Primer (with T7)**")
                    st.code(primer.reverse_seq, language="text")
                    st.write("**Core Reverse Primer**")
                    st.code(primer.core_reverse_seq, language="text")
                    st.write(f"- Core Position: {primer.reverse_start}")
                    st.write(f"- Core Length: {len(primer.core_reverse_seq)} bp")
                    st.write(f"- Core Tm: {primer.reverse_tm:.2f}°C")
                    st.write(f"- Core GC Content: {primer.gc_content_r:.1f}%")
                    st.write(f"- Full Length (with T7): {len(primer.reverse_seq)} bp")
                
                st.write(f"**dsRNA Product Size:** {primer.product_size} bp")
                st.write(f"**T7 Promoter:** {primer.t7_promoter_seq}")
                
                # dsRNA analysis
                if st.session_state.current_sequence:
                    designer = PrimerDesigner()
                    dsrna_props = designer.calculate_dsrna_properties(primer, st.session_state.current_sequence)
                    
                    if dsrna_props:
                        st.subheader("dsRNA Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("dsRNA Length", f"{dsrna_props.get('dsrna_length', 'N/A')} bp")
                        with col2:
                            st.metric("dsRNA GC Content", f"{dsrna_props.get('dsrna_gc_content', 'N/A'):.1f}%")
                        with col3:
                            st.metric("T7 Efficiency", dsrna_props.get('transcription_efficiency', 'N/A'))
                        
                        st.write("**dsRNA Quality Indicators:**")
                        if dsrna_props.get('optimal_length'):
                            st.success("✅ Optimal length for RNAi (100-500 bp)")
                        else:
                            st.warning("⚠️ Length outside optimal range for RNAi")
                        
                        if dsrna_props.get('moderate_gc'):
                            st.success("✅ Moderate GC content (40-60%)")
                        else:
                            st.warning("⚠️ Extreme GC content may affect efficiency")
                        
                        st.write(f"**Transcription Start:** {dsrna_props.get('transcription_start', 'N/A')} (G is optimal for T7)")
                        st.write(f"**Estimated Yield:** {dsrna_props.get('estimated_yield', 'N/A')}")
                        
                        if 'target_sequence' in dsrna_props:
                            st.write("**Complete Target Sequence (for gBlock synthesis):**")
                            st.code(dsrna_props['target_sequence'], language="text")
                            st.info("💡 **gBlock Synthesis**: This complete sequence can be used to order a synthetic DNA fragment (gBlock) as a positive control for your dsRNA production.")
                        
                        # siRNA Analysis within dsRNA
                        st.subheader("siRNA Analysis within dsRNA")
                        if st.checkbox("Show siRNA optimization analysis", key=f"sirna_analysis_{selected_primer}"):
                            display_sirna_analysis(primer, st.session_state.current_sequence)
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Forward Primer**")
                    st.code(primer.forward_seq, language="text")
                    st.write(f"- Position: {primer.forward_start}")
                    st.write(f"- Length: {len(primer.forward_seq)} bp")
                    st.write(f"- Tm: {primer.forward_tm:.2f}°C")
                    st.write(f"- GC Content: {primer.gc_content_f:.1f}%")
                
                with col2:
                    st.write("**Reverse Primer**")
                    st.code(primer.reverse_seq, language="text")
                    st.write(f"- Position: {primer.reverse_start}")
                    st.write(f"- Length: {len(primer.reverse_seq)} bp")
                    st.write(f"- Tm: {primer.reverse_tm:.2f}°C")
                    st.write(f"- GC Content: {primer.gc_content_r:.1f}%")
                
                st.write(f"**Product Size:** {primer.product_size} bp")
                
                # Show complete target sequence for gBlock synthesis
                if st.session_state.current_sequence:
                    target_start = primer.forward_start + len(primer.forward_seq)
                    target_end = primer.reverse_start
                    if target_end > target_start:
                        target_sequence = st.session_state.current_sequence[target_start:target_end]
                        st.write("**Complete Target Sequence (for gBlock synthesis):**")
                        st.code(target_sequence, language="text")
                        st.info("💡 **gBlock Synthesis**: This complete sequence can be used to order a synthetic DNA fragment (gBlock) as a positive control for your PCR.")
            
            st.write(f"**Penalty Score:** {primer.penalty:.4f}")
        
        # Complementarity Analysis
        st.subheader("Primer Complementarity Analysis")
        if st.checkbox("Show Primer3 complementarity analysis", key=f"complementarity_{selected_primer}"):
            primer3_results = st.session_state.get('primer3_results', None)
            display_primer3_complementarity_analysis(primer, selected_primer, primer3_results)
        
        # Literature Comparison
        st.subheader("Literature Comparison")
        if st.checkbox("Compare with literature primers", key=f"literature_{selected_primer}"):
            # Try to extract target name from organism or sequence info
            target_name = "Unknown"
            if st.session_state.sequence_info:
                organism = st.session_state.sequence_info.get('organism', '')
                if 'HLVd' in organism or 'hop latent' in organism.lower():
                    target_name = "HLVd"
                elif 'CEVd' in organism or 'citrus exocortis' in organism.lower():
                    target_name = "CEVd"
                elif 'ASBVd' in organism or 'avocado sunblotch' in organism.lower():
                    target_name = "ASBVd"
            
            if target_name == "Unknown":
                target_name = st.text_input("Enter target name for literature comparison:", value="HLVd", 
                                          help="Enter the target name (e.g., HLVd, CEVd, ASBVd) to compare with literature primers")
            
            if target_name and target_name != "Unknown":
                designer = PrimerDesigner()
                designer.compare_with_literature([primer], target_name)
        
        # Conservation and Specificity Summary
        if is_conservation_based:
            with st.expander("Analysis Summary", expanded=False):
                st.write("**Conservation-Based Primer Design Summary:**")
                st.write(f"- Analyzed {analysis_metadata.get('sequences_analyzed', 'N/A')} sequences from {st.session_state.get('target_organism', 'target organism')}")
                st.write(f"- Used {analysis_metadata.get('conservation_threshold', 0):.0%} conservation threshold")
                
                if analysis_metadata.get('specificity_tested'):
                    st.write(f"- Tested specificity against related organisms")
                    st.write(f"- Used {analysis_metadata.get('specificity_threshold', 0):.0%} specificity threshold")
                    
                    if hasattr(st.session_state, 'specificity_results'):
                        total_tested = len(st.session_state.specificity_results)
                        specific_count = sum(1 for result in st.session_state.specificity_results.values() 
                                           if result.get('is_specific', False))
                        st.write(f"- Specificity results: {specific_count}/{total_tested} organisms passed")
                
                st.write(f"- Final result: {len(primers)} primer pairs designed")
        
    
    with tab3:
        st.header("Primer Analysis")
        
        # Check session state validity fresh for this tab
        current_state_check = check_session_state_validity()
        
        if not current_state_check['has_primers']:
            st.info("No primers designed yet. Please use the Input tab to design primers.")
            st.stop()
        
        primers = OptimizedSessionManager.get_primers_optimized()
        t7_enabled = st.session_state.get('t7_dsrna_enabled', False)
        
        if t7_enabled:
            st.info("🧬 **T7 dsRNA Analysis Mode** - Showing analysis for dsRNA production primers")
        
        # Primer visualization charts
        st.subheader("Primer Properties Overview")
        fig = create_primer_visualization(primers)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not create primer visualization")
        
        # Enhanced sequence visualization section
        if current_state_check['has_sequence']:
            st.subheader("Sequence Visualization Options")
            
            # Visualization type selector
            viz_type = st.radio(
                "Choose visualization type:",
                ["Single Primer Detail", "Multi-Primer Comparison", "Coverage Heatmap"],
                horizontal=True
            )
            
            if viz_type == "Single Primer Detail":
                st.write("**Detailed view of individual primer binding sites**")
                selected_for_diagram = st.selectbox(
                    "Select primer pair for detailed visualization:", 
                    range(len(primers)), 
                    format_func=lambda x: f"Pair {x+1} (Product: {primers[x].product_size} bp, Penalty: {primers[x].penalty:.3f})",
                    key="enhanced_diagram_select"
                )
                
                enhanced_fig = create_enhanced_sequence_diagram(
                    st.session_state.current_sequence, 
                    primers, 
                    selected_for_diagram
                )
                if enhanced_fig:
                    st.plotly_chart(enhanced_fig, use_container_width=True)
                    
                    # Additional primer details
                    primer = primers[selected_for_diagram]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                            st.metric("Core Forward Length", f"{len(primer.core_forward_seq)} bp")
                        else:
                            st.metric("Forward Length", f"{len(primer.forward_seq)} bp")
                    with col2:
                        if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                            st.metric("Core Reverse Length", f"{len(primer.core_reverse_seq)} bp")
                        else:
                            st.metric("Reverse Length", f"{len(primer.reverse_seq)} bp")
                    with col3:
                        st.metric("Forward Position", f"{primer.forward_start:,}")
                    with col4:
                        st.metric("Reverse Position", f"{primer.reverse_start:,}")
            
            elif viz_type == "Multi-Primer Comparison":
                st.write("**Compare multiple primer pairs on the same sequence**")
                max_primers_to_show = st.slider("Number of primer pairs to compare:", 2, min(10, len(primers)), 5)
                
                multi_fig = create_multi_primer_comparison(
                    st.session_state.current_sequence,
                    primers,
                    max_primers_to_show
                )
                if multi_fig:
                    st.plotly_chart(multi_fig, use_container_width=True)
                    
                    # Summary table for compared primers
                    st.write("**Comparison Summary:**")
                    comparison_data = []
                    for i in range(min(max_primers_to_show, len(primers))):
                        primer = primers[i]
                        comparison_data.append({
                            'Pair': i + 1,
                            'Product Size': f"{primer.product_size} bp",
                            'Forward Tm': f"{primer.forward_tm:.1f}°C",
                            'Reverse Tm': f"{primer.reverse_tm:.1f}°C",
                            'Penalty': f"{primer.penalty:.3f}",
                            'Forward Pos': f"{primer.forward_start:,}",
                            'Reverse Pos': f"{primer.reverse_start:,}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
            
            elif viz_type == "Coverage Heatmap":
                st.write("**Primer coverage density across the sequence**")
                
                col1, col2 = st.columns(2)
                with col1:
                    window_size = st.selectbox(
                        "Window size for coverage analysis:",
                        [50, 100, 200, 500, 1000],
                        index=1,
                        help="Larger windows show broader coverage patterns"
                    )
                with col2:
                    seq_len = len(st.session_state.current_sequence)
                    coverage_percent = (len(primers) * 50) / seq_len * 100  # Rough estimate
                    st.metric("Estimated Coverage", f"{coverage_percent:.1f}%")
                
                heatmap_fig = create_primer_coverage_heatmap(
                    st.session_state.current_sequence,
                    primers,
                    window_size
                )
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    st.write("**Coverage Analysis:**")
                    st.write("- Dark regions indicate high primer density")
                    st.write("- Light regions indicate low primer density")
                    st.write("- This helps identify over/under-represented sequence regions")
        
        # dsRNA-specific analysis
        if t7_enabled and primers:
            st.subheader("dsRNA Production Analysis")
            
            designer = PrimerDesigner()
            dsrna_analysis = []
            
            for i, primer in enumerate(primers):
                if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                    props = designer.calculate_dsrna_properties(primer, st.session_state.current_sequence)
                    if props:
                        dsrna_analysis.append({
                            'Pair': i + 1,
                            'dsRNA Length': f"{props.get('dsrna_length', 'N/A')} bp",
                            'GC Content': f"{props.get('dsrna_gc_content', 'N/A'):.1f}%",
                            'T7 Efficiency': props.get('transcription_efficiency', 'N/A'),
                            'Optimal Length': '✅' if props.get('optimal_length') else '❌',
                            'Moderate GC': '✅' if props.get('moderate_gc') else '❌',
                            'Estimated Yield': props.get('estimated_yield', 'N/A')
                        })
            
            if dsrna_analysis:
                dsrna_df = pd.DataFrame(dsrna_analysis)
                st.dataframe(dsrna_df, use_container_width=True)
                
                st.subheader("dsRNA Quality Summary")
                total_pairs = len(dsrna_analysis)
                optimal_length_count = sum(1 for row in dsrna_analysis if row['Optimal Length'] == '✅')
                moderate_gc_count = sum(1 for row in dsrna_analysis if row['Moderate GC'] == '✅')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Length", f"{optimal_length_count}/{total_pairs}")
                with col2:
                    st.metric("Moderate GC", f"{moderate_gc_count}/{total_pairs}")
                with col3:
                    quality_score = (optimal_length_count + moderate_gc_count) / (2 * total_pairs) * 100
                    st.metric("Quality Score", f"{quality_score:.0f}%")
        
        st.subheader("Statistics")
        if primers:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_tm = sum(p.forward_tm + p.reverse_tm for p in primers) / (2 * len(primers))
                if t7_enabled:
                    st.metric("Average Core Tm", f"{avg_tm:.1f}°C")
                else:
                    st.metric("Average Tm", f"{avg_tm:.1f}°C")
            
            with col2:
                avg_gc = sum(p.gc_content_f + p.gc_content_r for p in primers) / (2 * len(primers))
                if t7_enabled:
                    st.metric("Average Core GC%", f"{avg_gc:.1f}%")
                else:
                    st.metric("Average GC%", f"{avg_gc:.1f}%")
            
            with col3:
                avg_product = sum(p.product_size for p in primers) / len(primers)
                if t7_enabled:
                    st.metric("Average dsRNA Size", f"{avg_product:.0f} bp")
                else:
                    st.metric("Average Product Size", f"{avg_product:.0f} bp")
            
            with col4:
                best_penalty = min(p.penalty for p in primers)
                st.metric("Best Penalty Score", f"{best_penalty:.3f}")
        
        # Primer3 Complementarity Overview
        st.subheader("Primer3 Complementarity Overview")
        primer3_results = st.session_state.get('primer3_results', None)

        if primer3_results and primers:
            analyzer = Primer3ComplementarityAnalyzer()
            analyzer.store_primer3_results(primer3_results)
            
            # Collect data for all primers
            all_comp_data = []
            for i in range(len(primers)):
                comp_data = analyzer.extract_complementarity_data(i)
                if comp_data:
                    all_comp_data.append(comp_data)
            
            if all_comp_data:
                # Create summary chart
                fig_comp = analyzer.create_complementarity_summary_chart(all_comp_data)
                if fig_comp:
                    st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab4:
        st.header("Export Results")
        
        # Check session state validity fresh for this tab
        current_state_check = check_session_state_validity()
        
        
        if not current_state_check['has_primers']:
            st.info("No primers to export. Please design primers first.")
            st.stop()
        
        primers = OptimizedSessionManager.get_primers_optimized()
        t7_enabled = st.session_state.get('t7_dsrna_enabled', False)
        gene_targets_available = 'selected_gene_targets' in st.session_state
        
        if gene_targets_available:
            st.success("🧬 **Gene Target Information Available** - Export will include comprehensive gene target data")
        
        if t7_enabled:
            st.info("🧬 **T7 dsRNA Export Mode** - Export includes both full T7 primers and core sequences")
        
        st.subheader("Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Enhanced Excel", type="primary"):
                try:
                    excel_data = export_with_gene_targets(primers, "excel")
                    if excel_data:
                        filename = "enhanced_primer_results_with_targets.xlsx"
                        st.download_button(
                            label="Click to Download Enhanced Excel File",
                            data=excel_data,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"Error creating enhanced Excel: {e}")
        
        with col2:
            try:
                csv_data = export_with_gene_targets(primers, "csv")
                filename = "primer_results_with_gene_targets.csv"
                st.download_button(
                    label="📄 Download Enhanced CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error creating enhanced CSV: {e}")
        
        with col3:
            if gene_targets_available:
                # Export comprehensive gene target database
                try:
                    all_targets_df = create_comprehensive_gene_target_export()
                    targets_csv = all_targets_df.to_csv(index=False)
                    st.download_button(
                        label="🗃️ Download Complete Gene Database",
                        data=targets_csv,
                        file_name="comprehensive_gene_targets_database.csv",
                        mime="text/csv"
                    )
                except:
                    st.info("Gene database export not available")
        
        # Gene Target Statistics
        if gene_targets_available:
            st.subheader("📊 Gene Target Statistics")
            
            try:
                stats, category_counts, gene_category_counts = generate_gene_target_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Organisms", stats['Total Organisms'])
                with col2:
                    st.metric("Total Gene Targets", stats['Total Gene Targets'])
                with col3:
                    st.metric("High Priority Targets", stats['High Priority Targets'])
                with col4:
                    st.metric("Organism Categories", stats['Organism Categories'])
                
                # Category breakdown
                with st.expander("📈 Detailed Statistics", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Organism Categories:**")
                        for category, count in list(category_counts.items())[:10]:
                            st.write(f"• {category}: {count} organisms")
                    
                    with col2:
                        st.write("**Top Gene Categories:**")
                        for gene_cat, count in list(gene_category_counts.items())[:10]:
                            st.write(f"• {gene_cat}: {count} targets")
            except:
                st.info("Gene target statistics not available")
        
        # Original export functionality (fallback)
        st.subheader("📋 Standard Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Standard Excel", type="secondary"):
                excel_data = export_to_excel(primers)
                if excel_data:
                    filename = "t7_dsrna_primers.xlsx" if t7_enabled else "primer_results.xlsx"
                    st.download_button(
                        label="Click to Download Standard Excel File",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            try:
                data = []
                for i, primer in enumerate(primers):
                    # Handle existing primer pairs that don't have gene_target attribute
                    gene_target = getattr(primer, 'gene_target', 'Standard Design')
                    
                    if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                        row = {
                            'Primer_Pair': i + 1,
                            'Gene_Target': gene_target,
                            'Forward_T7_Sequence': primer.forward_seq,
                            'Reverse_T7_Sequence': primer.reverse_seq,
                            'Forward_Core_Sequence': primer.core_forward_seq,
                            'Reverse_Core_Sequence': primer.core_reverse_seq,
                            'Core_Forward_Tm': round(primer.forward_tm, 2),
                            'Core_Reverse_Tm': round(primer.reverse_tm, 2),
                            'dsRNA_Size': primer.product_size,
                            'Core_Forward_GC_Percent': round(primer.gc_content_f, 2),
                            'Core_Reverse_GC_Percent': round(primer.gc_content_r, 2),
                            'Forward_Start': primer.forward_start,
                            'Reverse_Start': primer.reverse_start,
                            'Penalty_Score': round(primer.penalty, 4),
                            'T7_Promoter': primer.t7_promoter_seq,
                            'Primer_Type': 'T7_dsRNA'
                        }
                    else:
                        row = {
                            'Primer_Pair': i + 1,
                            'Gene_Target': gene_target,
                            'Forward_Sequence': primer.forward_seq,
                            'Reverse_Sequence': primer.reverse_seq,
                            'Forward_Tm': round(primer.forward_tm, 2),
                            'Reverse_Tm': round(primer.reverse_tm, 2),
                            'Product_Size': primer.product_size,
                            'Forward_GC_Percent': round(primer.gc_content_f, 2),
                            'Reverse_GC_Percent': round(primer.gc_content_r, 2),
                            'Forward_Start': primer.forward_start,
                            'Reverse_Start': primer.reverse_start,
                            'Penalty_Score': round(primer.penalty, 4),
                            'Primer_Type': 'Standard'
                        }
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                
                filename = "t7_dsrna_primers.csv" if t7_enabled else "primer_results.csv"
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error creating CSV: {e}")
        
        st.subheader("Export Preview")
        try:
            if 'df' in locals():
                st.dataframe(df, use_container_width=True)
        except:
            st.warning("Could not create preview")
        
        st.subheader("Primer Ordering Format")
        
        if t7_enabled:
            st.write("**T7 dsRNA Primer Ordering Format** - Optimized for synthesis companies:")
            
            try:
                ordering_data = []
                for i, primer in enumerate(primers):
                    if hasattr(primer, 'has_t7_promoter') and primer.has_t7_promoter:
                        ordering_data.extend([
                            {
                                'Name': f"T7_Forward_Primer_{i+1}",
                                'Sequence': primer.forward_seq,
                                'Length': len(primer.forward_seq),
                                'Core_Tm': round(primer.forward_tm, 1),
                                'Notes': f"T7 promoter + {len(primer.core_forward_seq)}bp core"
                            },
                            {
                                'Name': f"T7_Reverse_Primer_{i+1}",
                                'Sequence': primer.reverse_seq,
                                'Length': len(primer.reverse_seq),
                                'Core_Tm': round(primer.reverse_tm, 1),
                                'Notes': f"T7 promoter + {len(primer.core_reverse_seq)}bp core"
                            }
                        ])
                
                ordering_df = pd.DataFrame(ordering_data)
                st.dataframe(ordering_df, use_container_width=True)
                
                ordering_csv = ordering_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download T7 Ordering Format",
                    data=ordering_csv,
                    file_name="t7_dsrna_primer_ordering.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error creating T7 ordering format: {e}")
        
        else:
            st.write("Format suitable for ordering from synthesis companies:")
            
            try:
                ordering_data = []
                for i, primer in enumerate(primers):
                    ordering_data.extend([
                        {
                            'Name': f"Forward_Primer_{i+1}",
                            'Sequence': primer.forward_seq,
                            'Length': len(primer.forward_seq),
                            'Tm': round(primer.forward_tm, 1)
                        },
                        {
                            'Name': f"Reverse_Primer_{i+1}",
                            'Sequence': primer.reverse_seq,
                            'Length': len(primer.reverse_seq),
                            'Tm': round(primer.reverse_tm, 1)
                        }
                    ])
                
                ordering_df = pd.DataFrame(ordering_data)
                st.dataframe(ordering_df, use_container_width=True)
                
                ordering_csv = ordering_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download Ordering Format",
                    data=ordering_csv,
                    file_name="primer_ordering.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error creating ordering format: {e}")
        
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🧬 Automated Primer Design Tool | Built with Streamlit</p>
            <p><small>Powered by Primer3, Biopython, and NCBI databases</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

class OptimizedSessionManager:
    """Optimized session state manager for large primer datasets"""
    
    @staticmethod
    def store_primers_optimized(primers: List[PrimerPair]):
        """Store primers in session state with compression for large datasets"""
        try:
            if len(primers) > 50:  # Compress large datasets
                compressed_data = OptimizedSessionManager.compress_session_data(primers)
                st.session_state.primers_designed = compressed_data
            else:
                st.session_state.primers_designed = primers
        except Exception as e:
            st.error(f"Error storing primers: {e}")
            st.session_state.primers_designed = primers
    
    @staticmethod
    def get_primers_optimized() -> List[PrimerPair]:
        """Retrieve primers from session state with decompression if needed"""
        try:
            primers_data = st.session_state.get('primers_designed', [])
            
            if isinstance(primers_data, dict) and primers_data.get('_compressed'):
                return OptimizedSessionManager.decompress_session_data(primers_data)
            elif isinstance(primers_data, list):
                return primers_data
            else:
                return []
        except Exception as e:
            st.error(f"Error retrieving primers: {e}")
            return []
    
    @staticmethod
    def compress_session_data(data):
        """Compress large datasets for session state"""
        try:
            compressed = gzip.compress(pickle.dumps(data))
            return {
                '_compressed': True,
                'data': base64.b64encode(compressed).decode('utf-8'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Compression failed: {e}")
            return data
    
    @staticmethod
    def decompress_session_data(compressed_data):
        """Decompress session state data"""
        try:
            if compressed_data.get('_compressed'):
                compressed = base64.b64decode(compressed_data['data'])
                return pickle.loads(gzip.decompress(compressed))
            else:
                return compressed_data
        except Exception as e:
            st.error(f"Decompression failed: {e}")
            return []
    
    @staticmethod
    def cleanup_session_state():
        """Clean up stale session state data"""
        try:
            # Remove old compressed data
            if 'primers_designed' in st.session_state:
                primers_data = st.session_state.primers_designed
                if isinstance(primers_data, dict) and primers_data.get('_compressed'):
                    timestamp = primers_data.get('timestamp')
                    if timestamp:
                        data_time = datetime.fromisoformat(timestamp)
                        if datetime.now() - data_time > timedelta(hours=24):
                            del st.session_state.primers_designed
        except Exception as e:
            pass  # Silent cleanup failure

if __name__ == "__main__":
    main()
