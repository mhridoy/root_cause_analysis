"""
Domain lexicons and label taxonomy for the autism/neurodevelopmental report analyzer.

Everything here is human-curated clinical vocabulary. It is used for:
  * deriving ground-truth labels from report filenames/folders,
  * masking diagnosis-revealing phrases for leakage-controlled evaluation,
  * extracting symptoms / root-cause signals / risk signals for explainable output.

IMPORTANT: This is decision-support vocabulary, not a diagnostic instrument.
"""

# ---------------------------------------------------------------------------
# Primary diagnosis classes the model is allowed to predict.
# The first six are well supported (>=50 examples each). "Other / Complex"
# is a catch-all for the diverse long-tail reports.
# ---------------------------------------------------------------------------
DISEASE_CLASSES = [
    "ADHD",
    "ASD",
    "Depression",
    "Dyslexia",
    "GAD",
    "OCD",
    "Other / Complex",
]

# Canonical disease -> regex-ish keyword fragments (lowercased substring match)
# Order matters for primary assignment: earlier = higher priority when a report
# lists several. ASD is prioritised because this product centres on autism.
DISEASE_KEYWORDS = {
    "ASD": [
        "autism spectrum disorder", "autism", "asd", "asperger",
        "pervasive developmental",
    ],
    "ADHD": [
        "adhd", "attention deficit", "attention-deficit", "hyperkinetic",
    ],
    "Dyslexia": [
        "dyslexia", "dyslexic", "specific learning disorder", "reading disorder",
    ],
    "OCD": [
        "ocd", "obsessive compulsive", "obsessive-compulsive",
    ],
    "GAD": [
        "gad", "generalized anxiety", "generalised anxiety",
    ],
    "Depression": [
        "major depressive", "depression", "depressive disorder", "mdd",
        "dysthymia",
    ],
}

# Co-occurring / secondary condition vocabulary (multi-label).
COOCCURRING_KEYWORDS = {
    "ADHD": ["adhd", "attention deficit", "attention-deficit"],
    "ASD": ["autism", "asd", "asperger"],
    "Anxiety": ["anxiety", "gad", "social anxiety", "separation anxiety", "panic"],
    "Depression": ["depress", "mdd", "low mood"],
    "OCD": ["ocd", "obsessive compulsive", "obsessive-compulsive"],
    "Learning Disorder": ["learning disorder", "dyslexia", "dysgraphia",
                          "dyscalculia", "specific learning"],
    "Speech / Language": ["speech", "language disorder", "language delay",
                          "communication disorder", "spcd"],
    "Intellectual Disability": ["intellectual disability", "borderline intellectual",
                               "global developmental delay", "gdd",
                               "cognitive deficit"],
    "Sensory Processing": ["sensory processing", "sensory avoidance",
                          "sensory seeking"],
    "Motor / Coordination": ["motor delay", "coordination disorder", "dcd",
                            "dyspraxia", "fine motor", "gross motor"],
    "ODD / Conduct": ["oppositional defiant", "odd", "conduct disorder",
                     "dmdd", "disruptive mood"],
    "Tics / Tourette": ["tourette", "tic disorder", "tics"],
    "PTSD / Trauma": ["ptsd", "post-traumatic", "trauma", "adjustment disorder"],
}

# ---------------------------------------------------------------------------
# Root-cause / contributing-factor groups (model target #2) and the lexical
# signals that point at each. These describe *contributing factors* discussed
# in the report, NOT a claim of etiology.
# ---------------------------------------------------------------------------
ROOT_CAUSE_GROUPS = {
    "Neurodevelopmental Differences": [
        "neurodevelopmental", "developmental delay", "milestone",
        "early development", "genetic", "hereditary", "family history of autism",
        "brain development", "neurological", "congenital",
    ],
    "Anxiety / Stress Reactivity": [
        "anxiety", "worry", "stress", "fear", "panic", "nervous",
        "hypervigilance", "rumination", "avoidance",
    ],
    "Mood / Emotional Vulnerability": [
        "low mood", "depress", "sadness", "emotional dysregulation",
        "irritability", "hopeless", "tearful", "self-esteem", "mood",
    ],
    "Learning / Cognitive Processing": [
        "learning", "reading", "writing", "phonolog", "working memory",
        "processing speed", "cognitive", "academic", "comprehension",
    ],
    "Sensory / Motor": [
        "sensory", "motor", "coordination", "tactile", "auditory sensitivity",
        "propriocept", "vestibular",
    ],
    "Sleep / Physiological Regulation": [
        "sleep", "appetite", "fatigue", "circadian", "insomnia", "energy",
    ],
    "Social / Environmental Pressure": [
        "peer", "bullying", "social pressure", "family conflict", "school refusal",
        "academic pressure", "psychosocial", "social isolation", "parenting",
    ],
}

# ---------------------------------------------------------------------------
# Symptom phrase lexicon for evidence extraction (grouped, lowercased).
# ---------------------------------------------------------------------------
SYMPTOM_PHRASES = [
    # social / communication
    "poor eye contact", "limited eye contact", "social withdrawal",
    "difficulty with social interaction", "social communication difficulties",
    "restricted interests", "repetitive behaviour", "repetitive behavior",
    "repetitive movements", "stereotyped", "echolalia", "literal interpretation",
    "difficulty with transitions", "insistence on sameness", "rigid routines",
    # attention / executive
    "inattention", "difficulty sustaining attention", "easily distracted",
    "hyperactivity", "impulsivity", "fidgeting", "restlessness",
    "poor organization", "forgetfulness", "difficulty following instructions",
    "executive function", "poor working memory",
    # mood / anxiety
    "persistent sadness", "low mood", "loss of interest", "anhedonia",
    "excessive worry", "rumination", "irritability", "emotional withdrawal",
    "panic", "fearfulness", "hopelessness", "tearfulness", "self-critical",
    # ocd
    "obsessions", "compulsions", "intrusive thoughts", "checking behaviour",
    "ritual", "contamination fear",
    # learning
    "reading difficulties", "spelling difficulties", "phonological",
    "letter reversal", "slow reading", "poor reading fluency",
    "writing difficulties", "academic struggles", "academic decline",
    # regulation / sensory / sleep
    "sensory sensitivity", "sensory seeking", "meltdowns", "emotional dysregulation",
    "sleep disturbance", "difficulty initiating sleep", "fragmented sleep",
    "appetite change", "fatigue", "motor delay", "coordination difficulties",
    "speech delay", "language delay",
]

# ---------------------------------------------------------------------------
# Diagnosis-revealing phrases to MASK for the leakage-controlled evaluation.
# These are the literal statements a clinician writes that announce the answer.
# When masked, the model must rely on described symptoms/findings instead.
# ---------------------------------------------------------------------------
LEAKAGE_PATTERNS = [
    r"primary diagnosis[^.\n]*",
    r"provisional diagnosis[^.\n]*",
    r"diagnostic impression[^.\n]*",
    r"final diagnosis[^.\n]*",
    r"diagnosis[s]?\s*[:\-][^.\n]*",
    r"clinical diagnosis[^.\n]*",
    r"dsm[\- ]?5[^.\n]*",
    r"icd[\- ]?1[01][^.\n]*",
    r"meets criteria for[^.\n]*",
    r"consistent with a diagnosis of[^.\n]*",
    r"diagnosed with[^.\n]*",
]
# Plus every disease keyword string gets masked token-wise (handled in textproc).

# ---------------------------------------------------------------------------
# Risk signals -> elevate risk level when present.
# ---------------------------------------------------------------------------
RISK_HIGH_SIGNALS = [
    "self-harm", "self harm", "suicidal", "suicide", "harm to self",
    "harm to others", "aggression towards", "safety risk", "crisis",
    "hospitalization", "severe impairment", "regression", "non-verbal",
    "nonverbal", "elopement", "abuse", "neglect",
]
RISK_MODERATE_SIGNALS = [
    "significant impairment", "functional impairment", "school refusal",
    "social isolation", "academic decline", "level 2", "level 3",
    "moderate to severe", "comorbid", "multiple comorbid",
]
