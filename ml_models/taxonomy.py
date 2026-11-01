"""
=============================================================================
CROP LEAF DISEASE TAXONOMY & AGRONOMIC DOMAIN DEFINITIONS
Project: Agritech Self-Supervised Cross-Domain Vision Transformers
         for Fine-Grained Crop Leaf Disease Severity Estimation
=============================================================================
Defines the botanical crop taxonomy, disease classes, pathogen details,
pathology profiles, and severity tier calculations for all supported crops:
- Tomato (Solanum lycopersicum)
- Potato (Solanum tuberosum)
- Rice (Oryza sativa)
- Wheat (Triticum aestivum)
- Maize (Zea mays)
=============================================================================
"""

# Supported target crops
CROPS = ["Tomato", "Potato", "Rice", "Wheat", "Maize"]

# Comprehensive crop disease taxonomy
TAXONOMY = {
    "Tomato": [
        {
            "id": 0,
            "disease": "Healthy Leaf",
            "pathogen": "None (Healthy tissue)",
            "type": "Healthy",
            "base_severity": 0.0,
            "severity_range": (0.0, 0.0),
            "typical_confidence": 0.96,
            "symptoms": "Uniform chlorophyll distribution, intact cuticle, no foliar lesions.",
            "management": "Maintain regular agronomic scouting and balanced N-P-K fertilization."
        },
        {
            "id": 1,
            "disease": "Early Blight",
            "pathogen": "Alternaria solani",
            "type": "Fungal",
            "base_severity": 22.4,
            "severity_range": (5.0, 45.0),
            "typical_confidence": 0.92,
            "symptoms": "Concentric dark brown rings ('target-board' pattern) surrounded by chlorotic yellow halos.",
            "management": "Apply protectant fungicides (Mancozeb, Chlorothalonil) and implement 3-year crop rotation."
        },
        {
            "id": 2,
            "disease": "Late Blight",
            "pathogen": "Phytophthora infestans",
            "type": "Oomycete",
            "base_severity": 36.8,
            "severity_range": (15.0, 85.0),
            "typical_confidence": 0.94,
            "symptoms": "Water-soaked dark lesions rapidly expanding into necrotic brown-black foliar patches with pale green borders.",
            "management": "Immediate curative treatment with systemic fungicides (Metalaxyl, Cymoxanil) and destroy infected cull piles."
        },
        {
            "id": 3,
            "disease": "Septoria Leaf Spot",
            "pathogen": "Septoria lycopersici",
            "type": "Fungal",
            "base_severity": 16.5,
            "severity_range": (5.0, 35.0),
            "typical_confidence": 0.91,
            "symptoms": "Numerous circular water-soaked spots with gray or tan centers and dark brown margins; tiny black pycnidia visible.",
            "management": "Copper-based bactericides/fungicides, drip irrigation to prevent foliar splashing, stake plants."
        }
    ],
    "Potato": [
        {
            "id": 4,
            "disease": "Healthy Leaf",
            "pathogen": "None (Healthy tissue)",
            "type": "Healthy",
            "base_severity": 0.0,
            "severity_range": (0.0, 0.0),
            "typical_confidence": 0.97,
            "symptoms": "Vibrant emerald green foliage without chlorosis or necrosis.",
            "management": "Optimal irrigation scheduling and certified disease-free seed tuber selection."
        },
        {
            "id": 5,
            "disease": "Early Blight",
            "pathogen": "Alternaria solani",
            "type": "Fungal",
            "base_severity": 21.4,
            "severity_range": (8.0, 40.0),
            "typical_confidence": 0.91,
            "symptoms": "Brown to black angular lesions restricted by leaf veins, developing characteristic concentric target rings.",
            "management": "Fungicide sprays (Azoxystrobin or Chlorothalonil) starting before canopy closure; avoid overhead irrigation."
        },
        {
            "id": 6,
            "disease": "Late Blight",
            "pathogen": "Phytophthora infestans",
            "type": "Oomycete",
            "base_severity": 41.5,
            "severity_range": (20.0, 90.0),
            "typical_confidence": 0.95,
            "symptoms": "Irregular necrotic brown lesions surrounded by lime-green margins; white mold on leaf underside under high humidity.",
            "management": "Urgent preventive and curative fungicide applications (Dimethomorph, Fluopicolide); eliminate volunteer potatoes."
        },
        {
            "id": 7,
            "disease": "Common Scab",
            "pathogen": "Streptomyces scabies",
            "type": "Bacterial / Actinomycete",
            "base_severity": 26.8,
            "severity_range": (10.0, 50.0),
            "typical_confidence": 0.88,
            "symptoms": "Foliar stunting and corky raised or pitted lesions affecting tuber and petiole tissue.",
            "management": "Maintain soil pH below 5.2 during tuber initiation; irrigate to keep soil moist for 4-6 weeks after emergence."
        }
    ],
    "Rice": [
        {
            "id": 8,
            "disease": "Healthy Leaf",
            "pathogen": "None (Healthy tissue)",
            "type": "Healthy",
            "base_severity": 0.0,
            "severity_range": (0.0, 0.0),
            "typical_confidence": 0.98,
            "symptoms": "Uniform slender upright leaves with clear green venation and absence of foliar spotting.",
            "management": "Balanced silica and potassium supplementation to reinforce cuticle integrity."
        },
        {
            "id": 9,
            "disease": "Brown Spot",
            "pathogen": "Bipolaris oryzae (Cochliobolus miyabeanus)",
            "type": "Fungal",
            "base_severity": 28.5,
            "severity_range": (10.0, 60.0),
            "typical_confidence": 0.89,
            "symptoms": "Oval to circular brown spots with yellow halos across the leaf blade; coalescing to cause leaf blade blighting.",
            "management": "Soil nutrient correction (especially potassium and zinc deficiency); seed treatment with Carbendazim."
        },
        {
            "id": 10,
            "disease": "Bacterial Leaf Blight",
            "pathogen": "Xanthomonas oryzae pv. oryzae",
            "type": "Bacterial",
            "base_severity": 46.2,
            "severity_range": (20.0, 85.0),
            "typical_confidence": 0.93,
            "symptoms": "Water-soaked lesions on leaf margins turning translucent yellow-white, waving along the edge of the leaf.",
            "management": "Plant resistant cultivars (e.g., IR64 or Xa21 gene lines), avoid excessive nitrogen fertilizers, drain paddy temporarily."
        },
        {
            "id": 11,
            "disease": "Leaf Blast",
            "pathogen": "Magnaporthe oryzae (Pyricularia oryzae)",
            "type": "Fungal",
            "base_severity": 37.0,
            "severity_range": (15.0, 75.0),
            "typical_confidence": 0.92,
            "symptoms": "Spindle-shaped / diamond-shaped lesions with grayish-white centers and brownish-red borders.",
            "management": "Foliar application of Tricyclazole or Isoprothiolane at onset; avoid late-stage nitrogen overload."
        }
    ],
    "Wheat": [
        {
            "id": 12,
            "disease": "Healthy Leaf",
            "pathogen": "None (Healthy tissue)",
            "type": "Healthy",
            "base_severity": 0.0,
            "severity_range": (0.0, 0.0),
            "typical_confidence": 0.97,
            "symptoms": "Linear erect leaf blades with vibrant green pigmentation and no visible pustules.",
            "management": "Regular field scouting during tillering and stem elongation stages."
        },
        {
            "id": 13,
            "disease": "Stripe Rust",
            "pathogen": "Puccinia striiformis f. sp. tritici",
            "type": "Fungal",
            "base_severity": 49.2,
            "severity_range": (15.0, 90.0),
            "typical_confidence": 0.95,
            "symptoms": "Characteristic linear stripes of bright yellow-orange urediniospores aligned parallel to leaf veins.",
            "management": "Timely application of triazole fungicides (Propiconazole, Tebuconazole) and planting Yr-resistant cultivars."
        },
        {
            "id": 14,
            "disease": "Leaf Rust",
            "pathogen": "Puccinia triticina",
            "type": "Fungal",
            "base_severity": 32.1,
            "severity_range": (10.0, 65.0),
            "typical_confidence": 0.91,
            "symptoms": "Small, round-to-oval orange-brown pustules scattered randomly across the upper leaf surface.",
            "management": "Deploy multi-gene adult plant resistance (APR); spray with systemic DMI fungicides if flag leaf is threatened."
        },
        {
            "id": 15,
            "disease": "Powdery Mildew",
            "pathogen": "Blumeria graminis f. sp. tritici",
            "type": "Fungal",
            "base_severity": 21.0,
            "severity_range": (5.0, 45.0),
            "typical_confidence": 0.90,
            "symptoms": "Fluffy white-to-gray cottony fungal mycelial patches on lower leaves, gradually forming black chasmothecia.",
            "management": "Ensure adequate air circulation through appropriate seed rates; apply sulfur or fenpropimorph if severe."
        }
    ],
    "Maize": [
        {
            "id": 16,
            "disease": "Healthy Leaf",
            "pathogen": "None (Healthy tissue)",
            "type": "Healthy",
            "base_severity": 0.0,
            "severity_range": (0.0, 0.0),
            "typical_confidence": 0.96,
            "symptoms": "Broad glossy arching leaves without foliar blight strips or rust pustules.",
            "management": "Balanced fertilization, maintaining soil drainage and optimal plant population density."
        },
        {
            "id": 17,
            "disease": "Northern Leaf Blight",
            "pathogen": "Exserohilum turcicum",
            "type": "Fungal",
            "base_severity": 43.1,
            "severity_range": (15.0, 80.0),
            "typical_confidence": 0.92,
            "symptoms": "Long, elliptical, grayish-green or tan cigar-shaped lesions (2.5 - 15 cm) running parallel to leaf veins.",
            "management": "Plant resistant hybrids with Ht resistance genes; deep tillage to bury crop residues; strobilurin/triazole sprays."
        },
        {
            "id": 18,
            "disease": "Common Rust",
            "pathogen": "Puccinia sorghi",
            "type": "Fungal",
            "base_severity": 25.6,
            "severity_range": (8.0, 50.0),
            "typical_confidence": 0.93,
            "symptoms": "Golden-brown to cinnamon-brown powdery pustules scattered on both upper and lower leaf surfaces.",
            "management": "Use Rp resistant hybrids; monitor temperature (cool, moist conditions favor spread); apply foliar fungicides early."
        },
        {
            "id": 19,
            "disease": "Gray Leaf Spot",
            "pathogen": "Cercospora zeae-maydis",
            "type": "Fungal",
            "base_severity": 34.0,
            "severity_range": (10.0, 65.0),
            "typical_confidence": 0.90,
            "symptoms": "Distinctly rectangular, pale brown to gray lesions strictly delimited by the leaf veins.",
            "management": "Crop rotation with non-host crops (soybean); utilize hybrids with quantitative resistance; residue management."
        }
    ]
}

# Flattened list of all 20 classes
ALL_CLASSES = []
for crop, diseases in TAXONOMY.items():
    for d in diseases:
        ALL_CLASSES.append(f"{crop}___{d['disease'].replace(' ', '_')}")

NUM_CLASSES = len(ALL_CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}

def get_disease_info(crop: str, disease_name: str) -> dict:
    """Retrieve full disease profile by crop and disease name."""
    crop_diseases = TAXONOMY.get(crop, TAXONOMY["Tomato"])
    for d in crop_diseases:
        if d["disease"].lower() == disease_name.lower():
            return d
    # Return default or first
    return crop_diseases[0]

def calculate_severity_level(severity_percentage: float) -> str:
    """
    Standardized agronomic severity classification tier:
    - 0.0%: Healthy
    - (0.0%, 20.0%]: Mild
    - (20.0%, 50.0%]: Moderate
    - > 50.0%: Severe
    """
    if severity_percentage <= 0.0:
        return "Healthy"
    elif severity_percentage <= 20.0:
        return "Mild"
    elif severity_percentage <= 50.0:
        return "Moderate"
    else:
        return "Severe"

