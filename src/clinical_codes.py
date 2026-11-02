"""
ReadmitAI - Clinical Code Dictionaries
Human-readable labels for the coded UCI fields, shared by the SHAP explanations
and the EHR interface dropdowns. Only codes that occur in the training data are listed.
"""

import json
import os
import re
from functools import lru_cache
from typing import Dict, Optional

ICD9_DICTIONARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "icd9_descriptions.json"))

ADMISSION_TYPE_LABELS = {
    "1": "Emergency",
    "2": "Urgent",
    "3": "Elective",
    "4": "Newborn",
    "5": "Not available",
    "6": "Not recorded (NULL)",
    "7": "Trauma center",
    "8": "Not mapped",
}

ADMISSION_SOURCE_LABELS = {
    "1": "Physician referral",
    "2": "Clinic referral",
    "3": "HMO referral",
    "4": "Transfer from a hospital",
    "5": "Transfer from a skilled nursing facility (SNF)",
    "6": "Transfer from another health care facility",
    "7": "Emergency room",
    "8": "Court / law enforcement",
    "9": "Not available",
    "10": "Transfer from critical access hospital",
    "17": "Not recorded (NULL)",
    "20": "Not mapped",
    "22": "Transfer from hospital inpatient (same facility, separate claim)",
    "25": "Transfer from ambulatory surgery center",
}

DISCHARGE_DISPOSITION_LABELS = {
    "1": "Discharged to home (self-care)",
    "2": "Transferred to another short-term hospital",
    "3": "Transferred to skilled nursing facility (SNF)",
    "4": "Transferred to intermediate care facility (ICF)",
    "5": "Transferred to another inpatient care institution",
    "6": "Home with home health service",
    "7": "Left against medical advice (AMA)",
    "8": "Home under care of home IV provider",
    "9": "Admitted as inpatient to this hospital",
    "15": "Transferred within this institution to swing bed",
    "16": "Referred to another institution for outpatient services",
    "17": "Referred to this institution for outpatient services",
    "18": "Not recorded (NULL)",
    "22": "Transferred to rehabilitation facility",
    "23": "Transferred to long-term care hospital",
    "24": "Transferred to Medicaid-only nursing facility",
    "25": "Not mapped",
    "27": "Transferred to federal health care facility",
    "28": "Transferred to psychiatric hospital",
}

# Death and hospice dispositions: excluded from the model cohort
INELIGIBLE_DISCHARGE_LABELS = {
    "11": "Expired",
    "13": "Hospice / home",
    "14": "Hospice / medical facility",
    "19": "Expired at home (hospice)",
    "20": "Expired in a medical facility (hospice)",
    "21": "Expired, place unknown (hospice)",
}

# The UCI dataset never defines its payer codes. These are the commonly used interpretations
# (Strack et al. 2014 names Medicare, Blue Cross/Blue Shield and self-pay as examples);
# codes without a reliable interpretation say so rather than guessing.
PAYER_LABELS = {
    "MC": "Medicare",
    "MD": "Medicaid",
    "HM": "HMO (health maintenance organization)",
    "BC": "Blue Cross / Blue Shield",
    "SP": "Self-pay",
    "WC": "Workers' compensation",
    "CH": "CHAMPUS / TRICARE (military)",
    "OG": "Other government program",
    "SI": "Self-insured",
    "OT": "Other payer",
    "UN": "Likely UnitedHealthcare",
    "CP": "Likely commercial / private insurance",
    "CM": "Likely commercial insurance",
    "PO": "Likely PPO (preferred provider organization)",
    "DM": "Undocumented payer",
    "MP": "Undocumented payer",
}
PAYER_CODES = list(PAYER_LABELS)

RACE_LABELS = {
    "Caucasian": "White / Caucasian",
    "AfricanAmerican": "Black / African American",
    "Hispanic": "Hispanic",
    "Asian": "Asian",
    "Other": "Other",
}
RACES = list(RACE_LABELS)

# Every medical_specialty value in the UCI data, written out (the raw values run words together).
# Only 9 are frequent enough to keep their own category in the model; the rest are grouped as "Other".
SPECIALTY_LABELS = {
    "AllergyandImmunology": "Allergy and immunology",
    "Anesthesiology": "Anesthesiology",
    "Anesthesiology-Pediatric": "Pediatric anesthesiology",
    "Cardiology": "Cardiology",
    "Cardiology-Pediatric": "Pediatric cardiology",
    "DCPTEAM": "DCP team",
    "Dentistry": "Dentistry",
    "Dermatology": "Dermatology",
    "Emergency/Trauma": "Emergency medicine / trauma",
    "Endocrinology": "Endocrinology",
    "Endocrinology-Metabolism": "Endocrinology and metabolism",
    "Family/GeneralPractice": "Family medicine / general practice",
    "Gastroenterology": "Gastroenterology",
    "Gynecology": "Gynecology",
    "Hematology": "Hematology",
    "Hematology/Oncology": "Hematology / oncology",
    "Hospitalist": "Hospitalist",
    "InfectiousDiseases": "Infectious diseases",
    "InternalMedicine": "Internal medicine",
    "Nephrology": "Nephrology",
    "Neurology": "Neurology",
    "Neurophysiology": "Neurophysiology",
    "Obsterics&Gynecology-GynecologicOnco": "Gynecologic oncology",
    "Obstetrics": "Obstetrics",
    "ObstetricsandGynecology": "Obstetrics and gynecology",
    "Oncology": "Oncology",
    "Ophthalmology": "Ophthalmology",
    "Orthopedics": "Orthopedics",
    "Orthopedics-Reconstructive": "Orthopedics (reconstructive)",
    "Osteopath": "Osteopathic medicine",
    "Otolaryngology": "Otolaryngology (ENT)",
    "OutreachServices": "Outreach services",
    "Pathology": "Pathology",
    "Pediatrics": "Pediatrics",
    "Pediatrics-AllergyandImmunology": "Pediatric allergy and immunology",
    "Pediatrics-CriticalCare": "Pediatric critical care",
    "Pediatrics-EmergencyMedicine": "Pediatric emergency medicine",
    "Pediatrics-Endocrinology": "Pediatric endocrinology",
    "Pediatrics-Hematology-Oncology": "Pediatric hematology / oncology",
    "Pediatrics-InfectiousDiseases": "Pediatric infectious diseases",
    "Pediatrics-Neurology": "Pediatric neurology",
    "Pediatrics-Pulmonology": "Pediatric pulmonology",
    "Perinatology": "Perinatology",
    "PhysicalMedicineandRehabilitation": "Physical medicine and rehabilitation",
    "PhysicianNotFound": "Physician not found",
    "Podiatry": "Podiatry",
    "Proctology": "Proctology",
    "Psychiatry": "Psychiatry",
    "Psychiatry-Addictive": "Addiction psychiatry",
    "Psychiatry-Child/Adolescent": "Child and adolescent psychiatry",
    "Psychology": "Psychology",
    "Pulmonology": "Pulmonology",
    "Radiologist": "Radiology",
    "Radiology": "Radiology",
    "Resident": "Resident physician",
    "Rheumatology": "Rheumatology",
    "Speech": "Speech therapy",
    "SportsMedicine": "Sports medicine",
    "Surgeon": "Surgery",
    "Surgery-Cardiovascular": "Cardiovascular surgery",
    "Surgery-Cardiovascular/Thoracic": "Cardiothoracic surgery",
    "Surgery-Colon&Rectal": "Colorectal surgery",
    "Surgery-General": "General surgery",
    "Surgery-Maxillofacial": "Maxillofacial surgery",
    "Surgery-Neuro": "Neurosurgery",
    "Surgery-Pediatric": "Pediatric surgery",
    "Surgery-Plastic": "Plastic surgery",
    "Surgery-PlasticwithinHeadandNeck": "Plastic surgery (head and neck)",
    "Surgery-Thoracic": "Thoracic surgery",
    "Surgery-Vascular": "Vascular surgery",
    "SurgicalSpecialty": "Surgical specialty",
    "Urology": "Urology",
    "Other": "Other specialty",
}
GENDERS = ["Female", "Male"]
AGE_BRACKETS = [f"[{lo}-{lo + 10})" for lo in range(0, 100, 10)]

LAB_A1C = {"None": "Not measured", "Norm": "Normal", ">7": "> 7%", ">8": "> 8%"}
LAB_GLUCOSE = {"None": "Not measured", "Norm": "Normal", ">200": "> 200 mg/dL", ">300": "> 300 mg/dL"}
MED_STATUS = {"No": "Not prescribed", "Steady": "Steady dose", "Up": "Dose increased", "Down": "Dose decreased"}

DIABETES_MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
]

NUMERIC_LABELS = {
    "number_inpatient": "Prior inpatient admissions (past year)",
    "number_emergency": "Prior emergency visits (past year)",
    "number_outpatient": "Prior outpatient visits (past year)",
    "time_in_hospital": "Length of stay (days)",
    "num_medications": "Medications given during stay",
    "num_lab_procedures": "Lab tests performed",
    "num_procedures": "Non-lab procedures performed",
    "number_diagnoses": "Diagnoses recorded",
}

DIAGNOSIS_SLOT_LABELS = {
    "diag_1_group": "Primary diagnosis",
    "diag_2_group": "Secondary diagnosis",
    "diag_3_group": "Additional diagnosis",
}


def specialty_label(value: Optional[str]) -> str:
    """Readable name for a UCI medical_specialty value, e.g. 'Surgery-Colon&Rectal' -> 'Colorectal surgery'."""
    if not value:
        return "Not recorded"
    if value in SPECIALTY_LABELS:
        return SPECIALTY_LABELS[value]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).replace("-", " - ")


def payer_label(code: Optional[str]) -> str:
    if not code:
        return "Not recorded"
    return f"{PAYER_LABELS.get(code, 'Unknown payer')} ({code})"


def race_label(value: Optional[str]) -> str:
    return RACE_LABELS.get(value, value) if value else "Not recorded"


@lru_cache(maxsize=1)
def _icd9_dictionary() -> Dict[str, str]:
    with open(ICD9_DICTIONARY_PATH, encoding="utf-8") as f:
        return json.load(f)["codes"]


def normalize_icd9(code: str) -> Optional[str]:
    """
    Converts a UCI-style code to CMS format: '250.83' -> '25083', '8' -> '008', 'V57' -> 'V57'.
    Returns None if the text is not shaped like an ICD-9-CM diagnosis code.
    """
    text = str(code).strip().upper()
    match = re.fullmatch(r"([VE]?)(\d{1,3})(?:\.(\d{1,2}))?", text)
    if not match:
        return None
    prefix, head, tail = match.groups()
    width = {"": 3, "V": 2, "E": 3}[prefix]
    if len(head) > width:
        return None
    return f"{prefix}{head.zfill(width)}{tail or ''}"


def is_known_icd9(code: Optional[str]) -> bool:
    """True if the code exists in ICD-9-CM, as a billable code or as a category/subcategory."""
    key = normalize_icd9(code) if code else None
    return key is not None and key in _icd9_dictionary()


def describe_icd9(code: Optional[str]) -> Optional[str]:
    """
    Description for an ICD-9-CM diagnosis code. Falls back to the closest parent category
    (e.g. an unknown 5th digit resolves to the 4-digit subcategory). None if unknown.
    """
    if not code:
        return None
    key = normalize_icd9(code)
    if key is None:
        return None
    dictionary = _icd9_dictionary()
    minimum = 4 if key.startswith("E") else 3
    while len(key) >= minimum:
        if key in dictionary:
            return dictionary[key]
        key = key[:-1]
    return None
