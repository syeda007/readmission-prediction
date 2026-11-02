"""
ReadmitAI - Pydantic Request & Response Schemas
Defines structured patient data models and strict typing for EHR API integration.
"""

from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.clinical_codes import is_known_icd9

# Allowed category values (exactly the levels the model was trained on)
AgeBracket = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
Gender = Literal["Female", "Male"]
Race = Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
A1CResult = Literal["None", "Norm", ">7", ">8"]
GlucoseResult = Literal["None", "Norm", ">200", ">300"]
MedStatus = Literal["No", "Steady", "Up", "Down"]
CodeId = Union[int, str]


class ClinicalEncounterFields(BaseModel):
    """
    Encounter-level model inputs known at discharge (everything except demographics).
    Fields that drive the prediction are required; nothing clinically meaningful is silently defaulted.
    Unrecorded optional fields are passed to the model as missing, which it was trained to handle.
    """
    # Encounter details
    admission_type_id: CodeId = Field(..., description="Admission type ID (1=Emergency, 2=Urgent, 3=Elective)")
    discharge_disposition_id: CodeId = Field(..., description="Discharge disposition ID (1=Home, 3=SNF, 6=Home health)")
    admission_source_id: CodeId = Field(..., description="Admission source ID (7=Emergency Room, 1=Physician referral)")
    medical_specialty: Optional[str] = Field(None, description="Admitting or attending medical specialty (omit if not recorded)")

    # Clinical utilization & procedures
    num_lab_procedures: int = Field(..., ge=1, le=150, description="Number of lab procedures performed")
    num_procedures: int = Field(..., ge=0, le=10, description="Number of non-lab operating or bedside procedures")
    num_medications: int = Field(..., ge=1, le=100, description="Total distinct medications administered")
    number_outpatient: int = Field(..., ge=0, le=50, description="Outpatient visits in the preceding year")
    number_emergency: int = Field(..., ge=0, le=50, description="Emergency department visits in the preceding year")
    number_inpatient: int = Field(..., ge=0, le=25, description="Inpatient hospital admissions in the preceding year")
    number_diagnoses: int = Field(..., ge=1, le=16, description="Number of diagnoses recorded on the encounter")

    # Diagnostic codes (ICD-9)
    diag_1: str = Field(..., min_length=1, description="Primary ICD-9 diagnosis code (e.g. 428, 250.02, V57)")
    diag_2: Optional[str] = Field(None, description="Secondary ICD-9 diagnosis code")
    diag_3: Optional[str] = Field(None, description="Tertiary ICD-9 diagnosis code")

    # Glycemic monitoring ("None" = test not performed, which is itself informative)
    max_glu_serum: GlucoseResult = Field("None", description="Max glucose serum result (None = not measured)")
    A1Cresult: A1CResult = Field("None", description="HbA1c test result (None = not measured)")

    # Medications ("No" = not prescribed)
    metformin: MedStatus = "No"
    repaglinide: MedStatus = "No"
    nateglinide: MedStatus = "No"
    chlorpropamide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    tolbutamide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    acarbose: MedStatus = "No"
    miglitol: MedStatus = "No"
    troglitazone: MedStatus = "No"
    tolazamide: MedStatus = "No"
    insulin: MedStatus = "No"
    glyburide_metformin: MedStatus = Field("No", alias="glyburide-metformin")
    glipizide_metformin: MedStatus = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedStatus = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedStatus = Field("No", alias="metformin-rosiglitazone")
    change: Literal["No", "Ch"] = Field(..., description="Change in diabetic medication (No, Ch)")
    diabetesMed: Literal["Yes", "No"] = Field(..., description="Diabetic medication prescribed (Yes, No)")

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("medical_specialty", "diag_2", "diag_3", mode="before")
    @classmethod
    def blank_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("diag_1", "diag_2", "diag_3")
    @classmethod
    def known_icd9_code(cls, v):
        if v is None:
            return v
        v = v.strip().upper()
        if not is_known_icd9(v):
            raise ValueError(f"'{v}' is not a recognised ICD-9-CM diagnosis code (e.g. 428, 250.02, V45.81)")
        return v


class PatientEncounterInput(ClinicalEncounterFields):
    """
    Patient encounter payload available at or before hospital discharge.
    """
    # Demographics
    race: Optional[Race] = Field(None, description="Race / Ethnicity (omit if not recorded)")
    gender: Gender = Field(..., description="Gender (Female, Male)")
    age: AgeBracket = Field(..., description="Age bracket (e.g. [60-70), [70-80))")
    payer_code: Optional[str] = Field(None, description="Payer code (e.g. MC, MD, HM, UN, BC); omit if not recorded")
    time_in_hospital: int = Field(..., ge=1, le=14, description="Length of hospital stay in days (1-14)")

    # Supplemental qualitative clinical observations (separate from ML features)
    patient_notes: Optional[str] = Field(None, description="Supplemental qualitative clinical notes for EHR record")
    symptoms_reported: Optional[List[str]] = Field(None, description="Qualitative symptom observations")

    @field_validator("race", "payer_code", mode="before")
    @classmethod
    def blank_demographic_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v


class FactorAttribution(BaseModel):
    raw_feature: str
    feature_name: str
    description: str
    shap_value: float
    impact: str


class ExplanationPayload(BaseModel):
    risk_increasing_factors: List[FactorAttribution]
    risk_decreasing_factors: List[FactorAttribution]
    clinical_disclaimer: str


class PredictionResponse(BaseModel):
    is_eligible: bool
    readmission_probability: Optional[float] = None
    readmission_risk_percentage: Optional[str] = None
    requires_review: Optional[bool] = None
    risk_level: Optional[str] = None
    clinical_recommendation: Optional[str] = None
    decision_threshold: Optional[float] = None
    model_name: str
    model_version: str
    explanation: Optional[ExplanationPayload] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    prediction_target: str
    dataset_scope: str
    operational_threshold: float
    framework: str
    training_timestamp: str
