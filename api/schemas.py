"""
ReadmitAI - Pydantic Request & Response Schemas
Defines structured patient data models and strict typing for EHR API integration.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PatientEncounterInput(BaseModel):
    """
    Patient encounter payload available at or before hospital discharge.
    """
    # Demographics
    race: Optional[str] = Field("Caucasian", description="Race / Ethnicity")
    gender: Optional[str] = Field("Female", description="Gender (Female, Male, Unknown/Invalid)")
    age: Optional[str] = Field("[70-80)", description="Age bracket (e.g. [60-70), [70-80))")

    # Encounter details
    admission_type_id: Union[int, str] = Field(1, description="Admission type ID (1=Emergency, 2=Urgent, 3=Elective)")
    discharge_disposition_id: Union[int, str] = Field(1, description="Discharge disposition ID (1=Home, 3=SNF, 6=Home health)")
    admission_source_id: Union[int, str] = Field(7, description="Admission source ID (7=Emergency Room, 1=Physician referral)")
    time_in_hospital: int = Field(3, ge=1, le=14, description="Length of hospital stay in days (1-14)")
    payer_code: Optional[str] = Field("MC", description="Payer code (e.g. MC, MD, HM, UN, BC)")
    medical_specialty: Optional[str] = Field("InternalMedicine", description="Admitting or attending medical specialty")

    # Clinical utilization & procedures
    num_lab_procedures: int = Field(40, ge=1, le=150, description="Number of lab procedures performed")
    num_procedures: int = Field(0, ge=0, le=10, description="Number of non-lab operating or bedside procedures")
    num_medications: int = Field(15, ge=1, le=100, description="Total distinct medications administered")
    number_outpatient: int = Field(0, ge=0, description="Outpatient visits in the preceding year")
    number_emergency: int = Field(0, ge=0, description="Emergency department visits in the preceding year")
    number_inpatient: int = Field(0, ge=0, description="Inpatient hospital admissions in the preceding year")
    number_diagnoses: int = Field(7, ge=1, le=16, description="Number of diagnoses recorded on the encounter")

    # Diagnostic codes (ICD-9)
    diag_1: Optional[str] = Field("428", description="Primary ICD-9 diagnosis code (e.g. 428, 250, 414)")
    diag_2: Optional[str] = Field("250.02", description="Secondary ICD-9 diagnosis code")
    diag_3: Optional[str] = Field("401", description="Tertiary ICD-9 diagnosis code")

    # Glycemic monitoring
    max_glu_serum: Optional[str] = Field("None", description="Max glucose serum result (None, Norm, >200, >300)")
    A1Cresult: Optional[str] = Field("None", description="HbA1c test result (None, Norm, >7, >8)")

    # Medications (21 fields)
    metformin: Optional[str] = Field("No", description="metformin status (No, Steady, Up, Down)")
    repaglinide: Optional[str] = Field("No")
    nateglinide: Optional[str] = Field("No")
    chlorpropamide: Optional[str] = Field("No")
    glimepiride: Optional[str] = Field("No")
    glipizide: Optional[str] = Field("No")
    glyburide: Optional[str] = Field("No")
    tolbutamide: Optional[str] = Field("No")
    pioglitazone: Optional[str] = Field("No")
    rosiglitazone: Optional[str] = Field("No")
    acarbose: Optional[str] = Field("No")
    miglitol: Optional[str] = Field("No")
    troglitazone: Optional[str] = Field("No")
    tolazamide: Optional[str] = Field("No")
    insulin: Optional[str] = Field("Steady", description="insulin status (No, Steady, Up, Down)")
    glyburide_metformin: Optional[str] = Field("No", alias="glyburide-metformin")
    glipizide_metformin: Optional[str] = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Optional[str] = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Optional[str] = Field("No", alias="metformin-rosiglitazone")
    change: Optional[str] = Field("No", description="Change in diabetic medication (No, Ch)")
    diabetesMed: Optional[str] = Field("Yes", description="Diabetic medication prescribed (Yes, No)")

    # Supplemental qualitative clinical observations (separate from ML features)
    patient_notes: Optional[str] = Field(None, description="Supplemental qualitative clinical notes for EHR record")
    symptoms_reported: Optional[List[str]] = Field(None, description="Qualitative symptom observations")

    model_config = ConfigDict(populate_by_name=True)


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
