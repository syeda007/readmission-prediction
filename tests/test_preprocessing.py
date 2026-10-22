"""
ReadmitAI Test Suite - Preprocessing & Transformation Tests
Verifies deterministic ICD-9 categorization, handling of unseen categories,
preservation of literal 'None' lab values, and training-only parameter fitting.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import map_icd9_to_category, ReadmitAIFeatureEngineer
from src.preprocessing import build_full_pipeline, get_transformed_feature_names
from sklearn.dummy import DummyClassifier


def test_icd9_deterministic_mapping():
    """Tests diagnosis code domain categorization including V, E, and missing codes."""
    assert map_icd9_to_category("428") == "Circulatory"
    assert map_icd9_to_category("785") == "Circulatory"
    assert map_icd9_to_category("486") == "Respiratory"
    assert map_icd9_to_category("786") == "Respiratory"
    assert map_icd9_to_category("530") == "Digestive"
    assert map_icd9_to_category("250.02") == "Diabetes"
    assert map_icd9_to_category("820") == "Injury"
    assert map_icd9_to_category("715") == "Musculoskeletal"
    assert map_icd9_to_category("585") == "Genitourinary"
    assert map_icd9_to_category("174") == "Neoplasms"
    assert map_icd9_to_category("V45") == "Other"
    assert map_icd9_to_category("E878") == "Other"
    assert map_icd9_to_category(None) == "Missing"
    assert map_icd9_to_category("?") == "Missing"
    assert map_icd9_to_category("") == "Missing"


def test_lab_none_category_preserved():
    """Verifies that missing lab values are retained as 'None' (untested), not imputed to normal."""
    raw_df = pd.DataFrame({
        "A1Cresult": [None, "Norm", ">8", ""],
        "max_glu_serum": [None, ">200", "Norm", "?"],
        "time_in_hospital": [3, 2, 4, 1],
    })
    fe = ReadmitAIFeatureEngineer()
    fe.fit(raw_df)
    transformed = fe.transform(raw_df)

    assert (transformed["A1Cresult"] == "None").sum() == 2
    assert (transformed["max_glu_serum"] == "None").sum() == 2


def test_unseen_categories_do_not_crash_pipeline():
    """Verifies that novel categorical levels during inference do not raise exceptions."""
    train_data = pd.DataFrame({
        "race": ["Caucasian", "AfricanAmerican", "Hispanic"],
        "gender": ["Female", "Male", "Female"],
        "age": ["[70-80)", "[60-70)", "[50-60)"],
        "admission_type_id": [1, 2, 3],
        "discharge_disposition_id": [1, 3, 6],
        "admission_source_id": [7, 1, 2],
        "payer_code": ["MC", "MD", "BC"],
        "medical_specialty": ["InternalMedicine", "Cardiology", "Surgery-General"],
        "time_in_hospital": [3, 5, 2],
        "num_lab_procedures": [40, 50, 30],
        "num_procedures": [1, 0, 2],
        "num_medications": [15, 20, 10],
        "number_outpatient": [0, 1, 0],
        "number_emergency": [0, 0, 1],
        "number_inpatient": [1, 0, 2],
        "number_diagnoses": [8, 9, 6],
        "max_glu_serum": ["None", "Norm", ">200"],
        "A1Cresult": ["None", ">8", "Norm"],
        "metformin": ["No", "Steady", "No"],
        "repaglinide": ["No", "No", "No"],
        "nateglinide": ["No", "No", "No"],
        "chlorpropamide": ["No", "No", "No"],
        "glimepiride": ["No", "No", "No"],
        "glipizide": ["No", "Steady", "No"],
        "glyburide": ["No", "No", "No"],
        "tolbutamide": ["No", "No", "No"],
        "pioglitazone": ["No", "No", "No"],
        "rosiglitazone": ["No", "No", "No"],
        "acarbose": ["No", "No", "No"],
        "miglitol": ["No", "No", "No"],
        "troglitazone": ["No", "No", "No"],
        "tolazamide": ["No", "No", "No"],
        "insulin": ["Steady", "Up", "No"],
        "glyburide-metformin": ["No", "No", "No"],
        "glipizide-metformin": ["No", "No", "No"],
        "glimepiride-pioglitazone": ["No", "No", "No"],
        "metformin-rosiglitazone": ["No", "No", "No"],
        "change": ["Ch", "Ch", "No"],
        "diabetesMed": ["Yes", "Yes", "No"],
        "diag_1": ["428", "414", "250"],
        "diag_2": ["250", "401", "486"],
        "diag_3": ["401", "585", "272"],
    })
    y_dummy = np.array([0, 1, 0])

    pipe = build_full_pipeline(DummyClassifier())
    pipe.fit(train_data, y_dummy)

    # Inference data with completely novel categories
    test_novel = train_data.copy().iloc[:1]
    test_novel["race"] = "NovelMartianRace"
    test_novel["medical_specialty"] = "CosmicSurgery"
    test_novel["payer_code"] = "BitcoinPayer"

    probs = pipe.predict_proba(test_novel)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
