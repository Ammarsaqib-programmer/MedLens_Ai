"""
MediLens AI - Agentic AI medical report & medicine assistant (Streamlit)

Pipeline:  Upload -> Extract -> Analyze -> Retrieve (RAG) -> Verify -> Explain -> Guide
Works in two modes:
  * AI mode      : Gemini API key present  -> vision extraction, richer explanations, translation
  * Offline mode : no key                  -> OCR / PDF-text + rule-based agents + curated knowledge base
Educational tool only. NOT a medical device. Never replaces a doctor or pharmacist.
"""
import base64
import difflib
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MediLens AI", page_icon="🩺", layout="wide")

DISCLAIMER = (
    "⚠️ **MediLens AI is an educational tool, not a doctor.** It does not diagnose, prescribe or "
    "change treatment. Always confirm results and any medicine decision with a qualified doctor or pharmacist."
)

# ----------------------------------------------------------------------------------------------
# 1. MEDICAL KNOWLEDGE BASE  (curated reference ranges + plain-language meaning)
# ----------------------------------------------------------------------------------------------


@dataclass
class Param:
    key: str
    name: str
    aliases: list
    unit: str
    lo: Optional[float]
    hi: Optional[float]
    what: str = ""
    low_txt: str = ""
    high_txt: str = ""
    mlo: Optional[float] = None
    mhi: Optional[float] = None
    flo: Optional[float] = None
    fhi: Optional[float] = None
    units: Optional[list] = None  # acceptable normalised units (None = skip unit check)
    clo: Optional[float] = None  # critical low
    chi: Optional[float] = None  # critical high
    group: str = "General"

    def range_for(self, sex):
        if sex == "male" and (self.mlo is not None or self.mhi is not None):
            return self.mlo, self.mhi
        if sex == "female" and (self.flo is not None or self.fhi is not None):
            return self.flo, self.fhi
        return self.lo, self.hi

    def sex_specific(self):
        return any(v is not None for v in (self.mlo, self.mhi, self.flo, self.fhi))


_P = []


def add(*a, **k):
    _P.append(Param(*a, **k))


# ---- CBC
add("hb", "Hemoglobin (Hb)", ["hemoglobin", "haemoglobin", "hgb", "hb"], "g/dL", 12.0, 17.0,
    "Hemoglobin is the protein in red blood cells that carries oxygen around the body.",
    "A low level is called anemia. Common causes are iron, vitamin B12 or folate deficiency, blood loss, or long-term illness. It can cause tiredness, weakness, pale skin and breathlessness.",
    "A high level can be seen with dehydration, smoking, living at high altitude, or conditions that increase red cell production. A doctor needs to look for the cause.",
    mlo=13.0, mhi=17.0, flo=12.0, fhi=15.5, units=["g/dl"], clo=7.0, chi=20.0, group="CBC")
add("wbc", "White Blood Cells (WBC/TLC)",
    ["total\\s*leu[ck]ocyte\\s*count", "total\\s*wbc\\s*count", "wbc\\s*count", "wbc", "tlc", "white\\s*blood\\s*cells?", "white\\s*cell\\s*count", "leu[ck]ocytes"],
    "×10³/µL", 4.0, 11.0,
    "White blood cells are the body's defence cells; they fight infections and take part in inflammation and allergy.",
    "A low count (leukopenia) can happen with viral infections, some medicines, or bone-marrow/immune problems, and may lower resistance to infection.",
    "A high count often means the body is fighting an infection or inflammation. Stress, smoking and steroid medicines can also raise it. Very high values need prompt review.",
    clo=2.0, chi=30.0, group="CBC")
add("plt", "Platelets", ["platelet\\s*count", "platelets?", "plt", "thrombocytes"], "×10³/µL", 150, 450,
    "Platelets are tiny cell fragments that help blood clot and stop bleeding.",
    "Low platelets (thrombocytopenia) can occur with viral infections such as dengue, some medicines, or immune/bone-marrow conditions. Very low counts raise the risk of bleeding.",
    "High platelets can follow infection, inflammation, iron deficiency or recent surgery. Persistently high values should be reviewed by a doctor.",
    clo=50, chi=1000, group="CBC")
add("rbc", "Red Blood Cells (RBC)", ["rbc\\s*count", "total\\s*rbc", "red\\s*blood\\s*cells?", "red\\s*cell\\s*count", "rbc", "erythrocytes"],
    "million/µL", 4.1, 5.9, "Red blood cells carry oxygen from the lungs to the body.",
    "Low RBC count usually goes along with anemia.", "High RBC count can be seen with dehydration, smoking, high altitude or increased red-cell production.",
    mlo=4.5, mhi=5.9, flo=4.1, fhi=5.1, group="CBC")
add("hct", "Hematocrit / PCV", ["hematocrit", "haematocrit", "hct", "pcv", "packed\\s*cell\\s*volume"], "%", 36, 52,
    "Hematocrit (PCV) is the percentage of your blood made up of red cells.",
    "Low values go with anemia or blood loss.", "High values can be seen with dehydration or increased red-cell production.",
    mlo=40, mhi=52, flo=36, fhi=46, units=["%"], group="CBC")
add("mcv", "MCV (average red-cell size)", ["mcv", "mean\\s*cell(?:ular)?\\s*volume"], "fL", 80, 100,
    "MCV tells whether red blood cells are small, normal or large.",
    "Small red cells are commonly seen in iron deficiency or thalassemia trait.",
    "Large red cells can be seen with vitamin B12/folate deficiency, alcohol use, thyroid or liver conditions, or some medicines.",
    units=["fl"], group="CBC")
add("mch", "MCH (hemoglobin per red cell)", ["mch", "mean\\s*cell\\s*hemoglobin(?!\\s*conc)"], "pg", 27, 33,
    "MCH is the average amount of hemoglobin inside each red blood cell.",
    "Low MCH usually accompanies iron deficiency or thalassemia trait.", "High MCH is usually seen with large red cells (B12/folate deficiency).",
    units=["pg"], group="CBC")
add("mchc", "MCHC", ["mchc", "mean\\s*cell\\s*hemoglobin\\s*conc\\w*"], "g/dL", 32, 36,
    "MCHC is the concentration of hemoglobin in red cells.", "Low MCHC (pale cells) is typical of iron deficiency.",
    "High MCHC is uncommon and can be a lab artefact or related to certain red-cell conditions.", units=["g/dl", "%"], group="CBC")
add("rdw", "RDW", ["rdw(?:[- ]?cv)?", "red\\s*cell\\s*distribution\\s*width"], "%", 11.5, 14.5,
    "RDW shows how much red cells vary in size.", "",
    "A high RDW means red cells differ a lot in size; it is often an early sign of iron, B12 or folate deficiency.", units=["%"], group="CBC")
add("neut", "Neutrophils (%)", ["neutrophils?", "polymorphs?", "segmented\\s*neutrophils?"], "%", 40, 75,
    "Neutrophils are the white cells that first respond to bacterial infection.",
    "Low neutrophils can follow viral infection or some medicines and may lower resistance to bacteria.",
    "High neutrophils often point to bacterial infection, inflammation or stress.", units=["%"], group="CBC")
add("lymph", "Lymphocytes (%)", ["lymphocytes?"], "%", 20, 45,
    "Lymphocytes are white cells important for fighting viruses and for immunity.",
    "Low lymphocytes can occur with stress, steroids or some infections.", "High lymphocytes are common after viral infections in children and adults.",
    units=["%"], group="CBC")
add("mono", "Monocytes (%)", ["monocytes?"], "%", 2, 10, "Monocytes are white cells that clean up debris and fight some infections.",
    "", "Mildly raised monocytes can be seen during recovery from infection or with chronic inflammation.", units=["%"], group="CBC")
add("eos", "Eosinophils (%)", ["eosinophils?"], "%", 1, 6, "Eosinophils are white cells involved in allergy and parasite responses.",
    "", "High eosinophils are often linked to allergy, asthma, skin conditions or parasitic infection.", units=["%"], group="CBC")
add("baso", "Basophils (%)", ["basophils?"], "%", 0, 1, "Basophils are rare white cells involved in allergic reactions.",
    "", "Slightly raised basophils are usually not significant on their own.", units=["%"], group="CBC")
add("esr", "ESR", ["esr", "erythrocyte\\s*sedimentation\\s*rate"], "mm/hr", 0, 20,
    "ESR is a non-specific marker of inflammation in the body.", "",
    "A high ESR means there may be inflammation, infection or another condition; it does not point to one specific disease.",
    mhi=15, mlo=0, fhi=20, flo=0, units=["mm/h", "mm/hr"], group="CBC")
# ---- Sugar
add("fbs", "Fasting Blood Glucose",
    ["fasting\\s*blood\\s*(?:sugar|glucose)", "fasting\\s*(?:plasma\\s*)?glucose", "fasting\\s*sugar", "glucose\\s*[,(]?\\s*fasting\\)?", "blood\\s*sugar\\s*fasting", "fbs", "fbg", "fpg"],
    "mg/dL", 70, 99, "Fasting glucose measures blood sugar after at least 8 hours without food.",
    "Low sugar (hypoglycemia) can cause sweating, shakiness, confusion; it can be dangerous if severe.",
    "Fasting sugar of 100–125 mg/dL is called prediabetes range and 126 mg/dL or more (confirmed on repeat) is in the diabetes range. Only a doctor can confirm the diagnosis.",
    units=["mg/dl"], clo=50, chi=400, group="Sugar")
add("rbs", "Random Blood Glucose", ["random\\s*blood\\s*(?:sugar|glucose)", "random\\s*(?:sugar|glucose)", "glucose\\s*[,(]?\\s*random\\)?", "blood\\s*sugar\\s*random", "rbs", "rbg"],
    "mg/dL", 70, 139, "Random glucose is blood sugar measured at any time of day, regardless of meals.",
    "Low sugar (hypoglycemia) can cause sweating, shakiness, confusion.",
    "A random value of 200 mg/dL or more with symptoms (thirst, frequent urination) can point to diabetes and needs a doctor's review.",
    units=["mg/dl"], clo=50, chi=400, group="Sugar")
add("hba1c", "HbA1c", ["hb\\s*a1c", "hemoglobin\\s*a1c", "haemoglobin\\s*a1c", "hba1c", "glyc(?:o)?sylated\\s*h(?:a)?emoglobin", "glycated\\s*h(?:a)?emoglobin", "a1c"],
    "%", 4.0, 5.6, "HbA1c shows your average blood sugar over the last 2–3 months.", "",
    "5.7–6.4% is the prediabetes range and 6.5% or more is in the diabetes range. Higher values mean sugar has been running high for months.",
    units=["%"], chi=14.0, group="Sugar")
# ---- Kidney
add("creat", "Creatinine", ["serum\\s*creatinine", "creatinine"], "mg/dL", 0.6, 1.3,
    "Creatinine is a waste product filtered by the kidneys; it helps estimate kidney function.",
    "Low creatinine is usually not worrying and can reflect low muscle mass.",
    "High creatinine can mean the kidneys are not filtering well; dehydration, some medicines and kidney disease are possible reasons.",
    mlo=0.7, mhi=1.3, flo=0.6, fhi=1.1, units=["mg/dl"], chi=10.0, group="Kidney")
add("urea", "Blood Urea", ["blood\\s*urea(?!\\s*nitrogen)", "serum\\s*urea", "urea"], "mg/dL", 15, 45,
    "Urea is a waste product made in the liver and removed by the kidneys.", "Low urea can occur with low protein intake or liver disease.",
    "High urea can be due to dehydration, high protein intake, or reduced kidney function.", units=["mg/dl"], chi=200, group="Kidney")
add("bun", "BUN (Blood Urea Nitrogen)", ["blood\\s*urea\\s*nitrogen", "bun"], "mg/dL", 7, 20,
    "BUN measures the nitrogen from urea in blood and is used to assess kidney function.", "", "High BUN can be due to dehydration or reduced kidney function.",
    units=["mg/dl"], group="Kidney")
add("uric", "Uric Acid", ["serum\\s*uric\\s*acid", "uric\\s*acid"], "mg/dL", 2.6, 7.2,
    "Uric acid is a waste product; too much can form crystals in joints (gout) or kidneys.", "Low uric acid is usually not significant.",
    "High uric acid can cause gout or kidney stones in some people; diet, alcohol and kidney function play a role.",
    mlo=3.5, mhi=7.2, flo=2.6, fhi=6.0, units=["mg/dl"], group="Kidney")
# ---- Liver
add("alt", "ALT (SGPT)", ["alanine\\s*(?:amino)?transferase", "alanine\\s*transaminase", "sgpt", "alt"], "U/L", 7, 56,
    "ALT is a liver enzyme; it rises when liver cells are irritated or damaged.", "",
    "High ALT suggests liver irritation: fatty liver, viral hepatitis, medicines, alcohol or other causes. A doctor should find the reason.",
    units=["u/l", "iu/l"], chi=1000, group="Liver")
add("ast", "AST (SGOT)", ["aspartate\\s*(?:amino)?transferase", "aspartate\\s*transaminase", "sgot", "ast"], "U/L", 10, 40,
    "AST is an enzyme found in the liver, heart and muscles.", "", "High AST can come from the liver, but also from muscle or heart; it is read together with ALT.",
    units=["u/l", "iu/l"], chi=1000, group="Liver")
add("alp", "ALP (Alkaline Phosphatase)", ["alkaline\\s*phosphatase", "alp"], "U/L", 44, 147,
    "ALP is an enzyme found in the liver and bones.", "", "High ALP can point to bile-duct or bone conditions; growing children normally have higher values.",
    units=["u/l", "iu/l"], group="Liver")
add("tbil", "Total Bilirubin", ["total\\s*bilirubin", "bilirubin\\s*[,(]?\\s*total\\)?", "t\\.?\\s*bilirubin", "serum\\s*bilirubin", "bilirubin"], "mg/dL", 0.1, 1.2,
    "Bilirubin is a yellow pigment made when old red cells break down; the liver clears it.", "",
    "High bilirubin can cause yellowing of skin/eyes (jaundice) and may relate to liver, bile-duct or red-cell breakdown issues.",
    units=["mg/dl"], chi=15.0, group="Liver")
add("dbil", "Direct Bilirubin", ["direct\\s*bilirubin", "bilirubin\\s*[,(]?\\s*direct\\)?", "d\\.?\\s*bilirubin", "conjugated\\s*bilirubin"], "mg/dL", 0.0, 0.3,
    "Direct bilirubin is the part of bilirubin already processed by the liver.", "", "High direct bilirubin often points to a bile-flow problem or liver disease.",
    units=["mg/dl"], group="Liver")
add("alb", "Albumin", ["serum\\s*albumin", "albumin"], "g/dL", 3.5, 5.0, "Albumin is the main protein in blood, made by the liver.",
    "Low albumin can occur with liver disease, kidney protein loss, malnutrition or long illness.", "High albumin is usually from dehydration.", units=["g/dl"], group="Liver")
add("tp", "Total Protein", ["total\\s*protein", "serum\\s*protein"], "g/dL", 6.0, 8.3, "Total protein is the sum of albumin and globulins in blood.",
    "Low values can relate to liver, kidney or nutrition problems.", "High values can relate to dehydration or chronic inflammation.", units=["g/dl"], group="Liver")
# ---- Lipids
add("chol", "Total Cholesterol", ["total\\s*cholesterol", "cholesterol\\s*,?\\s*total", "serum\\s*cholesterol", "cholesterol"], "mg/dL", None, 199,
    "Cholesterol is a fatty substance in blood; too much raises long-term heart and stroke risk.", "",
    "A total cholesterol of 200 mg/dL or more is above the desirable level (240+ is high). Diet, activity, weight and family history all play a role.",
    units=["mg/dl"], group="Lipids")
add("ldl", "LDL Cholesterol ('bad')", ["ldl[- ]?c(?:holesterol)?", "low\\s*density\\s*lipoprotein", "ldl"], "mg/dL", None, 129,
    "LDL is the 'bad' cholesterol that can build up in artery walls.", "",
    "LDL above about 130 mg/dL is borderline-high or high and raises heart risk over time; the target is lower for people with diabetes or heart disease.",
    units=["mg/dl"], group="Lipids")
add("hdl", "HDL Cholesterol ('good')", ["hdl[- ]?c(?:holesterol)?", "high\\s*density\\s*lipoprotein", "hdl"], "mg/dL", 40, None,
    "HDL is the 'good' cholesterol that helps remove cholesterol from blood.",
    "Low HDL (below ~40 in men, ~50 in women) is linked with higher heart risk; exercise and healthy fats can help.", "",
    mlo=40, flo=50, units=["mg/dl"], group="Lipids")
add("tg", "Triglycerides", ["triglycerides?", "tgs?", "trigs?"], "mg/dL", None, 149,
    "Triglycerides are fats in blood that come from food and are made by the liver.", "",
    "High triglycerides (150+ mg/dL) are linked with sugary/fatty diets, obesity, diabetes, alcohol and inactivity; very high levels can inflame the pancreas.",
    units=["mg/dl"], chi=1000, group="Lipids")
# ---- Electrolytes
add("na", "Sodium", ["serum\\s*sodium", "sodium"], "mmol/L", 135, 145, "Sodium balances water in the body and supports nerves and muscles.",
    "Low sodium can cause weakness, headache and confusion; causes include fluid overload, some medicines and illness.",
    "High sodium usually means the body is short of water.", units=["mmol/l", "meq/l"], clo=120, chi=160, group="Electrolytes")
add("k", "Potassium", ["serum\\s*potassium", "potassium"], "mmol/L", 3.5, 5.1, "Potassium is vital for heart rhythm and muscle function.",
    "Low potassium can cause weakness, cramps and rhythm problems (vomiting, diarrhoea and some medicines are common causes).",
    "High potassium can disturb heart rhythm and needs prompt medical attention if significantly raised; kidney problems and some medicines are common causes.",
    units=["mmol/l", "meq/l"], clo=2.8, chi=6.0, group="Electrolytes")
add("cl", "Chloride", ["serum\\s*chloride", "chloride"], "mmol/L", 98, 107, "Chloride works with sodium to keep fluid balance.",
    "Low chloride can accompany vomiting or fluid imbalance.", "High chloride can accompany dehydration or acid-base changes.", units=["mmol/l", "meq/l"], group="Electrolytes")
add("ca", "Calcium", ["serum\\s*calcium", "total\\s*calcium", "calcium"], "mg/dL", 8.5, 10.5, "Calcium is essential for bones, nerves, muscles and the heart.",
    "Low calcium can come from low vitamin D or parathyroid/kidney problems and may cause cramps or tingling.",
    "High calcium may come from parathyroid problems or excess supplements and needs medical review.", units=["mg/dl"], clo=6.5, chi=13.0, group="Electrolytes")
# ---- Others
add("tsh", "TSH (thyroid)", ["thyroid\\s*stimulating\\s*hormone", "tsh"], "mIU/L", 0.4, 4.0, "TSH is the brain's signal to the thyroid gland; it is the main screening test for thyroid function.",
    "Low TSH can mean an overactive thyroid (hyperthyroidism) - e.g. palpitations, weight loss, anxiety.",
    "High TSH can mean an underactive thyroid (hypothyroidism) - e.g. tiredness, weight gain, feeling cold. Doctors usually confirm with T3/T4 tests.",
    units=["miu/l", "uiu/ml", "uiu/l", "mu/l", "miu/ml"], group="Thyroid")
add("vitd", "Vitamin D (25-OH)", ["(?:25[- ]?(?:oh|hydroxy)[- ]*)?vitamin\\s*d(?:\\s*[,(-]?\\s*25[- ]?(?:oh|hydroxy)[a-z ]*\\)?)?", "25[- ]?(?:oh|hydroxy)[- ]*vit(?:amin)?[- ]*d"],
    "ng/mL", 30, 100, "Vitamin D helps the body absorb calcium and keeps bones and muscles strong.",
    "Below 20 ng/mL is deficiency and 20–29 is insufficiency. It is very common in South Asia and can cause bone/muscle aches and fatigue. A doctor decides on treatment.",
    "Very high vitamin D usually results from excess supplements and can raise calcium.", units=["ng/ml"], group="Vitamins & Iron")
add("b12", "Vitamin B12", ["vitamin\\s*b[- ]?12", "vit\\.?\\s*b[- ]?12", "cobalamin", "b[- ]?12"], "pg/mL", 200, 900, "Vitamin B12 is needed for nerves and for making healthy red blood cells.",
    "Low B12 can cause anemia with large red cells, tingling/numbness and tiredness; vegetarian diets and absorption problems are common causes.", "",
    units=["pg/ml", "pmol/l", "ng/l"], group="Vitamins & Iron")
add("ferr", "Ferritin", ["serum\\s*ferritin", "ferritin"], "ng/mL", 11, 307, "Ferritin reflects the body's iron stores.",
    "Low ferritin is the most reliable sign of low iron stores (iron deficiency), even before anemia appears.",
    "High ferritin can be due to inflammation, liver disease or iron overload.", mlo=24, mhi=336, flo=11, fhi=307, units=["ng/ml", "ug/l"], group="Vitamins & Iron")
add("iron", "Serum Iron", ["serum\\s*iron", "iron"], "µg/dL", 60, 170, "Serum iron measures iron currently circulating in the blood.",
    "Low iron suggests iron deficiency or chronic illness.", "High iron can follow iron supplements or iron overload conditions.", units=["ug/dl"], group="Vitamins & Iron")
add("crp", "CRP", ["c[- ]?reactive\\s*protein", "crp"], "mg/L", 0, 5, "CRP is a blood marker that rises quickly with infection or inflammation.", "",
    "A raised CRP means inflammation or infection somewhere in the body; it does not show where.", units=["mg/l"], chi=200, group="Inflammation")

PARAMS = {p.key: p for p in _P}

# Compiled alias patterns (longest first)
_ALIAS = []
for _p in _P:
    for _a in _p.aliases:
        _ALIAS.append((len(_a), _p.key, re.compile(r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z])" % _a, re.I)))
_ALIAS.sort(key=lambda x: -x[0])

# Extra knowledge documents (patterns + radiology glossary)
EXTRA_DOCS = [
    ("Pattern: iron-deficiency picture", "low hemoglobin low mcv low mch low ferritin low iron anemia microcytic thalassemia",
     "Low hemoglobin together with small red cells (low MCV/MCH) and low ferritin or iron is the typical picture of iron-deficiency anemia. Thalassemia trait and chronic disease can look similar. Doctors usually decide on iron tests, diet review and looking for hidden blood loss."),
    ("Pattern: B12/folate picture", "low hemoglobin high mcv low vitamin b12 macrocytic folate anemia",
     "Low hemoglobin with large red cells (high MCV) suggests vitamin B12 or folate deficiency, thyroid or liver causes. B12 level testing helps; deficiency is common with vegetarian diets or poor absorption."),
    ("Pattern: infection or inflammation", "high wbc high neutrophils high crp high esr infection inflammation",
     "A raised white-cell count with high neutrophils, CRP or ESR suggests the body is fighting infection or inflammation. The test cannot tell where; symptoms and examination guide the doctor."),
    ("Pattern: low platelets after fever", "low platelets low wbc fever viral dengue thrombocytopenia",
     "Low platelets with a low or normal white-cell count during or after fever is commonly seen with viral illnesses such as dengue. Doctors track counts over days. Bleeding, severe stomach pain, persistent vomiting or extreme weakness need urgent medical care."),
    ("Pattern: blood sugar control", "high fasting glucose high hba1c high random glucose diabetes prediabetes sugar",
     "Raised fasting glucose or HbA1c suggests prediabetes or diabetes. Diagnosis needs confirmation by a doctor. Diet, weight, activity and follow-up testing are the usual starting points."),
    ("Pattern: cholesterol and heart health", "high ldl high cholesterol high triglycerides low hdl lipid profile heart",
     "High LDL or triglycerides and low HDL increase long-term risk of heart disease and stroke. Doctors consider the full lipid profile together with age, blood pressure, sugar, smoking and family history."),
    ("Pattern: liver enzymes", "high alt high ast high alp high bilirubin liver fatty liver hepatitis jaundice",
     "Raised ALT/AST point to liver cell irritation (fatty liver, hepatitis, alcohol, medicines). Raised ALP or bilirubin may point to bile-flow problems. A doctor may suggest an ultrasound or hepatitis tests."),
    ("Pattern: kidney function", "high creatinine high urea high bun kidney dehydration",
     "High creatinine and urea can reflect dehydration or reduced kidney function. Repeat tests, a urine test and a kidney ultrasound are common next steps decided by the doctor."),
    ("Radiology: impression", "impression conclusion findings radiology report summary",
     "The 'Impression' or 'Conclusion' section is the radiologist's short summary of the most important findings. Ask your doctor to explain it in the context of your symptoms."),
    ("Radiology: effusion", "pleural effusion fluid chest x-ray costophrenic angle blunting",
     "A pleural effusion means extra fluid around the lung. It can have many causes (infection, heart, kidney, liver or other conditions) and the cause needs medical evaluation."),
    ("Radiology: consolidation / opacity", "consolidation opacity infiltrate haziness lung pneumonia chest x-ray",
     "Consolidation or an opacity is a cloudy area in the lung where air spaces are filled with fluid or cells. It is often seen with infections such as pneumonia, but the doctor must correlate it with symptoms."),
    ("Radiology: cardiomegaly", "cardiomegaly enlarged heart cardiothoracic ratio",
     "Cardiomegaly means the heart shadow looks larger than usual. It may be due to many reasons (including technique of the X-ray), so doctors often advise an echocardiogram."),
    ("Radiology: nodule or mass", "nodule mass lesion lump growth tumor",
     "A nodule or mass is an abnormal spot or lump. Many are harmless, but the radiologist usually advises follow-up imaging or further tests to be sure."),
    ("Radiology: fracture", "fracture broken bone crack cortical break",
     "A fracture means a break in a bone. The report usually describes the site and alignment; an orthopaedic doctor decides on treatment."),
    ("Radiology: fatty liver / hepatomegaly", "fatty liver hepatomegaly echogenic liver steatosis ultrasound",
     "A 'bright' (echogenic) or enlarged liver on ultrasound is commonly due to fatty liver, which is very common and often linked to weight, sugar and cholesterol. Lifestyle changes and doctor follow-up are usual."),
    ("Radiology: stones", "gallstones cholelithiasis kidney stone calculus calculi renal ultrasound",
     "Calculi or stones are hard deposits in the gallbladder or urinary tract. Size and position decide whether they need only monitoring or treatment, which a doctor/urologist/surgeon will advise."),
    ("Radiology: cyst", "cyst cystic simple cyst fluid-filled",
     "A simple cyst is a fluid-filled sac and is usually harmless. Complex or growing cysts may need follow-up as advised by the doctor."),
    ("Radiology: lymph nodes", "lymph nodes lymphadenopathy enlarged nodes",
     "Enlarged lymph nodes are often reactive to infection, but sometimes need further evaluation. The doctor will correlate with symptoms."),
]


def build_kb():
    docs = []
    for p in PARAMS.values():
        txt = f"{p.what} LOW: {p.low_txt or 'No specific meaning.'} HIGH: {p.high_txt or 'No specific meaning.'}"
        docs.append({"id": f"param:{p.key}", "title": p.name, "keywords": " ".join(a.replace("\\s*", " ") for a in p.aliases),
                     "text": txt, "key": p.key})
    for i, (t, kw, tx) in enumerate(EXTRA_DOCS):
        docs.append({"id": f"extra:{i}", "title": t, "keywords": kw, "text": tx, "key": None})
    return docs


@st.cache_resource
def kb_index():
    docs = build_kb()
    corpus = [f"{d['title']} {d['keywords']} {d['text']}" for d in docs]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(corpus)
    return docs, vec, mat


# ----------------------------------------------------------------------------------------------
# 2. LLM / OCR HELPERS
# ----------------------------------------------------------------------------------------------
class LLMError(Exception):
    pass


def _secret(name):
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return v or os.environ.get(name, "")


def gemini_key():
    return (st.session_state.get("user_key") or "").strip() or _secret("GEMINI_API_KEY")


def groq_key():
    return (st.session_state.get("user_groq_key") or "").strip() or _secret("GROQ_API_KEY")


def get_api_key():
    """Truthy when any AI provider is configured."""
    return groq_key() or gemini_key()


def provider():
    return "groq" if groq_key() else ("gemini" if gemini_key() else "")


def get_models():
    m = _secret("GEMINI_MODEL")
    return list(dict.fromkeys([x for x in [m, "gemini-2.5-flash", "gemini-flash-latest"] if x]))


def _call_groq(prompt, files, system, json_mode, max_tokens):
    has_img = any(m.startswith("image/") for _, m in files)
    if any(not m.startswith("image/") for _, m in files):
        raise LLMError("Groq cannot read PDF files directly")
    if has_img:
        models = [_secret("GROQ_VISION_MODEL"), "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"]
    else:
        models = [_secret("GROQ_TEXT_MODEL"), "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    models = list(dict.fromkeys([x for x in models if x]))
    content = [{"type": "text", "text": prompt}]
    for data, mime in files:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}})
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content if has_img else prompt}]
    last = ""
    for model in models:
        body = {"model": model, "messages": msgs, "temperature": 0.2, "max_completion_tokens": min(max_tokens, 4096)}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {groq_key()}", "Content-Type": "application/json"}, json=body, timeout=120)
        except requests.RequestException as e:
            raise LLMError(f"Network error: {e}")
        if r.status_code in (400, 404) and any(w in r.text.lower() for w in ("decommission", "not found", "does not exist", "model_not_found")):
            last = f"model {model} unavailable"
            continue
        if r.status_code != 200:
            raise LLMError(f"Groq API error {r.status_code}: {r.text[:200]}")
        try:
            text = r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError("Empty response from Groq")
        if text.strip():
            return text
        last = "empty response"
    raise LLMError(last or "No Groq model responded")


def call_gemini(prompt, files=(), system=None, json_mode=False, max_tokens=8192):
    """Provider-agnostic LLM call (name kept for compatibility): Groq if its key is set, else Gemini."""
    if provider() == "groq":
        return _call_groq(prompt, files, system, json_mode, max_tokens)
    key = gemini_key()
    if not key:
        raise LLMError("No API key")
    parts = [{"text": prompt}]
    for data, mime in files:
        parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    last = ""
    for model in get_models():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=120)
        except requests.RequestException as e:
            raise LLMError(f"Network error: {e}")
        if r.status_code == 404:
            last = f"model {model} not found"
            continue
        if r.status_code != 200:
            raise LLMError(f"Gemini API error {r.status_code}: {r.text[:200]}")
        data = r.json()
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError("Empty/blocked response from the model")
        if text.strip():
            return text
        last = "empty response"
    raise LLMError(last or "No model responded")


def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def shrink_image(data, max_side=1600):
    try:
        from PIL import Image, ImageOps
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, "image/jpeg"


def ocr_image(data):
    try:
        import pytesseract
        from PIL import Image, ImageOps
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
        return pytesseract.image_to_string(im, config="--psm 6")
    except Exception:
        return ""


def pdf_text(data):
    try:
        import pdfplumber
        out = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pg in pdf.pages[:15]:
                out.append(pg.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""


# ----------------------------------------------------------------------------------------------
# 3. AGENTS - REPORT PIPELINE
# ----------------------------------------------------------------------------------------------
NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")
UNIT_RE = re.compile(r"(?<![a-z])(mmol/l|umol/l|meq/l|miu/ml|miu/l|uiu/ml|uiu/l|mu/l|iu/l|u/l|pg/ml|ng/ml|ug/l|ng/dl|ug/dl|mg/dl|mg/l|g/dl|g/l|mm/hr|mm/h|fl|pg|%)(?![a-z/])", re.I)


def norm_unit(u):
    return u.lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def match_param(name):
    best = None
    for ln, key, rx in _ALIAS:
        m = rx.search(name)
        if m and (best is None or (m.end() - m.start()) > best[0]):
            best = (m.end() - m.start(), key)
    return best[1] if best else None


def detect_sex(text):
    m = re.search(r"(?:sex|gender)\s*[:/\-]?\s*(male|female|m|f)\b", text, re.I)
    if m:
        return "male" if m.group(1).lower().startswith("m") else "female"
    m = re.search(r"\b(mr\.?|mrs\.?|ms\.?|miss)\s", text, re.I)
    if m:
        return "male" if m.group(1).lower().startswith("mr.") or m.group(1).lower() == "mr" else "female"
    return None


def regex_extract(text):
    """Offline extraction agent: find known tests + value + (optional) printed reference range."""
    found = {}
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw.replace("µ", "u").replace("μ", "u")).strip()
        if len(line) < 3:
            continue
        best = None
        for ln, key, rx in _ALIAS:
            m = rx.search(line)
            if m and (best is None or (m.end() - m.start()) > (best[0].end() - best[0].start())):
                best = (m, key)
        if not best:
            continue
        m, key = best
        if key in found:
            continue
        low = line.lower()
        if key in ("neut", "lymph", "mono", "eos", "baso") and ("absolute" in low or "abs." in low or "abs " in low):
            continue
        rest = line[m.end():]
        rest_c = re.sub(r"(?i)x?\s*10\s*[\^*]\s*\d+\s*/?\s*[a-z]*", " ", rest)
        nm = NUM.search(rest_c)
        if not nm:
            continue
        val = to_float(nm.group(0))
        if val is None:
            continue
        after = rest_c[nm.end():]
        rng_lo = rng_hi = None
        rm = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)", after)
        if rm:
            a, b = float(rm.group(1)), float(rm.group(2))
            if a < b:
                rng_lo, rng_hi = a, b
        else:
            lt = re.search(r"(?:<|≤|up to|upto|less than)\s*=?\s*(\d+(?:\.\d+)?)", after, re.I)
            gt = re.search(r"(?:>|≥|more than|greater than)\s*=?\s*(\d+(?:\.\d+)?)", after, re.I)
            if lt:
                rng_hi = float(lt.group(1))
            if gt:
                rng_lo = float(gt.group(1))
        seg = after[:rm.start()] if rm else after[:25]
        um = UNIT_RE.search(seg)
        unit = um.group(1) if um else ""
        fm = re.search(r"(?<![A-Za-z])(HH|LL|H|L)(?![A-Za-z])", seg + " " + rest_c[:nm.start()])
        flag = fm.group(1)[0] if fm else ""
        found[key] = {"key": key, "name": PARAMS[key].name, "value": val, "raw": nm.group(0), "unit": unit,
                      "ref_lo": rng_lo, "ref_hi": rng_hi, "flag": flag, "line": line}
    return list(found.values())


def extraction_agent(data, mime, filename):
    """Agent 1 - Extract values/findings. Returns dict(items, narrative, text, sex, mode, notes)."""
    notes, text, items, narrative, sex = [], "", [], "", None
    is_pdf = mime == "application/pdf" or filename.lower().endswith(".pdf")
    is_txt = filename.lower().endswith(".txt")
    if is_pdf:
        text = pdf_text(data)
    elif is_txt:
        text = data.decode("utf-8", "ignore")
    else:
        text = ocr_image(data) if not get_api_key() else ""
    mode = "offline"
    if get_api_key() and not is_txt:
        try:
            prompt = (
                "You are a medical-report DATA EXTRACTOR. Read the attached report and return ONLY JSON with this shape:\n"
                '{"report_type":"lab|radiology|other","patient_sex":"male|female|unknown",'
                '"values":[{"name":"test name exactly as printed","value":number or null,"unit":"unit as printed",'
                '"ref_low":number or null,"ref_high":number or null,"flag":"H|L|N|"}],'
                '"narrative":"for radiology/imaging/clinical text: copy the findings and impression verbatim, otherwise empty string"}\n'
                "Rules: copy numbers exactly as printed; do NOT interpret, correct or invent anything; skip fields you cannot read."
            )
            if is_pdf:
                if provider() == "groq":
                    if not text.strip():
                        raise LLMError("Groq cannot read scanned PDFs - upload a photo/screenshot instead (or use a Gemini key)")
                    files = []
                    prompt += "\n\nREPORT TEXT:\n" + text[:12000]
                else:
                    files = [(data, "application/pdf")]
            else:
                d, m = shrink_image(data)
                files = [(d, m)]
            out = parse_json(call_gemini(prompt, files=files, json_mode=True))
            mode = "ai"
            sx = (out.get("patient_sex") or "").lower()
            sex = sx if sx in ("male", "female") else None
            narrative = out.get("narrative") or ""
            seen = set()
            for v in out.get("values", []):
                name = str(v.get("name", "")).strip()
                val = v.get("value")
                if isinstance(val, str):
                    val = to_float(val)
                if not name or val is None:
                    continue
                key = match_param(name)
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                items.append({"key": key, "name": PARAMS[key].name if key else name, "value": float(val), "raw": f"{val:g}",
                              "unit": str(v.get("unit") or ""), "ref_lo": v.get("ref_low"), "ref_hi": v.get("ref_high"),
                              "flag": (v.get("flag") or "").upper()[:1] if v.get("flag") in ("H", "L") else "", "line": name})
            if is_pdf and not text:
                text = pdf_text(data)
        except LLMError as e:
            notes.append(f"AI extraction unavailable ({e}); used offline extraction instead.")
        except Exception as e:  # json problems etc.
            notes.append(f"AI extraction could not be parsed ({type(e).__name__}); used offline extraction instead.")
    if mode == "offline":
        if not text.strip():
            notes.append("No readable text found. For scanned PDFs/photos add a free Gemini API key in the sidebar, or upload a clear photo.")
        items = regex_extract(text)
        narrative = "" if items else text.strip()[:4000]
    else:
        # Cross-check: regex over any text layer can add values AI missed
        if text:
            known = {i["key"] for i in items if i["key"]}
            for it in regex_extract(text):
                if it["key"] not in known:
                    items.append(it)
    if not sex and text:
        sex = detect_sex(text)
    return {"items": items, "narrative": narrative, "text": text, "sex": sex, "mode": mode, "notes": notes}


def analysis_agent(items, sex_choice, detected_sex):
    """Agent 2 - Compare values with report range (preferred) or standard adult range."""
    sex = sex_choice if sex_choice in ("male", "female") else detected_sex
    rows = []
    for it in items:
        p = PARAMS.get(it["key"]) if it["key"] else None
        v = float(it["value"])
        unit = it.get("unit") or (p.unit if p else "")
        if p and p.key in ("wbc", "plt") and v >= 1000:
            v, unit = v / 1000.0, p.unit
        if p and p.key == "rbc" and v >= 100000:
            v, unit = v / 1e6, p.unit
        lo, hi, src = it.get("ref_lo"), it.get("ref_hi"), "report"
        lo = to_float(lo) if lo is not None else None
        hi = to_float(hi) if hi is not None else None
        if p and p.key in ("wbc", "plt"):
            if lo and lo >= 1000: lo /= 1000.0
            if hi and hi >= 1000: hi /= 1000.0
        if lo is not None and hi is not None and lo > hi:
            lo = hi = None
        if lo is None and hi is None:
            if p:
                lo, hi = p.range_for(sex)
                src = "standard" + (f" ({sex})" if sex and p.sex_specific() else "")
            else:
                src = "n/a"
        unit_issue = ""
        if p and p.units and it.get("unit"):
            nu = norm_unit(it["unit"])
            if nu not in p.units:
                unit_issue = f"Unit '{it['unit']}' differs from the usual '{p.unit}' - value not classified; check with your lab."
        status = "Unknown"
        if lo is not None or hi is not None:
            if hi is not None and v > hi:
                status = "High"
            elif lo is not None and v < lo:
                status = "Low"
            else:
                status = "Normal"
        if unit_issue:
            status = "Check unit"
        severity = ""
        if status in ("High", "Low"):
            if p and ((p.clo is not None and v <= p.clo) or (p.chi is not None and v >= p.chi)):
                severity = "critical"
            else:
                ref = hi if status == "High" else lo
                dev = abs(v - ref) / abs(ref) if ref else 0
                severity = "marked" if dev > 0.25 else "mild"
        rows.append({"key": it["key"], "name": it["name"], "value": v, "unit": unit, "ref_lo": lo, "ref_hi": hi, "ref_src": src,
                     "status": status, "severity": severity, "report_flag": it.get("flag", ""), "unit_issue": unit_issue,
                     "raw": it.get("raw", ""), "group": p.group if p else "Other"})
    return rows, sex


def fmt_ref(lo, hi):
    if lo is None and hi is None:
        return "-"
    if lo is None:
        return f"< {hi:g}"
    if hi is None:
        return f"> {lo:g}"
    return f"{lo:g} - {hi:g}"


def rag_agent(rows, narrative):
    """Agent 3 - Retrieve trusted knowledge for abnormal findings (TF-IDF over curated KB)."""
    docs, vec, mat = kb_index()
    by_key = {d["key"]: d for d in docs if d["key"]}
    picked, seen = [], set()

    def push(d, score, why):
        if d["id"] not in seen:
            seen.add(d["id"])
            picked.append({"id": d["id"], "title": d["title"], "text": d["text"], "score": round(float(score), 3), "why": why})

    abnormal = [r for r in rows if r["status"] in ("High", "Low") and r["key"]]
    for r in abnormal:
        push(by_key[r["key"]], 1.0, f"{r['name']} is {r['status'].lower()}")
    if abnormal:
        q = " ".join(f"{r['status'].lower()} {r['name']} {PARAMS[r['key']].aliases[0].replace(chr(92)+'s*', ' ')}" for r in abnormal)
        sims = cosine_similarity(vec.transform([q]), mat)[0]
        for i in sims.argsort()[::-1][:6]:
            if docs[i]["key"] is None and sims[i] > 0.12:
                push(docs[i], sims[i], "pattern match")
    if narrative.strip():
        sims = cosine_similarity(vec.transform([narrative[:3000]]), mat)[0]
        for i in sims.argsort()[::-1][:4]:
            if docs[i]["key"] is None and sims[i] > 0.08:
                push(docs[i], sims[i], "report text match")
    return picked


def verification_agent(rows, text, explanation_ctx):
    """Agent 4 - Cross-check extraction + analysis before anything is explained."""
    issues, ok = [], 0
    clean = (text or "").replace(",", "")
    for r in rows:
        flag_note = []
        in_src = None
        if clean:
            cands = {r["raw"].replace(",", ""), f"{r['value']:g}", f"{r['value']:.1f}", f"{r['value']:.2f}"}
            in_src = any(c and c in clean for c in cands)
            if not in_src:
                flag_note.append("value not found in the report text layer")
        p = PARAMS.get(r["key"]) if r["key"] else None
        if p and r["status"] in ("High", "Low"):
            lo, hi = p.lo, p.hi
            if (hi and r["value"] > hi * 8) or (lo and lo > 0 and r["value"] < lo / 8):
                flag_note.append("value looks physiologically implausible - possible extraction/OCR error")
        if r["report_flag"] in ("H", "L") and r["status"] in ("High", "Low", "Normal"):
            exp = "H" if r["status"] == "High" else "L" if r["status"] == "Low" else "N"
            if exp != r["report_flag"]:
                flag_note.append(f"report marks '{r['report_flag']}' but our comparison says {r['status']}")
        if r["unit_issue"]:
            flag_note.append(r["unit_issue"])
        r["verified"] = "✅" if not flag_note and in_src is not False else ("⚠️" if flag_note else "✅")
        r["verify_note"] = "; ".join(flag_note)
        if flag_note:
            issues.append(f"**{r['name']}**: " + "; ".join(flag_note))
        else:
            ok += 1
    if not explanation_ctx and any(r["status"] in ("High", "Low") for r in rows):
        issues.append("No knowledge-base context found for some abnormal values; explanation will be limited.")
    level = "High" if not issues else ("Medium" if len(issues) <= max(1, len(rows) // 4) else "Low")
    return {"issues": issues, "ok": ok, "total": len(rows), "level": level}


LANG = {
    "English": {
        "summary": "Summary", "all_ok": "All the values we could read are inside the usual range.",
        "n_abn": "{n} value(s) are outside the usual range.", "stands": "What stands out", "normal": "Values in the usual range",
        "next": "What to do next", "urgent": "🔴 One or more values are in a range that may need prompt medical attention. Please contact a doctor today, or go to an emergency department if you feel unwell.",
        "n1": "Show this report to your doctor - only a doctor can interpret it together with your symptoms, history and examination.",
        "n2": "Do not start, stop or change any medicine because of this summary.",
        "n3": "Questions to ask: What could be causing this? Do I need repeat or extra tests? Do I need diet or lifestyle changes?",
        "is": "is", "low": "low", "high": "high", "yours": "Your value", "ref": "usual range", "note_en": ""},
    "Roman Urdu": {
        "summary": "Khulasa", "all_ok": "Jo values parhi ja sakin woh sab normal range mein hain.",
        "n_abn": "{n} value(s) normal range se bahar hain.", "stands": "Khaas baatein", "normal": "Normal range wali values",
        "next": "Ab kya karein", "urgent": "🔴 Ek ya zyada values aisi range mein hain jahan fori doctor se rujoo karna zaroori ho sakta hai. Aaj hi doctor se milein, ya tabiyat kharab ho to emergency jayein.",
        "n1": "Ye report apne doctor ko dikhayein - sirf doctor aap ki alamaat, history aur muayine ke saath isay sahi samajh sakta hai.",
        "n2": "Is khulase ki bunyad par koi dawai shuru, band ya tabdeel na karein.",
        "n3": "Doctor se poochein: Iski wajah kya ho sakti hai? Kya dobara ya mazeed tests chahiye? Kya khurak ya lifestyle mein tabdeeli chahiye?",
        "is": "hai", "low": "kam", "high": "zyada", "yours": "Aap ki value", "ref": "normal range",
        "note_en": "(Detailed explanation English mein hai - poora Roman Urdu/Urdu tarjuma AI key ke saath milta hai.)"},
    "اردو": {
        "summary": "خلاصہ", "all_ok": "جو ویلیوز پڑھی جا سکیں وہ سب نارمل رینج میں ہیں۔",
        "n_abn": "{n} ویلیو(ز) نارمل رینج سے باہر ہیں۔", "stands": "اہم باتیں", "normal": "نارمل رینج والی ویلیوز",
        "next": "اب کیا کریں", "urgent": "🔴 ایک یا زیادہ ویلیوز ایسی رینج میں ہیں جہاں فوری طبی توجہ کی ضرورت ہو سکتی ہے۔ آج ہی ڈاکٹر سے رابطہ کریں، یا طبیعت خراب ہو تو ایمرجنسی جائیں۔",
        "n1": "یہ رپورٹ اپنے ڈاکٹر کو دکھائیں - صرف ڈاکٹر آپ کی علامات، ہسٹری اور معائنے کے ساتھ اسے درست سمجھ سکتا ہے۔",
        "n2": "اس خلاصے کی بنیاد پر کوئی دوا شروع، بند یا تبدیل نہ کریں۔",
        "n3": "ڈاکٹر سے پوچھیں: اس کی وجہ کیا ہو سکتی ہے؟ کیا دوبارہ یا مزید ٹیسٹ چاہئیں؟ کیا خوراک یا طرزِ زندگی میں تبدیلی چاہیے؟",
        "is": "ہے", "low": "کم", "high": "زیادہ", "yours": "آپ کی ویلیو", "ref": "نارمل رینج",
        "note_en": "(تفصیلی وضاحت انگریزی میں ہے - مکمل اردو ترجمہ AI کلید کے ساتھ دستیاب ہے۔)"},
}

BAD_OUTPUT = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|tablets?|capsules?|iu)\b(?!\s*/)|\byou (?:definitely |certainly )?have (?:been diagnosed|got)\b|\bstart taking\b|\bstop taking\b",
    re.I)


def template_explanation(rows, rag, lang):
    L = LANG[lang]
    abn = [r for r in rows if r["status"] in ("High", "Low")]
    nor = [r for r in rows if r["status"] == "Normal"]
    by_key = {d["id"]: d for d in rag}
    out = [f"### {L['summary']}"]
    out.append(L["n_abn"].format(n=len(abn)) if abn else L["all_ok"])
    if lang != "English" and abn:
        out.append(f"_{L['note_en']}_")
    if abn:
        out.append(f"\n### {L['stands']}")
        for r in abn:
            p = PARAMS.get(r["key"]) if r["key"] else None
            word = L["high"] if r["status"] == "High" else L["low"]
            out.append(f"**{r['name']}** - {L['yours']}: **{r['value']:g} {r['unit']}** ({word}; {L['ref']}: {fmt_ref(r['ref_lo'], r['ref_hi'])})")
            if p:
                out.append(f"- {p.what}")
                out.append(f"- {p.low_txt if r['status'] == 'Low' else p.high_txt}")
        for d in rag:
            if d["id"].startswith("extra:") and d["why"] == "pattern match":
                out.append(f"- 🔎 **{d['title']}**: {d['text']}")
    if nor:
        out.append(f"\n### {L['normal']}")
        out.append(", ".join(f"{r['name']} ({r['value']:g})" for r in nor))
    return "\n".join(out)


def explanation_agent(rows, rag, narrative, lang, mode_ai):
    """Agent 5 - Plain-language explanation, grounded ONLY in retrieved context."""
    if mode_ai and get_api_key():
        findings = [{"test": r["name"], "value": r["value"], "unit": r["unit"], "range": fmt_ref(r["ref_lo"], r["ref_hi"]),
                     "status": r["status"], "severity": r["severity"]} for r in rows if r["status"] != "Check unit"]
        context = "\n".join(f"- {d['title']}: {d['text']}" for d in rag)
        style = {"English": "simple English", "Roman Urdu": "Roman Urdu (Urdu written in English/Latin letters, simple everyday words)",
                 "اردو": "simple Urdu in Urdu script"}[lang]
        system = ("You are a careful medical-information explainer for ordinary people in Pakistan. "
                  "Use ONLY the FINDINGS and CONTEXT given. NEVER diagnose, NEVER name or suggest medicines, doses, supplements or treatments, "
                  "NEVER tell the user to start/stop/change any medicine. Use hedged wording ('can', 'may'). Keep medical terms but explain them. Be calm and kind. "
                  "If something is not in the context, say a doctor needs to check it.")
        prompt = (f"Write the explanation in {style}.\nStructure with markdown headings: Summary (2-3 sentences) / What stands out (each abnormal item: what it is, what low/high may mean) / "
                  f"Values in the usual range (one short line) / What to do next (see a doctor, bring the report, 3 questions to ask). "
                  f"If any severity is 'critical', start with a clear urgent-care note.\n\nFINDINGS:\n{json.dumps(findings, ensure_ascii=False)}\n\n"
                  f"REPORT TEXT (imaging/clinical narrative, may be empty):\n{narrative[:3000]}\n\nCONTEXT (trusted knowledge base):\n{context}")
        try:
            txt = call_gemini(prompt, system=system)
            if BAD_OUTPUT.search(txt):
                raise LLMError("guardrail: unsafe phrasing detected")
            return txt, "ai"
        except LLMError as e:
            return template_explanation(rows, rag, lang) + f"\n\n_(AI explanation unavailable: {e}. Showing rule-based summary.)_", "template"
    txt = template_explanation(rows, rag, lang)
    if narrative.strip() and not rows:
        txt += "\n\n### Report text\n" + narrative[:2500]
        if rag:
            txt += "\n\n### Plain-language notes\n" + "\n".join(f"- **{d['title']}**: {d['text']}" for d in rag)
    return txt, "template"


def guide_agent(rows, lang):
    """Agent 6 - Urgency + next steps."""
    L = LANG[lang]
    sev = [r["severity"] for r in rows if r["status"] in ("High", "Low")]
    if "critical" in sev:
        level, badge = "urgent", "🔴 Urgent - see a doctor today"
    elif "marked" in sev:
        level, badge = "soon", "🟠 See a doctor soon"
    elif sev:
        level, badge = "routine", "🟡 Discuss at your next visit"
    else:
        level, badge = "ok", "🟢 No out-of-range values detected"
    steps = [L["n1"], L["n2"], L["n3"]]
    return {"level": level, "badge": badge, "steps": steps, "urgent_text": L["urgent"] if level == "urgent" else ""}


def run_report(data, mime, filename, sex_choice, lang, status):
    ex = extraction_agent(data, mime, filename)
    status.write(f"🔍 **Extraction Agent** ({'AI vision' if ex['mode']=='ai' else 'OCR/text + rules'}) - {len(ex['items'])} value(s) found")
    rows, sex = analysis_agent(ex["items"], sex_choice, ex["sex"])
    n_abn = sum(r["status"] in ("High", "Low") for r in rows)
    status.write(f"🧠 **Analysis Agent** - {n_abn} abnormal of {len(rows)}")
    rag = rag_agent(rows, ex["narrative"])
    status.write(f"📚 **Medical RAG Agent** - {len(rag)} knowledge snippet(s) retrieved")
    ver = verification_agent(rows, ex["text"], rag)
    status.write(f"✅ **Verification Agent** - confidence: {ver['level']} ({len(ver['issues'])} issue(s))")
    expl, how = explanation_agent(rows, rag, ex["narrative"], lang, ex["mode"] == "ai" or bool(get_api_key()))
    status.write(f"🗣️ **Explanation Agent** ({'AI' if how=='ai' else 'rule-based'}, {lang})")
    guide = guide_agent(rows, lang)
    status.write("🧭 **Guidance Agent** - next steps ready")
    return {"ex": ex, "rows": rows, "sex": sex, "rag": rag, "ver": ver, "expl": expl, "how": how, "guide": guide, "lang": lang}


def esc(x):
    import html
    return html.escape(str(x))


def axis_pos(lo, hi, v):
    """Return (zone_left%, zone_width%, marker%) for the range bar."""
    if lo is not None and hi is not None and hi > lo:
        span = hi - lo
        a, b = lo - span * 0.9, hi + span * 0.9
        if lo >= 0 and a < 0:
            a = 0
    elif hi is not None:
        a, b = 0, hi * 1.7
    elif lo is not None:
        a, b = 0, lo * 2.2
    else:
        return None
    if b <= a:
        return None
    z0 = ((lo if lo is not None else a) - a) / (b - a)
    z1 = ((hi if hi is not None else b) - a) / (b - a)
    m = min(max((v - a) / (b - a), 0.02), 0.98)
    return z0 * 100, max(z1 - z0, 0.02) * 100, m * 100


SHORT = {"hb": "Hb", "wbc": "WBC", "plt": "PLT", "rbc": "RBC", "hct": "PCV", "mcv": "MCV", "mch": "MCH", "mchc": "MCHC", "rdw": "RDW",
         "neut": "Neut", "lymph": "Lymph", "mono": "Mono", "eos": "Eos", "baso": "Baso", "esr": "ESR", "fbs": "FBS", "rbs": "RBS",
         "hba1c": "HbA1c", "creat": "Creat", "urea": "Urea", "bun": "BUN", "uric": "Uric", "alt": "ALT", "ast": "AST", "alp": "ALP",
         "tbil": "T.Bil", "dbil": "D.Bil", "alb": "Alb", "tp": "T.Prot", "chol": "Chol", "ldl": "LDL", "hdl": "HDL", "tg": "TG",
         "na": "Na", "k": "K", "cl": "Cl", "ca": "Ca", "tsh": "TSH", "vitd": "Vit D", "b12": "B12", "ferr": "Ferritin", "iron": "Iron", "crp": "CRP"}


def state_of(r):
    if r["status"] == "Normal":
        return "ok"
    if r["status"] in ("High", "Low"):
        return "crit" if r["severity"] == "critical" else "out"
    return "na"


def pocket_strip(rows):
    cells = []
    for r in rows:
        stt = state_of(r)
        lab = SHORT.get(r["key"]) or r["name"].split(" (")[0][:7]
        cells.append(f'<div class="pk" title="{esc(r["name"])}: {r["value"]:g} {esc(r["unit"])}"><div class="dome {stt}">{r["value"]:g}</div><span>{esc(lab)}</span></div>')
    st.markdown('<div class="sec">Your report at a glance</div><div class="foil">' + "".join(cells) + "</div>"
                '<div class="legend"><span><i class="ok"></i>In usual range</span><span><i class="out"></i>Outside range</span><span><i class="crit"></i>Needs urgent attention</span></div>',
                unsafe_allow_html=True)


def range_row(r):
    st_ = r["status"]
    state = state_of(r)
    if st_ == "Normal":
        label, icon = "Normal", "✓"
    elif st_ in ("High", "Low"):
        label = ("Critical " if state == "crit" else "") + st_ + (" - marked" if r["severity"] == "marked" else "")
        icon = "▲" if st_ == "High" else "▼"
    else:
        label, icon = st_, "?"
    pos = axis_pos(r["ref_lo"], r["ref_hi"], r["value"]) if st_ != "Check unit" else None
    bar = ""
    if pos:
        bar = f'<div class="bar"><div class="zone" style="left:{pos[0]:.1f}%;width:{pos[1]:.1f}%"></div><div class="mk {state}" style="left:{pos[2]:.1f}%"></div></div>'
    note = f'<div class="vnote">⚠ {esc(r["verify_note"])}</div>' if r.get("verify_note") else ""
    return (f'<div class="rr {state}"><div class="ic {state}">{icon}</div>'
            f'<div class="rr-top"><div class="rr-name">{esc(r["name"])}</div><span class="chip {state}">{esc(label)}</span></div>'
            f'<div class="rr-val"><b>{r["value"]:g}</b> {esc(r["unit"])}<em>Usual range: {esc(fmt_ref(r["ref_lo"], r["ref_hi"]))}</em></div>{bar}{note}</div>')


def render_rows(rows):
    group = None
    out = []
    for r in sorted(rows, key=lambda x: x["group"]):
        if r["group"] != group:
            group = r["group"]
            out.append(f'<div class="grp">{esc(group)}</div>')
        out.append(range_row(r))
    st.markdown("".join(out), unsafe_allow_html=True)


def render_report(res):
    g, ver, rows = res["guide"], res["ver"], res["rows"]
    lang = res["lang"]
    body = f'<div class="t">{esc(g["badge"])}</div>' + (f'<div>{esc(g["urgent_text"])}</div>' if g["urgent_text"] else "")
    st.markdown(f'<div class="banner {g["level"]}">{body}</div>', unsafe_allow_html=True)
    for n in res["ex"]["notes"]:
        st.caption("ℹ️ " + n)
    n_abn = sum(r["status"] in ("High", "Low") for r in rows)
    n_ok = sum(r["status"] == "Normal" for r in rows)
    st.markdown(
        f'<div class="tiles"><div class="tile"><b>{len(rows)}</b><span>Tests read</span></div>'
        f'<div class="tile good"><b>{n_ok}</b><span>In usual range</span></div>'
        f'<div class="tile bad"><b>{n_abn}</b><span>Outside range</span></div>'
        f'<div class="tile"><b>{esc(ver["level"])}</b><span>Check confidence</span></div></div>', unsafe_allow_html=True)
    if rows:
        pocket_strip(rows)

    attention = [r for r in rows if r["status"] != "Normal"]
    if attention:
        st.markdown('<div class="sec">Needs attention</div>', unsafe_allow_html=True)
        render_rows(attention)
    with st.container(border=True):
        st.markdown("### Explanation" if lang != "اردو" else "### وضاحت")
        if lang == "اردو":
            st.markdown('<div class="rtl">\n\n' + res["expl"] + "\n\n</div>", unsafe_allow_html=True)
        else:
            st.markdown(res["expl"])
    with st.container(border=True):
        st.markdown("### " + LANG[lang]["next"])
        txt = "\n".join(f"- {s}" for s in g["steps"])
        st.markdown(('<div class="rtl">\n\n' + txt + "\n\n</div>") if lang == "اردو" else txt, unsafe_allow_html=True)
    normal = [r for r in rows if r["status"] == "Normal"]
    if normal:
        with st.expander(f"In usual range ({len(normal)})"):
            render_rows(normal)
    if rows:
        with st.expander("Table view"):
            df = pd.DataFrame([{"Test": r["name"], "Your value": f"{r['value']:g}", "Unit": r["unit"], "Reference": fmt_ref(r["ref_lo"], r["ref_hi"]),
                                "Range from": r["ref_src"], "Status": r["status"] + (f" ({r['severity']})" if r["severity"] else ""),
                                "Checked": r.get("verified", "")} for r in rows])
            st.dataframe(df, width="stretch", hide_index=True)
        st.caption((f"Sex used for ranges: {res['sex']}. " if res["sex"] else "Sex not detected, so general adult ranges were used for sex-specific tests (pick sex in the sidebar). ")
                   + "Ranges printed on your report are preferred over standard adult ranges.")
    with st.expander("Verification Agent details"):
        st.write(f"Confidence: **{ver['level']}**. {ver['ok']} of {ver['total']} values passed every check.")
        for i in ver["issues"]:
            st.markdown("- ⚠️ " + i)
        if not ver["issues"]:
            st.write("No inconsistencies found.")
    with st.expander("Knowledge retrieved by the RAG Agent"):
        if not res["rag"]:
            st.write("Nothing retrieved (no abnormal values).")
        for d in res["rag"]:
            st.markdown(f"**{d['title']}** _(relevance {d['score']}, {d['why']})_  \n{d['text']}")
    if res["ex"]["text"]:
        with st.expander("Raw text read from the report"):
            st.text(res["ex"]["text"][:5000])


# ----------------------------------------------------------------------------------------------
# 4. AGENTS - MEDICINE PIPELINE
# ----------------------------------------------------------------------------------------------
def norm(s):
    return re.sub(r"[^a-z0-9+ ]", " ", str(s).lower()).strip()


@st.cache_data
def load_meds():
    here = Path(__file__).parent
    path = here / "Medicine_research.xlsx"
    if not path.exists():
        cands = list(here.glob("*.xlsx")) + list(here.glob("*.csv"))
        if not cands:
            return None
        path = cands[0]
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path, sheet_name=0)
    df = df.dropna(how="all").iloc[:, :7]
    df.columns = ["generic", "brands", "active", "mclass", "use", "form", "source"][:df.shape[1]]
    df = df.fillna("").astype(str).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        brands = [b.strip() for b in re.split(r"/|,", r["brands"]) if b.strip() and "various" not in b.lower()]
        generic_clean = re.sub(r"\(.*?\)", "", r["generic"]).strip()
        parts = [x.strip() for x in re.split(r"\+", r["active"]) if x.strip()]
        terms = set()
        for t in brands + [generic_clean, r["generic"]] + parts + re.split(r"\+", generic_clean):
            n = norm(t)
            if len(n) >= 3:
                terms.add(n)
        rows.append({"idx": i, "terms": sorted(terms), "parts": [norm(p) for p in parts], "brands": brands})
    return df, rows


def match_medicine(cands, free_text="", top=6):
    data = load_meds()
    if not data:
        return []
    df, idx = data
    cands = [norm(c) for c in cands if c and norm(c)]
    ft = norm(free_text)
    out = []
    for r in idx:
        best = 0.0
        for t in r["terms"]:
            for c in cands:
                s = difflib.SequenceMatcher(None, c, t).ratio()
                if len(t) >= 4 and re.search(rf"\b{re.escape(t)}\b", c):
                    s = max(s, 0.9)
                if len(c) >= 4 and re.search(rf"\b{re.escape(c)}\b", t):
                    s = max(s, 0.85)
                best = max(best, s)
            if ft and len(t) >= 4 and re.search(rf"\b{re.escape(t)}\b", ft):
                best = max(best, 0.88)
        if best >= 0.7:
            hits = sum(1 for p in r["parts"] if p and len(p.split()[0]) >= 4 and (p.split()[0] in ft or any(p.split()[0] in c for c in cands)))
            out.append((min(best + 0.02 * hits, 1.0), r["idx"]))
    out.sort(key=lambda x: -x[0])
    return [(s, df.iloc[i].to_dict()) for s, i in out[:top]]


def medicine_extract_agent(data):
    """Agent M1 - read pack/strip photo."""
    d, m = shrink_image(data)
    if get_api_key():
        prompt = ("Read this photo of a medicine pack/strip/bottle. Return ONLY JSON: "
                  '{"brand_name":"","generic_names":["active ingredient(s) printed"],"strength":"e.g. 500 mg","form":"tablet|capsule|syrup|injection|cream|other",'
                  '"manufacturer":"","visible_text":"other legible text","readable":true|false,"confidence":0.0-1.0,"notes":"why unclear, if so"}. '
                  "Only report what is actually visible. Never guess a name you cannot read; use readable=false and a low confidence instead.")
        try:
            out = parse_json(call_gemini(prompt, files=[(d, m)], json_mode=True))
            out["mode"] = "ai"
            return out
        except LLMError as e:
            err = str(e)
        except Exception as e:
            err = type(e).__name__
    else:
        err = ""
    text = ocr_image(d)
    return {"brand_name": "", "generic_names": [], "strength": "", "form": "", "manufacturer": "", "visible_text": text, "readable": bool(text.strip()),
            "confidence": 0.5 if text.strip() else 0.0, "notes": ("AI unavailable: " + err) if err else "Offline OCR used (add a free Gemini key for much better pack reading).", "mode": "ocr"}


CLASS_NOTES = {
    "antibiotic": "Antibiotics work only against bacteria and only when prescribed. Taking them without a doctor's advice, or stopping early, can cause resistance and side effects.",
    "antitubercular": "TB treatment must follow a specialist's regimen for the full duration; never self-start or stop.",
    "corticosteroid": "Steroids have important side effects and must not be started or stopped suddenly without a doctor's advice.",
    "nsaid": "NSAIDs can irritate the stomach and affect kidneys; people with ulcers, kidney disease, asthma, or on blood thinners should ask a doctor/pharmacist first.",
    "antifungal": "Antifungals are chosen by site and type of infection; some interact with other medicines - check with a pharmacist.",
    "anthelmintic": "Dose and repeat depend on the type of worm and age/weight; follow the prescriber.",
    "antihistamine": "Many antihistamines cause drowsiness (especially older types); be careful with driving or machinery.",
    "bronchodilator": "Inhalers need correct technique; frequent need for a reliever inhaler means asthma/COPD control should be reviewed by a doctor.",
    "antiviral": "Antivirals work best when started early and as prescribed by a doctor.",
    "analgesic": "Do not combine several products containing the same painkiller (e.g. paracetamol) - this can cause overdose.",
}


def class_note(mclass):
    m = mclass.lower()
    for k, v in CLASS_NOTES.items():
        if k in m:
            return v
    return ""


def show_medicine(row, pack=None):
    brands = "".join(f"<span>{esc(x.strip())}</span>" for x in re.split(r"/|,", row["brands"]) if x.strip())
    pack_strength = f'<div><label>Strength read from your pack</label><p>{esc(pack["strength"])}</p></div>' if pack and pack.get("strength") else ""
    st.markdown(
        f'<div class="med"><div class="med-head"><div><div class="med-name">{esc(row["generic"])}</div><div class="med-class">{esc(row["mclass"])}</div></div>'
        f'<div class="brands">{brands}</div></div><div class="med-grid">'
        f'<div><label>Active ingredient</label><p>{esc(row["active"])}</p></div>'
        f'<div><label>What it is generally used for</label><p>{esc(row["use"])}</p></div>'
        f'<div><label>Typical form and strength</label><p>{esc(row["form"])}</p></div>{pack_strength}</div></div>', unsafe_allow_html=True)
    n = class_note(row["mclass"])
    if n:
        st.info("ℹ️ " + n)
    data = load_meds()
    if data:
        df, _ = data
        same = df[(df["active"].str.lower() == row["active"].lower()) & (df["generic"] != row["generic"])]
        if len(same):
            st.markdown("**Other listed entries with the same active ingredient:** " + ", ".join(f"{r.generic} ({r.brands})" for r in same.itertuples()))
    if row.get("source", "").startswith("http"):
        st.caption(f"Source: {row['source']}")
    if get_api_key():
        key = "alt_" + hashlib.md5((row["generic"] + str(pack and pack.get("strength"))).encode()).hexdigest()
        with st.expander("🏷️ More brands with the same ingredient/strength (AI-suggested, unverified)"):
            if key not in st.session_state:
                try:
                    out = parse_json(call_gemini(
                        f"List up to 6 commonly available brand names in Pakistan that contain {row['active']}"
                        + (f" at strength {pack['strength']}" if pack and pack.get("strength") else "")
                        + ' in the same dosage form family. Return ONLY JSON {"brands":["..."]}. If you are not confident, return an empty list.',
                        json_mode=True))
                    st.session_state[key] = out.get("brands", [])
                except Exception as e:
                    st.session_state[key] = None
            b = st.session_state.get(key)
            if b:
                st.write(", ".join(b))
                st.caption("AI-generated list - not verified against DRAP. Ask your pharmacist to confirm that a brand has the same ingredient and strength before substituting.")
            else:
                st.write("No reliable suggestions available.")
    st.markdown('<div class="safe"><b>Confirm with a pharmacist or doctor</b> before taking, switching or stopping any medicine. MediLens AI never tells you to take a medicine or change a prescription.</div>', unsafe_allow_html=True)


def medicine_tab():
    data = load_meds()
    if not data:
        st.error("Medicine_research.xlsx not found in the repository. Upload it next to app.py on GitHub.")
        return
    df, _ = data
    st.caption(f"Medicine database: {len(df)} entries (Medicine_research.xlsx, DRAP Pakistan-based).")
    src = st.radio("Input", ["📤 Upload photo", "📷 Take photo", "⌨️ Type name"], horizontal=True)
    pack, free, cands, ready = None, "", [], False
    if src == "⌨️ Type name":
        q = st.text_input("Medicine name (brand or generic)", placeholder="e.g. Panadol, Augmentin, Cetirizine")
        if q:
            cands, ready = [q], True
    else:
        f = st.file_uploader("Photo of medicine pack/strip", type=["jpg", "jpeg", "png", "webp"]) if src == "📤 Upload photo" else st.camera_input("Take a clear photo of the pack")
        if f is not None:
            data_b = f.getvalue()
            fid = hashlib.md5(data_b).hexdigest()
            if st.session_state.get("med_fid") != fid:
                with st.spinner("💊 Medicine Agent is reading the pack..."):
                    st.session_state["med_pack"] = medicine_extract_agent(data_b)
                st.session_state["med_fid"] = fid
            pack = st.session_state["med_pack"]
            st.image(data_b, width=260)
            cands = [pack.get("brand_name", "")] + list(pack.get("generic_names") or [])
            free = pack.get("visible_text", "") or ""
            ready = True
            with st.expander("🔍 What the agent read from the image"):
                st.json({k: v for k, v in pack.items() if k != "visible_text"})
                if free:
                    st.text(free[:1500])
    if not ready:
        return
    matches = match_medicine(cands, free)
    if not matches:
        st.warning("No match in the medicine database. The image may be unclear or the medicine may not be listed. Try a clearer photo, or type the name.")
        return
    top_score = matches[0][0]
    conf = float((pack or {}).get("confidence") or 1.0)
    unsure = (pack is not None) and (not pack.get("readable", True) or conf < 0.6 or top_score < 0.8)
    labels = [f"{r['generic']} - {r['brands']} (match {s:.0%})" for s, r in matches]
    if unsure:
        st.warning("🤔 I am not fully sure about this medicine. Please **confirm** the correct one below (or retake a clearer photo).")
        choice = st.selectbox("Which of these is your medicine?", ["- select -"] + labels + ["None of these"])
        if choice == "- select -":
            return
        if choice == "None of these":
            st.info("Please retake a clearer photo (good light, label facing camera) or type the name.")
            return
        row = matches[labels.index(choice)][1]
    else:
        st.success(f"Best match: **{matches[0][1]['generic']}** (match {top_score:.0%})")
        if len(matches) > 1:
            alt = st.selectbox("Not right? Pick another match:", ["Use best match"] + labels[1:])
            row = matches[0][1] if alt == "Use best match" else matches[labels.index(alt)][1]
        else:
            row = matches[0][1]
    show_medicine(row, pack)



CSS = """
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@600;800&family=Instrument+Sans:wght@400;500;600;700&family=Noto+Naskh+Arabic:wght@400;600&display=swap');
:root{--ink:#16201B;--foil:#ECEEE8;--card:#FFFFFF;--green:#0F5C41;--dome-g:#1E9E62;--orange:#F2701D;--yellow:#FFD23F;--red:#D6322B;--muted:#5E6B63;--line:#CBD1C6;}
html,body,.stApp{font-family:'Instrument Sans',system-ui,sans-serif;color:var(--ink);}
.stApp{background:var(--foil);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
.block-container{padding-top:1.1rem;max-width:1080px;}
h1,h2,h3,h4{font-family:'Instrument Sans',sans-serif;font-weight:700;letter-spacing:-.01em;color:var(--ink);}
.stMarkdown h3{font-size:1.3rem;margin:.9rem 0 .3rem;}
.stMarkdown p,.stMarkdown li{line-height:1.65;font-size:1rem;}
/* hero */
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:24px;align-items:center;background:var(--green);color:#E9F5EE;border:2px solid var(--ink);border-radius:26px;box-shadow:7px 7px 0 var(--ink);padding:30px 32px;margin:4px 6px 26px 0;}
@media(max-width:800px){.hero{grid-template-columns:1fr;padding:22px 20px;box-shadow:5px 5px 0 var(--ink)}}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:1rem;color:#fff;margin-bottom:16px;}
.cap{width:44px;height:20px;border-radius:99px;border:2px solid var(--ink);background:linear-gradient(90deg,var(--yellow) 50%,#fff 50%);transform:rotate(-18deg);}
.hero h1{font-family:'Unbounded',sans-serif;font-weight:800;color:#fff;font-size:2.15rem;line-height:1.12;margin:0 0 12px;letter-spacing:-.02em;}
@media(max-width:800px){.hero h1{font-size:1.55rem}}
.hero .sub{font-size:1.05rem;line-height:1.55;color:#CDE8DA;max-width:520px;margin-bottom:16px;}
.hero .ur{font-family:'Noto Naskh Arabic',serif;font-size:1.15rem;color:var(--yellow);margin-bottom:14px;}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}
.flow span{font-size:.8rem;font-weight:600;padding:4px 11px;border-radius:99px;background:rgba(255,255,255,.1);border:1.5px solid rgba(255,255,255,.35);color:#fff;}
.flow i{color:var(--yellow);font-style:normal;font-weight:700;}
/* blister foil */
.foil{background:linear-gradient(135deg,#F7F8F4 0%,#D4D9D0 38%,#F0F2EC 58%,#C7CDC3 100%);border:2px solid var(--ink);border-radius:20px;padding:18px 14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(76px,1fr));gap:16px 8px;box-shadow:5px 5px 0 var(--ink);}
.hero .foil{grid-template-columns:repeat(4,1fr);transform:rotate(3deg);gap:12px 6px;box-shadow:6px 6px 0 rgba(0,0,0,.45);}
@media(max-width:800px){.hero .foil{transform:none;max-width:340px}}
.pk{text-align:center;font-size:.78rem;font-weight:700;color:var(--ink);}
.pk span{display:block;margin-top:5px;line-height:1.1;}
.dome{width:56px;height:56px;margin:0 auto;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:.82rem;border:1.5px solid rgba(22,32,27,.55);
box-shadow:inset 0 -7px 10px rgba(0,0,0,.2),inset 0 5px 7px rgba(255,255,255,.55),0 2px 0 rgba(0,0,0,.25);}
.hero .dome{width:46px;height:46px}
.dome.ok{background:radial-gradient(circle at 35% 28%,#9BE8BE,#1E9E62 70%);color:#06311F}
.dome.out{background:radial-gradient(circle at 35% 28%,#FFD0A3,#F2701D 72%);color:#3B1A00}
.dome.crit{background:radial-gradient(circle at 35% 28%,#FFA79F,#D6322B 72%);color:#fff}
.dome.na{background:radial-gradient(circle at 35% 28%,#F2F2F2,#A3ABA6 72%);color:#2B312D}
.dome.empty{background:radial-gradient(circle at 35% 28%,#fff,#DDE1DA 75%);box-shadow:inset 0 -4px 8px rgba(0,0,0,.12),0 2px 0 rgba(0,0,0,.2)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.82rem;color:var(--muted);margin:10px 2px 0;}
.legend i{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px;border:1.5px solid var(--ink);}
.legend .ok{background:var(--dome-g)}.legend .out{background:var(--orange)}.legend .crit{background:var(--red)}
/* tabs as capsules */
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:none;margin-bottom:6px;}
.stTabs [data-baseweb="tab"]{font-weight:700;padding:8px 18px;border-radius:99px;border:2px solid var(--ink);background:#fff;height:auto;}
.stTabs [aria-selected="true"]{background:var(--ink);color:#fff;}
.stTabs [aria-selected="true"] p{color:#fff;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}
/* buttons + uploader */
.stButton>button{border-radius:14px;font-weight:700;border:2px solid var(--ink);padding:.55rem 1.2rem;background:#fff;color:var(--ink);box-shadow:3px 3px 0 var(--ink);transition:transform .08s,box-shadow .08s;}
.stButton>button:hover{transform:translate(1px,1px);box-shadow:2px 2px 0 var(--ink);border-color:var(--ink);color:var(--ink);}
.stButton>button[kind="primary"]{background:var(--yellow);color:var(--ink);}
.stButton>button[kind="primary"]:hover{background:#FFC400;color:var(--ink);}
[data-testid="stFileUploaderDropzone"]{border:2px dashed var(--ink);background:#fff;border-radius:18px;}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:2px solid var(--ink)!important;border-radius:18px;box-shadow:4px 4px 0 var(--ink);}
[data-testid="stSidebar"]{background:#E1E5DB;border-right:2px solid var(--ink);}
[data-testid="stExpander"]{border:2px solid var(--ink);border-radius:14px;background:#fff;}
/* banner + tiles */
.banner{border:2px solid var(--ink);border-radius:18px;padding:16px 20px;margin:6px 4px 16px 0;box-shadow:5px 5px 0 var(--ink);}
.banner .t{font-family:'Unbounded',sans-serif;font-weight:800;font-size:1.05rem;margin-bottom:3px;}
.banner.urgent{background:#FFD9D4}.banner.soon{background:#FFE2C2}.banner.routine{background:#FFF1B8}.banner.ok{background:#CDEFD9}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 4px 20px 0;}
@media(max-width:700px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{background:#fff;border:2px solid var(--ink);border-radius:16px;padding:12px 14px;box-shadow:3px 3px 0 var(--ink);}
.tile b{display:block;font-family:'Unbounded',sans-serif;font-size:1.6rem;line-height:1.15;}
.tile span{font-size:.82rem;color:var(--muted);font-weight:600;}
.tile.bad b{color:var(--red)}.tile.good b{color:var(--dome-g)}
.sec{font-weight:700;font-size:1.3rem;margin:22px 0 10px;}
/* range rows */
.grp{font-weight:700;color:var(--muted);margin:16px 0 6px;font-size:.92rem;}
.rr{display:grid;grid-template-columns:44px 1fr;gap:4px 14px;align-items:center;background:#fff;border:2px solid var(--ink);border-radius:16px;padding:12px 16px;margin:0 4px 10px 0;box-shadow:3px 3px 0 var(--ink);}
.rr .ic{grid-row:1 / span 3;width:40px;height:40px;border-radius:50%;display:grid;place-items:center;font-weight:800;border:1.5px solid rgba(22,32,27,.55);
box-shadow:inset 0 -5px 8px rgba(0,0,0,.18),inset 0 4px 6px rgba(255,255,255,.5);}
.ic.ok{background:radial-gradient(circle at 35% 28%,#9BE8BE,#1E9E62 70%);color:#06311F}
.ic.out{background:radial-gradient(circle at 35% 28%,#FFD0A3,#F2701D 72%);color:#3B1A00}
.ic.crit{background:radial-gradient(circle at 35% 28%,#FFA79F,#D6322B 72%);color:#fff}
.ic.na{background:radial-gradient(circle at 35% 28%,#F2F2F2,#A3ABA6 72%);color:#2B312D}
.rr-top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;}
.rr-name{font-weight:700;}
.chip{font-size:.76rem;font-weight:700;padding:3px 11px;border-radius:99px;white-space:nowrap;border:1.5px solid var(--ink);}
.chip.ok{background:#CDEFD9}.chip.out{background:#FFE2C2}.chip.crit{background:#FFD9D4}.chip.na{background:#E7E9E4}
.rr-val{color:var(--muted);font-size:.9rem;}
.rr-val b{font-family:'Unbounded',sans-serif;font-size:1.35rem;color:var(--ink);margin-right:3px;}
.rr-val em{font-style:normal;margin-left:10px;}
.bar{position:relative;height:12px;border-radius:99px;background:#E4E8DF;border:1.5px solid var(--ink);margin:6px 0 2px;}
.bar .zone{position:absolute;top:-1.5px;bottom:-1.5px;border-radius:99px;background:repeating-linear-gradient(135deg,#7FD3A6 0 6px,#9CE0BA 6px 12px);border:1.5px solid var(--ink);}
.bar .mk{position:absolute;top:-7px;width:20px;height:20px;border-radius:50%;transform:translateX(-10px);border:2px solid var(--ink);}
.mk.ok{background:var(--dome-g)}.mk.out{background:var(--orange)}.mk.crit{background:var(--red)}.mk.na{background:#A3ABA6}
.vnote{font-size:.8rem;color:var(--red);margin-top:6px;font-weight:600;}
/* medicine */
.med{background:#fff;border:2px solid var(--ink);border-radius:22px;box-shadow:6px 6px 0 var(--ink);padding:0;margin:10px 6px 18px 0;overflow:hidden;}
.med-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center;background:var(--yellow);border-bottom:2px solid var(--ink);padding:18px 24px;}
.med-name{font-family:'Unbounded',sans-serif;font-weight:800;font-size:1.45rem;line-height:1.15;}
.med-class{font-weight:600;margin-top:4px;}
.brands{display:flex;gap:6px;flex-wrap:wrap;}
.brands span{background:var(--ink);color:#fff;font-weight:700;font-size:.85rem;padding:5px 13px;border-radius:99px;}
.med-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px 30px;padding:20px 24px 22px;}
@media(max-width:700px){.med-grid{grid-template-columns:1fr}.med-head{padding:16px}}
.med-grid label{display:block;font-size:.8rem;color:var(--muted);font-weight:700;margin-bottom:2px;}
.med-grid p{margin:0;line-height:1.5;}
.safe{background:#FFE2C2;border:2px solid var(--ink);border-radius:14px;padding:11px 15px;font-size:.92rem;margin:4px 6px 8px 0;box-shadow:3px 3px 0 var(--ink);}
.disc{font-size:.8rem;color:var(--muted);text-align:center;margin-top:26px;padding-top:12px;border-top:2px solid var(--line);}
.rtl{direction:rtl;text-align:right;font-family:'Noto Naskh Arabic','Instrument Sans',serif;font-size:1.1rem;}
.pill{display:inline-block;font-size:.8rem;font-weight:700;padding:4px 12px;border-radius:99px;margin-bottom:6px;border:1.5px solid var(--ink);}
.pill.on{background:#CDEFD9}.pill.off{background:#E7E9E4}
"""

def _hero_strip():
    pat = ["ok", "ok", "out", "ok", "ok", "empty", "ok", "crit", "ok", "ok", "out", "empty"]
    return "".join(f'<div class="pk"><div class="dome {p}"></div></div>' for p in pat)


HERO = f"""
<div class="hero"><div>
<div class="brand"><span class="cap"></span>MediLens AI</div>
<div class="ur">آپ کی رپورٹ، آسان زبان میں</div>
<h1>Read your report. Know your medicine.</h1>
<div class="sub">Upload a lab report or a medicine photo. A team of AI agents reads it, double-checks it, and explains it in plain English, Roman Urdu or Urdu.</div>
<div class="flow"><span>Upload</span><i>›</i><span>Extract</span><i>›</i><span>Analyze</span><i>›</i><span>Retrieve</span><i>›</i><span>Verify</span><i>›</i><span>Explain</span><i>›</i><span>Guide</span></div>
</div><div class="foil">{_hero_strip()}</div></div>
"""


# ----------------------------------------------------------------------------------------------
# 5. UI
# ----------------------------------------------------------------------------------------------
def sidebar():
    st.sidebar.markdown("## 🩺 MediLens AI")
    st.sidebar.selectbox("Explanation language", list(LANG.keys()), key="lang")
    st.sidebar.selectbox("Patient sex (for reference ranges)", ["Auto-detect", "Male", "Female"], key="sex_choice")
    st.sidebar.text_input("Groq API key (optional, free)", type="password", key="user_groq_key",
                          help="Free key at console.groq.com. Enables AI extraction (photos), better explanations and Urdu translation.")
    st.sidebar.text_input("Gemini API key (optional, free)", type="password", key="user_key",
                          help="Free key at aistudio.google.com. Also reads scanned PDFs directly. Used only if no Groq key is set.")
    if provider():
        st.sidebar.markdown(f'<span class="pill on">AI mode on: {provider().capitalize()}</span>', unsafe_allow_html=True)
        st.sidebar.caption("Your uploaded files are sent to the selected AI provider for processing.")
    else:
        st.sidebar.markdown('<span class="pill off">Offline mode</span>', unsafe_allow_html=True)
        st.sidebar.caption("OCR / PDF text and rule-based agents. Add a free key for photo reading and Urdu.")
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)


def report_tab():
    st.markdown("Upload a **CBC / blood test / lab report / radiology report** (PDF, photo or .txt).")
    f = st.file_uploader("Medical report", type=["pdf", "jpg", "jpeg", "png", "webp", "txt"], key="rep_file")
    c1, c2 = st.columns([1, 3])
    sample = c2.button("Try with a sample CBC report")
    go = c1.button("🔬 Analyze", type="primary", disabled=f is None)
    if sample:
        txt = SAMPLE_REPORT.encode()
        run(txt, "text/plain", "sample.txt")
    elif go and f is not None:
        run(f.getvalue(), f.type or "", f.name)
    if st.session_state.get("report_res"):
        render_report(st.session_state["report_res"])


def run(data, mime, name):
    sx = {"Male": "male", "Female": "female"}.get(st.session_state.get("sex_choice"), None)
    with st.status("Agents are working...", expanded=True) as status:
        st.session_state["report_res"] = run_report(data, mime, name, sx, st.session_state.get("lang", "English"), status)
        status.update(label="Analysis complete", state="complete", expanded=False)


SAMPLE_REPORT = """Patient Name: Sample Patient    Sex: Female   Age: 24
COMPLETE BLOOD COUNT (CBC)
Test                  Result    Unit        Reference Range
Hemoglobin (Hb)       9.8  L    g/dL        12.0 - 15.5
RBC Count             4.1       million/uL  4.1 - 5.1
Hematocrit (PCV)      31.2 L    %           36 - 46
MCV                   72.0 L    fL          80 - 100
MCH                   23.5 L    pg          27 - 33
MCHC                  31.0 L    g/dL        32 - 36
RDW                   16.2 H    %           11.5 - 14.5
WBC Count             7.2       x10^3/uL    4.0 - 11.0
Platelet Count        410       x10^3/uL    150 - 450
Neutrophils           58        %           40 - 75
Lymphocytes           34        %           20 - 45
Serum Ferritin        6         ng/mL       11 - 307
Vitamin D (25-OH)     14.2 L    ng/mL       30 - 100
Fasting Blood Sugar   92        mg/dL       70 - 99
"""


def about_tab():
    st.markdown("""
### How MediLens AI works
**Upload → Extract → Analyze → Retrieve → Verify → Explain → Guide**

| Agent | Job | How |
|---|---|---|
| 🔍 Extraction | Reads values/findings from PDF or photo | Gemini vision (AI mode) or OCR/PDF-text + regex (offline) |
| 🧠 Analysis | Normal / high / low + severity | Deterministic comparison with report range or standard adult ranges |
| 📚 Medical RAG | Retrieves trusted explanations | TF-IDF retrieval over a curated medical knowledge base |
| ✅ Verification | Cross-checks the data | Value-in-source check, plausibility, unit and flag consistency, safety guardrails |
| 🗣️ Explanation | Simple English / Roman Urdu / Urdu | LLM grounded only on retrieved context (or rule-based templates) |
| 🧭 Guidance | Urgency + next steps | Rule-based; always refers to a doctor |
| 💊 Medicine | Identify pack, match to DRAP-based medicine sheet | Vision/OCR + fuzzy matching, asks to confirm when unsure |

**Safety design:** the numeric analysis is done by code (not by the LLM); the LLM only explains.
The app never recommends a medicine or dose, and blocks outputs that look like dosing/prescribing advice.
""")
    st.warning(DISCLAIMER)


def main():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    sidebar()
    st.markdown(HERO, unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Report Analyzer", "Medicine Lens", "How it works"])
    with t1:
        report_tab()
    with t2:
        medicine_tab()
    with t3:
        about_tab()
    st.markdown('<div class="disc">MediLens AI is an educational tool, not a doctor. It does not diagnose, prescribe or change treatment. Always confirm with a qualified doctor or pharmacist.</div>', unsafe_allow_html=True)


main()
