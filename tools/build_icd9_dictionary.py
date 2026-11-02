"""
ReadmitAI - ICD-9-CM Description Dictionary Builder
Builds data/icd9_descriptions.json, which turns diagnosis codes into readable descriptions.

Sources (both US-government public domain):
  * CMS ICD-9-CM v32 (FY2015) long diagnosis descriptions: CMS32_DESC_LONG_DX.txt
    https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes
  * AHRQ HCUP Clinical Classifications Software (CCS) single-level diagnosis grouper, Appendix A
    https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
  * ICD-9-CM tabular-list hierarchy (chapter / section / category titles), as packaged in the
    icdcodex project (icdcodex/data/icd-9-hierarchy.json)

The CMS file only lists billable codes (e.g. 4280), but the UCI dataset mostly records 3-digit
categories (e.g. 428). Category titles are resolved in this order:
  1. CATEGORY_TITLES below - tabular-list titles written out for every category in the UCI data whose
     sub-codes fall into more than one CCS group (kept because they include the official punctuation);
  2. the category title from the tabular-list hierarchy;
  3. the CCS group shared by all of the category's sub-codes;
  4. the longest common wording of the sub-code descriptions;
  5. the most common CCS group among the sub-codes.

Usage:
    py -3.12 tools/build_icd9_dictionary.py <CMS32_DESC_LONG_DX.txt> <CCS-SingleDiagnosisGrouper.txt> <icd-9-hierarchy.json>
"""

import collections
import json
import os
import re
import sys

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "icd9_descriptions.json"))

CATEGORY_TITLES = {
    "003": "Other salmonella infections",
    "031": "Diseases due to other mycobacteria",
    "034": "Streptococcal sore throat and scarlet fever",
    "036": "Meningococcal infection",
    "039": "Actinomycotic infections",
    "040": "Other bacterial diseases",
    "046": "Slow virus infections and prion diseases of central nervous system",
    "049": "Other non-arthropod-borne viral diseases of central nervous system",
    "052": "Chickenpox",
    "053": "Herpes zoster",
    "054": "Herpes simplex",
    "058": "Other human herpesvirus",
    "066": "Other arthropod-borne viral diseases",
    "079": "Viral and chlamydial infection in conditions classified elsewhere and of unspecified site",
    "112": "Candidiasis",
    "114": "Coccidioidomycosis",
    "115": "Histoplasmosis",
    "136": "Other and unspecified infectious and parasitic diseases",
    "139": "Late effects of other infectious and parasitic diseases",
    "162": "Malignant neoplasm of trachea, bronchus, and lung",
    "183": "Malignant neoplasm of ovary and other uterine adnexa",
    "189": "Malignant neoplasm of kidney and other and unspecified urinary organs",
    "195": "Malignant neoplasm of other and ill-defined sites",
    "202": "Other malignant neoplasms of lymphoid and histiocytic tissue",
    "203": "Multiple myeloma and immunoproliferative neoplasms",
    "230": "Carcinoma in situ of digestive organs",
    "233": "Carcinoma in situ of breast and genitourinary system",
    "250": "Diabetes mellitus",
    "256": "Ovarian dysfunction",
    "258": "Polyglandular dysfunction and related disorders",
    "272": "Disorders of lipoid metabolism",
    "276": "Disorders of fluid, electrolyte, and acid-base balance",
    "277": "Other and unspecified disorders of metabolism",
    "279": "Disorders involving the immune mechanism",
    "282": "Hereditary hemolytic anemias",
    "285": "Other and unspecified anemias",
    "287": "Purpura and other hemorrhagic conditions",
    "289": "Other diseases of blood and blood-forming organs",
    "293": "Transient mental disorders due to conditions classified elsewhere",
    "300": "Anxiety, dissociative and somatoform disorders",
    "302": "Sexual and gender identity disorders",
    "305": "Nondependent abuse of drugs",
    "307": "Special symptoms or syndromes, not elsewhere classified",
    "309": "Adjustment reaction",
    "310": "Specific nonpsychotic mental disorders due to brain damage",
    "312": "Disturbance of conduct, not elsewhere classified",
    "313": "Disturbance of emotions specific to childhood and adolescence",
    "327": "Organic sleep disorders",
    "331": "Other cerebral degenerations",
    "332": "Parkinson's disease",
    "333": "Other extrapyramidal disease and abnormal movement disorders",
    "337": "Disorders of the autonomic nervous system",
    "341": "Other demyelinating diseases of central nervous system",
    "344": "Other paralytic syndromes",
    "346": "Migraine",
    "348": "Other conditions of brain",
    "349": "Other and unspecified disorders of the nervous system",
    "357": "Inflammatory and toxic neuropathy",
    "360": "Disorders of the globe",
    "363": "Chorioretinal inflammations, scars, and other disorders of choroid",
    "364": "Disorders of iris and ciliary body",
    "370": "Keratitis",
    "372": "Disorders of conjunctiva",
    "373": "Inflammation of eyelids",
    "375": "Disorders of lacrimal system",
    "376": "Disorders of the orbit",
    "377": "Disorders of optic nerve and visual pathways",
    "379": "Other disorders of eye",
    "384": "Other disorders of tympanic membrane",
    "385": "Other disorders of middle ear and mastoid",
    "398": "Other rheumatic heart disease",
    "401": "Essential hypertension",
    "414": "Other forms of chronic ischemic heart disease",
    "415": "Acute pulmonary heart disease",
    "425": "Cardiomyopathy",
    "427": "Cardiac dysrhythmias",
    "429": "Ill-defined descriptions and complications of heart disease",
    "433": "Occlusion and stenosis of precerebral arteries",
    "437": "Other and ill-defined cerebrovascular disease",
    "440": "Atherosclerosis",
    "443": "Other peripheral vascular disease",
    "447": "Other disorders of arteries and arterioles",
    "456": "Varicose veins of other sites",
    "458": "Hypotension",
    "459": "Other disorders of circulatory system",
    "477": "Allergic rhinitis",
    "507": "Pneumonitis due to solids and liquids",
    "511": "Pleurisy",
    "512": "Pneumothorax and air leak",
    "513": "Abscess of lung and mediastinum",
    "517": "Lung involvement in conditions classified elsewhere",
    "518": "Other diseases of lung",
    "519": "Other diseases of respiratory system",
    "528": "Diseases of the oral soft tissues, excluding lesions specific for gingiva and tongue",
    "530": "Diseases of esophagus",
    "531": "Gastric ulcer",
    "532": "Duodenal ulcer",
    "533": "Peptic ulcer, site unspecified",
    "534": "Gastrojejunal ulcer",
    "535": "Gastritis and duodenitis",
    "536": "Disorders of function of stomach",
    "558": "Other and unspecified noninfectious gastroenteritis and colitis",
    "564": "Functional digestive disorders, not elsewhere classified",
    "567": "Peritonitis and retroperitoneal infections",
    "569": "Other disorders of intestine",
    "571": "Chronic liver disease and cirrhosis",
    "573": "Other disorders of liver",
    "579": "Intestinal malabsorption",
    "596": "Other disorders of bladder",
    "598": "Urethral stricture",
    "599": "Other disorders of urethra and urinary tract",
    "603": "Hydrocele",
    "607": "Disorders of penis",
    "608": "Other disorders of male genital organs",
    "620": "Noninflammatory disorders of ovary, fallopian tube, and broad ligament",
    "625": "Pain and other symptoms associated with female genital organs",
    "626": "Disorders of menstruation and other abnormal bleeding from female genital tract",
    "648": "Other current conditions in the mother complicating pregnancy, childbirth, or the puerperium",
    "649": "Other conditions or status of the mother complicating pregnancy, childbirth, or the puerperium",
    "654": "Abnormality of organs and soft tissues of pelvis complicating pregnancy",
    "655": "Known or suspected fetal abnormality affecting management of mother",
    "656": "Other known or suspected fetal and placental problems affecting management of mother",
    "660": "Obstructed labor",
    "669": "Other complications of labor and delivery, not elsewhere classified",
    "692": "Contact dermatitis and other eczema",
    "710": "Diffuse diseases of connective tissue",
    "716": "Other and unspecified arthropathies",
    "718": "Other derangement of joint",
    "720": "Ankylosing spondylitis and other inflammatory spondylopathies",
    "727": "Other disorders of synovium, tendon, and bursa",
    "733": "Other disorders of bone and cartilage",
    "736": "Other acquired deformities of limbs",
    "737": "Curvature of spine",
    "780": "General symptoms",
    "781": "Symptoms involving nervous and musculoskeletal systems",
    "782": "Symptoms involving skin and other integumentary tissue",
    "783": "Symptoms concerning nutrition, metabolism, and development",
    "784": "Symptoms involving head and neck",
    "785": "Symptoms involving cardiovascular system",
    "786": "Symptoms involving respiratory system and other chest symptoms",
    "787": "Symptoms involving digestive system",
    "788": "Symptoms involving urinary system",
    "789": "Other symptoms involving abdomen and pelvis",
    "790": "Nonspecific findings on examination of blood",
    "791": "Nonspecific findings on examination of urine",
    "792": "Nonspecific abnormal findings in other body substances",
    "793": "Nonspecific abnormal findings on radiological and other examination of body structure",
    "794": "Nonspecific abnormal results of function studies",
    "795": "Other and nonspecific abnormal cytological, histological, immunological and DNA test findings",
    "796": "Other nonspecific abnormal findings",
    "799": "Other ill-defined and unknown causes of morbidity and mortality",
    "800": "Fracture of vault of skull",
    "801": "Fracture of base of skull",
    "803": "Other and unqualified skull fractures",
    "905": "Late effects of musculoskeletal and connective tissue injuries",
    "906": "Late effects of injuries to skin and subcutaneous tissues",
    "907": "Late effects of injuries to the nervous system",
    "908": "Late effects of other and unspecified injuries",
    "909": "Late effects of other and unspecified external causes",
    "965": "Poisoning by analgesics, antipyretics, and antirheumatics",
    "980": "Toxic effect of alcohol",
    "995": "Certain adverse effects not elsewhere classified",
    "999": "Complications of medical care, not elsewhere classified",
    "E821": "Nontraffic accident involving other off-road motor vehicle",
    "E822": "Other motor vehicle nontraffic accident involving collision with moving object",
    "E825": "Other motor vehicle nontraffic accident of other and unspecified nature",
    "E826": "Pedal cycle accident",
    "E828": "Accident involving animal being ridden",
    "E829": "Other road vehicle accidents",
    "E922": "Accident caused by firearm and air gun missile",
    "E927": "Overexertion and strenuous movements or postures",
    "E928": "Other and unspecified environmental and accidental causes",
    "E929": "Late effects of accidental injury",
    "E965": "Assault by firearms and explosives",
    "E968": "Assault by other and unspecified means",
    "V07": "Need for isolation and other prophylactic or treatment measures",
    "V10": "Personal history of malignant neoplasm",
    "V12": "Personal history of certain other diseases",
    "V13": "Personal history of other diseases",
    "V15": "Other personal history presenting hazards to health",
    "V23": "Supervision of high-risk pregnancy",
    "V27": "Outcome of delivery",
    "V42": "Organ or tissue replaced by transplant",
    "V43": "Organ or tissue replaced by other means",
    "V44": "Artificial opening status",
    "V45": "Other postprocedural states",
    "V46": "Other dependence on machines and devices",
    "V49": "Other conditions influencing health status",
    "V50": "Elective surgery for purposes other than remedying health states",
    "V53": "Fitting and adjustment of other device",
    "V54": "Other orthopedic aftercare",
    "V55": "Attention to artificial openings",
    "V58": "Encounter for other and unspecified procedures and aftercare",
    "V62": "Other psychosocial circumstances",
    "V65": "Other persons seeking consultation",
    "V66": "Convalescence and palliative care",
    "V67": "Follow-up examination",
    "V70": "General medical examination",
    "V71": "Observation and evaluation for suspected conditions not found",
    "V72": "Special investigations and examinations",
    "V85": "Body mass index",
}


def load_cms(path):
    codes = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            code, desc = line[:6].strip(), line[6:].strip()
            if code:
                codes[code] = desc
    return codes


def clean_ccs_label(label):
    return re.sub(r"^E Codes:\s*", "", label)


def load_ccs(path):
    groups, current = {}, None
    with open(path, encoding="latin-1") as f:
        for line in f:
            header = re.match(r"^(\d+)\s+(\S.*)$", line.rstrip())
            if header:
                current = header.group(2).strip()
            elif current and line.startswith(" "):
                for code in line.split():
                    groups[code] = clean_ccs_label(current)
    return groups


DANGLING_WORDS = {"type", "of", "with", "and", "or", "the", "in", "to", "by", "without", "due", "a", "an", "not"}


ACRONYMS = {"hiv", "dna", "rna", "ckd", "bmi", "copd", "aids", "nos", "nec", "cns", "gi", "hpv", "mrsa", "esrd"}


def restore_acronyms(title):
    """The hierarchy file lowercases acronyms ('Chronic kidney disease (ckd)')."""
    return re.sub(r"\b[a-z]+\b", lambda m: m.group(0).upper() if m.group(0) in ACRONYMS else m.group(0), title)


def load_hierarchy_titles(path):
    """Maps each category code (e.g. '808') to its tabular-list title (e.g. 'Fracture of pelvis')."""
    with open(path, encoding="utf-8") as f:
        tree = json.load(f)["tree"]
    titles = {}

    def walk(node):
        if "children" not in node:
            return [node["id"]]
        leaves = [leaf for child in node["children"] for leaf in walk(child)]
        prefix = os.path.commonprefix(leaves)
        if node["id"] != "root" and len(prefix) >= category_length(prefix or "0"):
            titles.setdefault(prefix, restore_acronyms(node["id"]))  # post-order, so the deepest node wins
        return leaves

    walk(tree)
    return titles


def common_wording(descriptions):
    prefix = os.path.commonprefix(descriptions)
    if len(prefix) < len(min(descriptions, key=len)):
        prefix = prefix[: max(prefix.rfind(","), prefix.rfind(" "), 0)]
    words = prefix.strip(" ,;:-").split()
    while words and words[-1].lower().strip(",;:") in DANGLING_WORDS:
        words.pop()
    return " ".join(words).strip(" ,;:-")


def category_title(prefix, children, cms, ccs, hierarchy):
    if prefix in CATEGORY_TITLES:
        return CATEGORY_TITLES[prefix]
    if prefix in hierarchy:
        return hierarchy[prefix]
    groups = [ccs[c] for c in children if c in ccs]
    if len(set(groups)) == 1:
        return groups[0]
    wording = common_wording([cms[c] for c in children])
    if len(wording) >= 12:
        return wording
    if groups:
        return collections.Counter(groups).most_common(1)[0][0]
    return cms[children[0]]


def category_length(code):
    return 4 if code.startswith("E") else 3


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    cms = load_cms(sys.argv[1])
    ccs = load_ccs(sys.argv[2])
    hierarchy = load_hierarchy_titles(sys.argv[3])

    descriptions = dict(cms)
    prefixes = collections.defaultdict(list)
    for code in cms:
        base = category_length(code)
        for length in range(base, len(code)):
            prefixes[code[:length]].append(code)

    for prefix, children in prefixes.items():
        if prefix not in descriptions:
            if len(prefix) == category_length(prefix):
                descriptions[prefix] = category_title(prefix, children, cms, ccs, hierarchy)
            elif prefix in hierarchy:
                descriptions[prefix] = hierarchy[prefix]
            else:
                wording = common_wording([cms[c] for c in children])
                category = prefix[: category_length(prefix)]
                descriptions[prefix] = wording if len(wording) >= 12 else category_title(category, prefixes[category], cms, ccs, hierarchy)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "_source": "CMS ICD-9-CM v32 long descriptions; category titles from the ICD-9-CM tabular list (AHRQ HCUP CCS grouper as fallback)",
                "codes": dict(sorted(descriptions.items())),
            },
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    print(f"Wrote {len(descriptions):,} ICD-9-CM descriptions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
