# Used to transform the manually extracted dataset into a cleaned and normalized format
import json
from collections import OrderedDict

with open('data/manual_extraction_raw.json') as f:
    data = json.load(f)

process_fields = {
    key.replace(" ", "_"): [field.replace(" ", "_") for field in fields]
    for key, fields in {
        "calcination": ['process', "did step", "start_form", "end_form", "gas", "gas_flow", "heat_rate", "temperature", "time"],
        "ion_exchange": ['process', "did step", "initial_form", "final_form", "solutes", "temperature", "time", "solvent", "repeats"],
        "commercial": ['process', "did step", "name", "supplier", "form", "cbv", "Si/Al", "SiO2/Al2O3", "M", "Si/M"],
        "solvent_etching": ['process', "did step", "mass_parent", "solutes", "solvent", "temperature", "time", "microwave", "ultrasound", "repeats"],
        "chemical_liquid_deposition": ['process', "did step", "mass_parent", "solutes", "solvent", "temperature", "time", "microwave", "ultrasound", "repeats", "repeated_with_calcination"],
        "steam_treatment": ['process', "did step", "mass parent", "gas", "temperature", "time", "pressure", "WHSV", "additives"],
        "hydrothermal_crystallization": ['process', "did step", "components", "gel_composition", "seed", "ratios", "pH", "pH_adjusted_with", "temperature", "time", "tumbling"],
        "recrystallisation": ["process", "did step", "composite", "dissolvent", "glycerol", "hydrothermal 1 temp", "hydrothermal 1 time", "pH adjusted with", "pH adjustment", "hydrothermal 2 temp", "hydrothermal 2 time", "parent_mass", "surfactant"],
        "impregnation": ['process', "did step", "metal", "loading"]
    }.items()
}

def normalize_keys(step):
    return {k.replace(" ", "_"): v for k, v in step.items()}

def clean_step(step):
    step = normalize_keys(step)
    proc = step.get("process", "").strip().lower().replace(" ", "_")

    # Map to standardized process types
    if proc in ["alkaline_treatment", "acid_treatment"]:
        proc = "solvent_etching"
        step["process"] = proc

    if proc.startswith("-->") or "exchange" in proc:
        proc = "ion_exchange"
        step["process"] = proc

    if proc in process_fields:
        return {k.replace(" ", "_"): step.get(k.replace(" ", "_"), "") for k in process_fields[proc]}
    else:
        return {"process": "something_else"}

new_data = []
for paper in data:
    new_paper = {"doi": paper["doi"], "text": paper['text'], "samples": []}
    for sample in paper['samples']:
        new_sample = {
            "name": sample['name'],
            "morphological_description": sample.get('morphological_description', ""),
            "zeolite_source": {"source": "unknown"}
        }
        history = sample.get("history", [])
        if not history:
            new_sample["post-synthesis"] = []
        else:
            first_step = history[0]
            proc_type = first_step.get("process", "").strip().lower()
            if proc_type in ["commercial", "hydrothermal crystallization"]:
                zeolite_step = normalize_keys(first_step)
                source_value = zeolite_step.pop("process")
                
                zeolite_ordered = OrderedDict()
                zeolite_ordered["source"] = source_value
                for k, v in zeolite_step.items():
                    zeolite_ordered[k] = v

                new_sample["zeolite_source"] = zeolite_ordered
                remaining_history = history[1:]  # ✅ Skip the first step
            else:
                remaining_history = history
            new_sample["post-synthesis"] = [clean_step(normalize_keys(step)) for step in remaining_history]

        new_paper["samples"].append(new_sample)
    new_data.append(new_paper)

with open('final_true_cleaned.json', 'w') as f:
    json.dump(new_data, f, indent=4)