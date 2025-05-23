from typing import Optional, List
from pydantic import BaseModel

class Sample(BaseModel):
    sample: str
    Si_Al: Optional[str] = None
    V_total: Optional[str] = None
    V_micro: Optional[str] = None
    V_meso: Optional[str] = None
    Bronsted_Acid_Sites: Optional[str] = None
    Lewis_Acid_Sites: Optional[str] = None
    S_BET: Optional[str] = None
    S_ext: Optional[str] = None
    D_micro: Optional[str] = None
    D_meso: Optional[str] = None
    crystallinity: Optional[str] = None
    metal_content: Optional[str] = None

class Output(BaseModel):
    samples: List[Sample]

def create_prompt(table_str: str) -> str:
    return f"""
Extract structured sample characterization data from the following table. Return the result as a JSON object with the structure:

{{
  "samples": [
    {{
      "sample": str,
      "Si_Al": Optional[str],
      "V_total": Optional[str],
      "V_micro": Optional[str],
      "V_meso": Optional[str],
      "Bronsted_Acid_Sites": Optional[str],
      "Lewis_Acid_Sites": Optional[str],
      "S_BET": Optional[str],
      "S_ext": Optional[str],
      "D_micro": Optional[str],
      "D_meso": Optional[str],
      "crystallinity": Optional[str],
      "metal_content": Optional[str]
    }},
    ...
  ]
}}

Guidelines:
- Only include actual sample rows (ignore headings or column labels).
- Use `null` or omit fields that are not available for a given sample.
- Preserve any units and the measurement method (e.g., "cm³/g - t-plot", "mol - EDS", "mmol/g - IR").
- If multiple measurement methods are listed for the same property (e.g. Brønsted acidity from both Py-IR and TMPy-IR), include both, joined with a pipe `|`.
- Keep the `sample` name exactly as shown in the text.
- Do not include extra commentary.
- Only report values in the table. Do not hallucinate.

### Table:
{table_str}

"""