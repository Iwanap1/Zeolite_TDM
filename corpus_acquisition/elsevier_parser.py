import json
import re
import xml.etree.ElementTree as ET
import os

class Table:
    def __init__(self, xml_table, file_path=None):
        self.namespaces = {
            'dc': 'http://purl.org/dc/elements/1.1/',
            'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
            'ce': 'http://www.elsevier.com/xml/common/dtd',
            'sa': 'http://www.elsevier.com/xml/common/struct-aff/dtd',
            'xocs': 'http://www.elsevier.com/xml/xocs/dtd',
            'dcterms': 'http://purl.org/dc/terms/',
            'cals': 'http://www.elsevier.com/xml/common/cals/dtd',
            'default': 'http://www.elsevier.com/xml/svapi/article/dtd',
            'ns': 'http://www.elsevier.com/xml/common/dtd',
        }
        self.file_path = file_path
        self.xml_table = xml_table
        self.glyph_map = self.glyphs()
        self.footnotes = self._get_footnotes()
        self.head = self.organise_headers()
        self.single_layer_head = self.flatten_headers(self.head) if self.head else None
        self.body = self.get_body_grid()
        self.table = self.head + self.body if self.head else self.body
        try:
            self.label = xml_table.find('ce:label', self.namespaces).text
        except: 
            print(f"Label not found in table in {file_path}")
            self.label = None
        self.flat_head_table = [self.single_layer_head] + self.body if self.single_layer_head else self.body
        self.caption = self.get_caption()
        self.formatted_table = self.format_table()

    def _get_footnotes(self):
        """Extract footnote labels and parse full inner content with nested tags."""
        raw_footnotes = {}

        for fn in self.xml_table.findall('.//ce:table-footnote', self.namespaces):
            label_elem = fn.find('ce:label', self.namespaces)
            note_para_elem = fn.find('ce:note-para', self.namespaces)
            if label_elem is not None and note_para_elem is not None:
                label = label_elem.text.strip()
                raw_footnotes[label] = note_para_elem  # store raw XML

        # Now parse text safely
        resolved = {}
        for label, elem in raw_footnotes.items():
            resolved[label] = self._get_text_from_entry(elem)

        return resolved

    def _get_text_from_entry(self, entry):
        parts = []
        footnote_labels = []

        def process(elem):
            tag = elem.tag.split('}')[-1]

            # Add this element's text
            if elem.text:
                parts.append(elem.text)

            # Handle footnote cross-refs: just collect label
            if tag == 'cross-ref' and 'refid' in elem.attrib:
                sup = elem.find('ns:sup', self.namespaces)
                if sup is not None and sup.text:
                    label = sup.text.strip()
                    footnote_labels.append(label)

                # ✅ Append tail text (e.g., ' (cm') that comes after <cross-ref>
                if elem.tail:
                    parts.append(elem.tail)
                return

            # Other tags
            elif tag in ('sup', 'inf', 'bold', 'italic', 'glyph'):
                for child in elem:
                    process(child)
                # Tail after tag
                if elem.tail:
                    parts.append(elem.tail)
                return

            # Default case: recurse
            for child in elem:
                process(child)

            if elem.tail:
                parts.append(elem.tail)

        process(entry)

        # Build base text
        text = self._clean_unicode("".join(parts))

        # Append notes at the end
        notes = []
        for label in footnote_labels:
            if label in self.footnotes:
                notes.append(f"[NOTE] {self.footnotes[label]}")
            else:
                notes.append(f"[NOTE] [{label}]")

        if notes:
            text += " " + " ".join(notes)

        return text.strip()

    def get_caption(self):
        caption = self.xml_table.find('.//ce:caption/ce:simple-para', self.namespaces)
        if caption is not None:
            return self._clean_unicode(self._get_text_from_entry(caption).strip())
        return ""

    def _clean_unicode(self, text):
        replacements = {
            '’': "'",
            '\u2009': ' ',  # thin space
            ' ': ' ',       # non-breaking thin space (alt code)
            ' ': ' ',       # non-breaking space
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return re.sub(r'\s{2,}', ' ', text).strip()

    def organise_headers(self):
        h = self._get_headers_position_and_text()
        if h is None:
            return None
        n_cols = max(entry['end_col'] for entry in h)
        n_rows = max(entry['row_idx'] + entry.get('morerows', 0) for entry in h) + 1

        tabl = [[''] * n_cols for _ in range(n_rows)]

        for entry in h:
            start_row = entry['row_idx']
            end_row = start_row + entry.get('morerows', 0)
            for row in range(start_row, end_row + 1):
                for col in range(entry['start_col'], entry['end_col'] + 1):
                    tabl[row][col - 1] = entry['text']
        return tabl

    def flatten_headers(self, headers):
        if not headers or len(headers) == 1:
            return headers[0] if headers else []

        num_rows = len(headers)
        num_cols = len(headers[0])
        flattened = []

        for col in range(num_cols):
            parts = []
            for row in range(num_rows):
                entry = headers[row][col].strip()
                if entry and (not parts or entry != parts[-1]):
                    parts.append(entry)
            flattened.append(" | ".join(parts).strip())
        return flattened

    def _get_headers_position_and_text(self):
        thead = self.xml_table.find('.//cals:thead', self.namespaces)

        if thead is None:
            return None

        rows = thead.findall('.//cals:row', self.namespaces)
        entries_info = []
        col_occupancy = {}  # Tracks occupied columns for row spanning

        for row_idx, row in enumerate(rows):
            entries = row.findall('.//ns:entry', self.namespaces)
            
            current_col = 1  # Columns are 1-indexed in the XML
            
            for entry in entries:
                # Move current_col forward if it is already occupied (row span from previous rows)
                while current_col in col_occupancy and col_occupancy[current_col] >= row_idx:
                    current_col += 1  

                text = self._get_text_from_entry(entry).replace('\n', "").strip()  # Get text content
                
                # Get column span attributes
                namest = entry.get('namest')
                nameend = entry.get('nameend')
                morerows = int(entry.get('morerows', 0))  # Defaults to 0 if not present
                
                # Determine column span
                if namest:
                    start_col_idx = int(namest.replace('col', ''))
                else:
                    start_col_idx = current_col  # Use current available column
                
                if nameend:
                    end_col_idx = int(nameend.replace('col', ''))
                else:
                    end_col_idx = start_col_idx  # Single column entry
                
                # Store entry details
                entries_info.append({
                    'text': text,
                    'start_col': start_col_idx,
                    'end_col': end_col_idx,
                    'morerows': morerows,
                    'row_idx': row_idx
                })
                
                # Mark columns as occupied for row spans
                if morerows > 0:
                    for col in range(start_col_idx, end_col_idx + 1):
                        col_occupancy[col] = row_idx + morerows  
                
                # Move to the next column after the span
                current_col = end_col_idx + 1
        
        return entries_info
        
    def get_body_grid(self):
        tbody = self.xml_table.find('.//cals:tbody', self.namespaces)
        if tbody is None:
            return []

        rows = tbody.findall('.//cals:row', self.namespaces)
        entries_info = []
        col_occupancy = {}

        for row_idx, row in enumerate(rows):
            entries = row.findall('.//ns:entry', self.namespaces)
            current_col = 1

            for entry in entries:
                # Skip occupied columns due to rowspan
                while current_col in col_occupancy and col_occupancy[current_col] >= row_idx:
                    current_col += 1

                text = self._clean_unicode(self._get_text_from_entry(entry).replace('\n', '').strip())

                namest = entry.get('namest')
                nameend = entry.get('nameend')
                morerows = int(entry.get('morerows', 0))

                if namest:
                    start_col_idx = int(namest.replace('col', ''))
                else:
                    start_col_idx = current_col

                if nameend:
                    end_col_idx = int(nameend.replace('col', ''))
                else:
                    end_col_idx = start_col_idx

                entries_info.append({
                    'text': text,
                    'start_col': start_col_idx,
                    'end_col': end_col_idx,
                    'morerows': morerows,
                    'row_idx': row_idx
                })

                if morerows > 0:
                    for col in range(start_col_idx, end_col_idx + 1):
                        col_occupancy[col] = row_idx + morerows

                current_col = end_col_idx + 1

        # Determine full grid size
        if not entries_info:
            return []

        n_cols = max(entry['end_col'] for entry in entries_info)
        n_rows = max(entry['row_idx'] + entry.get('morerows', 0) for entry in entries_info) + 1

        grid = [[''] * n_cols for _ in range(n_rows)]

        for entry in entries_info:
            start_row = entry['row_idx']
            end_row = start_row + entry.get('morerows', 0)
            for row in range(start_row, end_row + 1):
                for col in range(entry['start_col'], entry['end_col'] + 1):
                    grid[row][col - 1] = entry['text']

        return grid

    def format_table(self):
        formatted_rows = []
        for row in self.table:
            formatted_cells = [' <C> ' + (cell if cell.strip() else '[EMPTY]') for cell in row]
            formatted_row = ' <R> ' + ''.join(formatted_cells)
            formatted_rows.append(formatted_row)
        return ''.join(formatted_rows) + ' <Cap> ' + self.caption + ' </Cap> '

    def glyphs(self):
        return {
        'sbnd': '-',
        'dbnd': '=',
        'tbnd': '≡',
        'bond': '-',
        'plusmn': '±',
        'minus': '-',
        'times': '×',
        'divide': '÷',
        'lt': '<',
        'gt': '>',
        'le': '≤',
        'ge': '≥',
        'approx': '≈',
        'equiv': '≡',
        'neq': '≠',
        'pm': '±',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'epsilon': 'ε',
        'theta': 'θ',
        'mu': 'μ',
        'pi': 'π',
        'sigma': 'σ',
        'omega': 'ω',
        'rightarrow': '→',
        'leftarrow': '←',
        'leftrightarrow': '↔',
        'uparrow': '↑',
        'downarrow': '↓',
        'deg': '°',
        'degree': '°',
        'micro': 'μ',
        'ohm': 'Ω',
        'ang': 'Å',
        'bullet': '•',
        'middot': '·',
        'mdash': '—',
        'ndash': '-',
        'infty': '∞',
        'subset': '⊂',
        'supset': '⊃',
        'int': '∫',
        'sum': '∑',
    }

    def to_dict(self):
        return {
            "label": self.label,
            "caption": self.caption,
            "as_string": self.formatted_table,
            "single_head_table": self.flat_head_table,
            "full_table": self.table
        } 

class ElsevierParser:
    def __init__(self):
        self.namespaces = {
            'dc': 'http://purl.org/dc/elements/1.1/',
            'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
            'ce': 'http://www.elsevier.com/xml/common/dtd',
            'sa': 'http://www.elsevier.com/xml/common/struct-aff/dtd',
            'xocs': 'http://www.elsevier.com/xml/xocs/dtd',
            'dcterms': 'http://purl.org/dc/terms/',
            'cals': 'http://www.elsevier.com/xml/common/cals/dtd',
            'default': 'http://www.elsevier.com/xml/svapi/article/dtd',
            'ns': 'http://www.elsevier.com/xml/common/dtd',
        }
        self.parsed = []

    def parse_directory(self, directory_path):
        file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.xml')]
        self.parsed = [self.parse_document(fp) for fp in file_paths if os.path.isfile(fp)]
        return self.parsed

    def parse_document(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            root = ET.fromstring(content)
            return self._parse_root(root, file_path)
        except ET.ParseError as e:
            print(f"XML Parse error in {file_path}: {e}")
        except UnicodeDecodeError:
            print(f"Encoding error in {file_path}")
        return None

    def parse_from_string(self, xml_string: str, file_path: str = "<api_response>"):
        try:
            root = ET.fromstring(xml_string)
            return self._parse_root(root, file_path)
        except ET.ParseError as e:
            print(f"Error parsing XML string: {e}")
            return None

    def _parse_root(self, root, file_path):
        core_data = root.find('default:coredata', self.namespaces)
        if core_data is None:
            print(f"No coredata found in {file_path}")
            return None
        docsubtype = root.findtext('.//xocs:document-subtype', default="", namespaces=self.namespaces).lower()
        authors = [a.text for a in core_data.findall('dc:creator', self.namespaces)]
        doi = core_data.findtext('prism:doi', default=None, namespaces=self.namespaces)
        title = core_data.findtext('dc:title', default=None, namespaces=self.namespaces)
        year = core_data.findtext('prism:coverDate', default="", namespaces=self.namespaces).split('-')[0]
        keywords = [k.text for k in core_data.findall('dcterms:subject', self.namespaces)]
        publication = core_data.findtext('prism:publicationName', default=None, namespaces=self.namespaces)
        abstract_elem = core_data.find('dc:description', self.namespaces)
        abstract_text = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else None

        return {
            'file_path': file_path,
            'type': docsubtype,
            'rejected_because': self.classify_paper(abstract_text, title, keywords, docsubtype) if abstract_text else 'no abstract',
            'authors': authors,
            'doi': doi,
            'title': title,
            'year': year,
            'publication': publication,
            'keywords': keywords,
            'abstract': abstract_text,
            'sections': self._get_sections(root),
            'tables': self.extract_tables(root, file_path)
        }

    def extract_tables(self, root, file_path):
        xml_tables = root.findall('.//ce:table', self.namespaces) + root.findall('.//cals:table', self.namespaces)
        return [Table(t, file_path).to_dict() for t in xml_tables]

    def _get_sections(self, root):
        def clean_text(text):
            return " ".join(text.replace('\n', ' ').split())

        def extract_paragraphs(section_element):
            return [
                clean_text(''.join(p.itertext()).strip())
                for p in section_element.findall('ce:para', self.namespaces)
                if p.text and p.text.strip()
            ]

        def recurse_sections(section_element):
            result = []
            title_elem = section_element.find('ce:section-title', self.namespaces)
            title = title_elem.text.strip() if title_elem is not None else "No Title"
            content = extract_paragraphs(section_element)
            if content:
                result.append({"section_name": title, "content": content})
            for sub in section_element.findall('ce:section', self.namespaces):
                result.extend(recurse_sections(sub))
            return result

        body = root.find('.//ce:sections', self.namespaces)
        return [] if body is None else [sec for section in body.findall('ce:section', self.namespaces) for sec in recurse_sections(section)]
    
    def classify_paper(self, abstract: str, title: str, keywords: list[str], docsubtype: str = "") -> str:
        if not abstract:
            return "no abstract"

        if docsubtype == "rev":
            return "review article"

        combined = " ".join([
            abstract or "",
            title or "",
            " ".join(keywords or [])
        ]).lower()

        if not (
            ("zeolite" in combined or "zsm" in combined) and
            ("mesop" in combined or "hierarchical" in combined or "modif" in combined)
        ):
            return "missing keywords"

        return "accepted"

    def save_to_json(self, output_path="parsed_elsevier.json"):
        if not self.parsed:
            print("No parsed data to save.")
            return
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([d for d in self.parsed if d], f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.parsed)} documents to {output_path}")

