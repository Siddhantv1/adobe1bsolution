import fitz  # PyMuPDF
import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

class EnhancedDocumentProcessor:
    
    def __init__(self):
        # Enhanced patterns for detecting different content types
        self.heading_patterns = [
            r'^\d+\.?\s+[A-Z]',                    # "1. Chapter Title" or "1 Chapter Title"
            r'^[IVX]+\.\s+[A-Z]',                  # Roman numerals
            r'^[A-Z][A-Z\s&-]{5,50}:?\s*$',      # ALL CAPS headings (5-50 chars)
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*$',  # Title Case headings
            r'^[a-z]\)\s+[A-Z]',                  # "a) Title"
            r'^\s*[-•]\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # Bulleted section titles
        ]
        
        # Ligature normalization mapping
        self.ligature_map = {
            'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'ﬅ': 'st', 'ﬆ': 'st', 'ﬞ': 'ts', 'ﬗ': 'tz', 'ﬆ': 'st',
            '＆': '&', '－': '-', '＋': '+', '＝': '=', '（': '(', '）': ')',
            '［': '[', '］': ']', '｛': '{', '｝': '}', '｜': '|', '＼': '\\',
            '／': '/', '？': '?', '！': '!', '．': '.', '，': ',', '；': ';',
            '：': ':', '＂': '"', '＇': "'", '｀': '`', '～': '~', '＠': '@',
            '＃': '#', '＄': '$', '％': '%', '＾': '^', '＊': '*', '＿': '_'
        }
        
        # Generic heading patterns to deboost
        self.generic_headings = {
            'introduction', 'intro', 'conclusion', 'summary', 'overview', 
            'preface', 'foreword', 'abstract', 'appendix', 'references',
            'bibliography', 'index', 'contents', 'table of contents',
            'acknowledgments', 'acknowledgements', 'about', 'background'
        }

    def normalize_text(self, text: str) -> str:
        """Normalize text handling ligatures and Unicode issues."""
        if not text:
            return ""
        
        # First apply custom ligature mapping
        for ligature, replacement in self.ligature_map.items():
            text = text.replace(ligature, replacement)
        
        # Apply Unicode normalization (NFKC handles most remaining cases)
        text = unicodedata.normalize('NFKC', text)
        
        # Clean up common PDF extraction artifacts
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Remove control chars
        
        return text.strip()

    def extract_sections(self, pdf_path: str, doc_name: str) -> List[Dict]:
        """Extract sections using enhanced techniques with TOC support."""
        sections = []
        try:
            doc = fitz.open(pdf_path)
            
            # Try to extract TOC first
            toc_sections = self._extract_toc_sections(doc, doc_name)
            if toc_sections and len(toc_sections) >= 3:
                print(f"Using TOC-based extraction for {doc_name}: {len(toc_sections)} sections")
                doc.close()
                return toc_sections
            
            # Fallback to enhanced heuristic extraction
            print(f"Using heuristic extraction for {doc_name}")
            sections = self._extract_heuristic_sections(doc, doc_name)
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
        
        return self._post_process_sections(sections)

    def _extract_toc_sections(self, doc, doc_name: str) -> List[Dict]:
        """Extract sections using PDF Table of Contents."""
        try:
            toc = doc.get_toc()
            if not toc:
                return []
            
            sections = []
            
            # Process TOC entries
            for level, title, page_num in toc:
                if level > 3:  # Skip very deep nesting
                    continue
                
                normalized_title = self.normalize_text(title)
                if len(normalized_title) < 3 or len(normalized_title) > 200:
                    continue
                
                # Extract content for this section
                content = self._extract_section_content_by_page(doc, page_num, normalized_title)
                
                if content and len(content.strip()) > 50:
                    sections.append({
                        'document': doc_name,
                        'page_number': page_num,
                        'section_title': normalized_title,
                        'content': content,
                        'toc_level': level,
                        'is_generic': self._is_generic_heading(normalized_title),
                        'procedural_sentences': self._extract_procedural_sentences(content)
                    })
            
            return sections
            
        except Exception as e:
            print(f"TOC extraction failed: {e}")
            return []

    def _extract_section_content_by_page(self, doc, start_page: int, section_title: str) -> str:
        """Extract content for a section starting from a specific page."""
        content_parts = []
        
        # Convert to 0-based indexing and ensure bounds
        page_idx = max(0, start_page - 1)
        max_pages = min(len(doc), page_idx + 3)  # Look at up to 3 pages
        
        for page_num in range(page_idx, max_pages):
            try:
                page = doc[page_num]
                page_text = self.normalize_text(page.get_text())
                
                # If this is the first page, try to find content after the section title
                if page_num == page_idx:
                    content_after_title = self._extract_content_after_title(page_text, section_title)
                    if content_after_title:
                        content_parts.append(content_after_title)
                    else:
                        # If we can't find the title, take the whole page
                        content_parts.append(page_text)
                else:
                    # For subsequent pages, take all content until we hit a new section
                    page_content = self._extract_content_until_next_section(page_text)
                    if page_content:
                        content_parts.append(page_content)
                
                # Stop if we have enough content
                combined_content = ' '.join(content_parts)
                if len(combined_content) > 1000:
                    break
                    
            except Exception as e:
                print(f"Error extracting content from page {page_num}: {e}")
                continue
        
        return ' '.join(content_parts)

    def _extract_content_after_title(self, page_text: str, section_title: str) -> str:
        """Extract content that appears after a section title on the page."""
        # Try to find the section title in the text
        title_variations = [
            section_title,
            section_title.upper(),
            section_title.lower(),
            section_title.title()
        ]
        
        for title_var in title_variations:
            pattern = re.escape(title_var)
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                # Return content after the title
                content_after = page_text[match.end():].strip()
                if len(content_after) > 50:
                    return content_after
        
        # If title not found, return the whole page (it might be formatted differently)
        return page_text

    def _extract_content_until_next_section(self, page_text: str) -> str:
        """Extract content until we hit what looks like a new section heading."""
        lines = page_text.split('\n')
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line looks like a new section heading
            if self._looks_like_heading(line) and len(content_lines) > 0:
                break
            
            content_lines.append(line)
        
        return ' '.join(content_lines)

    def _looks_like_heading(self, line: str) -> bool:
        """Check if a line looks like a section heading."""
        if len(line) < 5 or len(line) > 150:
            return False
        
        for pattern in self.heading_patterns:
            if re.match(pattern, line):
                return True
        
        return False

    def _extract_heuristic_sections(self, doc, doc_name: str) -> List[Dict]:
        """Extract sections using enhanced heuristic approach."""
        sections = []
        
        # First pass: analyze document structure
        doc_analysis = self._analyze_document_structure(doc)
        
        current_section = None
        
        for page_num, page in enumerate(doc, 1):
            try:
                blocks = page.get_text("dict")["blocks"]
                if not blocks:
                    continue

                page_sections = self._process_page_blocks(
                    blocks, page_num, doc_name, doc_analysis
                )
                
                # Merge with current section or start new ones
                for section in page_sections:
                    if section.get('is_heading', False):
                        # Save previous section if it exists
                        if current_section and len(current_section.get('content', '').strip()) > 50:
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = section
                    else:
                        # Add content to current section
                        if current_section:
                            current_content = current_section.get('content', '')
                            new_content = section.get('content', '')
                            current_section['content'] = f"{current_content} {new_content}".strip()
                            
                            # Merge procedural sentences
                            current_proc = current_section.get('procedural_sentences', [])
                            new_proc = section.get('procedural_sentences', [])
                            current_section['procedural_sentences'] = current_proc + new_proc
                        else:
                            # No current section, start one with generic title
                            current_section = {
                                'document': doc_name,
                                'page_number': page_num,
                                'section_title': 'Content',
                                'content': section.get('content', ''),
                                'is_generic': True,
                                'procedural_sentences': section.get('procedural_sentences', [])
                            }
                
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                continue
        
        # Add the final section
        if current_section and len(current_section.get('content', '').strip()) > 50:
            sections.append(current_section)
        
        return sections

    def _analyze_document_structure(self, doc) -> Dict:
        """Enhanced document structure analysis."""
        all_font_sizes = []
        all_fonts = []
        text_blocks = []
        
        # Sample more pages for better analysis
        sample_pages = min(5, len(doc))
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if not block.get("lines"):
                    continue
                    
                block_text = self._extract_block_text(block)
                normalized_text = self.normalize_text(block_text)
                
                if len(normalized_text.strip()) < 5:
                    continue
                    
                text_blocks.append(normalized_text)
                
                # Collect font information
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_font_sizes.append(span["size"])
                        all_fonts.append(span["font"])
        
        return {
            "font_sizes": self._analyze_font_patterns(all_font_sizes),
            "common_fonts": Counter(all_fonts),
            "text_patterns": self._analyze_text_patterns(text_blocks),
            "total_blocks": len(text_blocks)
        }

    def _analyze_font_patterns(self, font_sizes: List[float]) -> Dict:
        """Enhanced font pattern analysis."""
        if not font_sizes:
            return {"body_size": 10, "heading_threshold": 12, "large_heading_threshold": 14}
        
        # Use more sophisticated analysis
        size_counts = Counter(font_sizes)
        sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Most common size is likely body text
        body_size = sorted_sizes[0][0]
        
        # Find heading thresholds based on distribution
        unique_sizes = sorted(set(font_sizes))
        body_idx = unique_sizes.index(body_size)
        
        # Heading thresholds
        heading_threshold = body_size + 1.0
        large_heading_threshold = body_size + 3.0
        
        # If there are larger sizes used frequently, adjust thresholds
        for size, count in sorted_sizes[1:]:
            if size > body_size and count >= len(font_sizes) * 0.05:  # At least 5% usage
                if size < heading_threshold + 2:
                    heading_threshold = min(heading_threshold, size - 0.5)
        
        return {
            "body_size": body_size,
            "heading_threshold": heading_threshold,
            "large_heading_threshold": large_heading_threshold,
            "size_distribution": size_counts,
            "unique_sizes": unique_sizes
        }

    def _process_page_blocks(self, blocks, page_num: int, doc_name: str, 
                           doc_analysis: Dict) -> List[Dict]:
        """Process blocks on a page with enhanced heading detection."""
        page_sections = []
        font_info = doc_analysis["font_sizes"]
        
        for block in blocks:
            if not block.get("lines"):
                continue
                
            block_text = self._extract_block_text(block)
            normalized_text = self.normalize_text(block_text)
            
            if not normalized_text.strip():
                continue
            
            # Enhanced heading detection
            is_heading = self._is_enhanced_heading(block, normalized_text, font_info)
            
            section_data = {
                'document': doc_name,
                'page_number': page_num,
                'content': normalized_text,
                'is_heading': is_heading,
                'procedural_sentences': self._extract_procedural_sentences(normalized_text)
            }
            
            if is_heading:
                section_data['section_title'] = self._clean_section_title(normalized_text)
                section_data['is_generic'] = self._is_generic_heading(normalized_text)
            
            page_sections.append(section_data)
        
        return page_sections

    def _is_enhanced_heading(self, block, normalized_text: str, font_info: Dict) -> bool:
        """Enhanced heading detection with multiple criteria."""
        text = normalized_text.strip()
        
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Font-based detection
        heading_score = 0
        
        if block.get("lines"):
            first_span = block["lines"][0]["spans"][0] if block["lines"][0].get("spans") else None
            if first_span:
                font_size = first_span["size"]
                font_name = first_span["font"].lower()
                
                # Font size scoring
                if font_size >= font_info["large_heading_threshold"]:
                    heading_score += 3
                elif font_size >= font_info["heading_threshold"]:
                    heading_score += 2
                
                # Font weight scoring
                if any(weight in font_name for weight in ["bold", "black", "heavy"]):
                    heading_score += 2
                
                # Font style considerations
                if "italic" in font_name and heading_score > 0:
                    heading_score += 1
        
        # Pattern-based scoring
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                heading_score += 2
                break
        
        # Structure-based scoring
        if text.isupper() and 10 < len(text) < 80:
            heading_score += 1
        
        if text.endswith(':') and len(text) < 100:
            heading_score += 1
        
        # Line count (headings are usually single line)
        if len(text.split('\n')) == 1 and len(text) < 150:
            heading_score += 1
        
        # Threshold for being considered a heading
        return heading_score >= 3

    def _extract_block_text(self, block) -> str:
        """Extract text from a block preserving structure."""
        lines = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                lines.append(line_text.strip())
        return " ".join(lines)

    def _is_generic_heading(self, title: str) -> bool:
        """Check if heading is generic and should be deboosted."""
        title_lower = title.lower().strip()
        return any(generic in title_lower for generic in self.generic_headings)

    def _clean_section_title(self, title: str) -> str:
        """Clean and normalize section titles."""
        title = title.strip()
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^\d+\.?\s*', '', title)  # Remove leading numbers
        title = re.sub(r'^[IVX]+\.\s*', '', title, flags=re.IGNORECASE)  # Remove roman numerals
        title = re.sub(r'^[a-z]\)\s*', '', title)  # Remove letter numbering
        title = re.sub(r':?\s*$', '', title)      # Remove trailing colons
        if '.' in title or len(title.split()) > 10:
        # Find all sequences of capitalized words (potential proper nouns/titles)
            matches = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', title)
            if matches:
                # The actual title is often the last one found in the line
                title = matches[-1]
        
        # Normalize case for very short or all-caps titles
        if title.isupper() and len(title) > 5:
            title = title.title()
        
        # Limit length
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title.strip()

    def _extract_procedural_sentences(self, text: str) -> List[str]:
        """Extract sentences that contain procedural/actionable content."""
        procedural_patterns = [
            r'\b(?:step|steps?)\s*\d*',
            r'\b(?:first|second|third|next|then|finally|last|follow|proceed)\b',
            r'\b(?:click|select|choose|open|close|save|create|delete|add|remove|edit|modify|visit|go|try|use|take|bring|pack|book|reserve|plan|consider)\b',
            r'\b(?:to\s+\w+|how\s+to)\b',
            r'^\d+\.\s+\w+',
            r'\b(?:should|must|need\s+to|have\s+to|ought\s+to)\b',
            r'\b(?:recommend|suggest|advise|tip|important|essential|remember)\b'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        procedural_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
            sentence_lower = sentence.lower()
            for pattern in procedural_patterns:
                if re.search(pattern, sentence_lower):
                    procedural_sentences.append(sentence)
                    break
        
        return procedural_sentences[:5]  # Limit to prevent bloat

    def _analyze_text_patterns(self, text_blocks: List[str]) -> Dict:
        """Analyze text patterns in the document."""
        patterns = {
            "numbered_sections": 0,
            "bullet_points": 0,
            "caps_headings": 0,
            "colon_labels": 0,
            "procedural_content": 0,
            "total_blocks": len(text_blocks)
        }
        
        for text in text_blocks:
            text_lower = text.lower().strip()
            
            if re.search(r'^\d+\.', text):
                patterns["numbered_sections"] += 1
            if re.search(r'^\s*[-•]', text):
                patterns["bullet_points"] += 1
            if text.isupper() and 10 < len(text) < 100:
                patterns["caps_headings"] += 1
            if re.search(r'^\w+:', text):
                patterns["colon_labels"] += 1
            
            # Count procedural content
            if self._extract_procedural_sentences(text):
                patterns["procedural_content"] += 1
        
        return patterns

    def _post_process_sections(self, sections: List[Dict]) -> List[Dict]:
        """Enhanced post-processing with better section merging and filtering."""
        if not sections:
            return sections
        
        processed = []
        
        for i, section in enumerate(sections):
            # Clean and validate content
            content = section.get('content', '').strip()
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            
            if len(content) < 40:  # Skip very short sections
                continue
            
            section['content'] = content
            
            # Enhance with procedural information
            procedural = section.get('procedural_sentences', [])
            if procedural:
                section['has_procedural_content'] = True
                section['procedural_count'] = len(procedural)
            else:
                section['has_procedural_content'] = False
                section['procedural_count'] = 0
            
            # Add content quality metrics
            section['content_length'] = len(content)
            section['word_count'] = len(content.split())
            
            processed.append(section)
        
        # Sort by page number to maintain document order
        processed.sort(key=lambda x: x['page_number'])
        
        return processed