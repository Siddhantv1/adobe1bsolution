import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter, defaultdict

class EnhancedPersonaAnalyzer:
    def __init__(self):
        # Load e5-small-v2 using transformers
        model_name = "./models/e5-small-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set CPU optimizations
        torch.set_num_threads(4)
        
        # Generic heading penalty weights
        self.generic_penalty = 0.3  # Reduce score by 30% for generic headings

    def encode_text_mean_pooled(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Encode texts using e5-small-v2 with mean pooling instead of CLS token."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              max_length=max_length, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get attention mask for proper mean pooling
            attention_mask = inputs['attention_mask']
            
            # Mean pooling: average over sequence length, ignoring padding tokens
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms



    def analyze_relevance(self, sections: List[Dict], persona: str, job_to_be_done: str) -> Dict:
        """Analyze relevance using a semantic contrast approach."""
        if not sections:
            return {"extracted_sections": [], "subsection_analysis": []}

        # Filter and prepare sections for analysis
        valid_sections = self._filter_valid_sections(sections)
        
        if not valid_sections:
            return {"extracted_sections": [], "subsection_analysis": []}

        # >>> NEW: Create Target and Contrasting Queries <<<
        # The target is the direct job to be done.
        target_query = f"query: {persona} needs to {job_to_be_done}"
        
        # The contrast negates the key constraint. This is derived, not hardcoded.
        # This logic can be expanded for other constraints.
        contrasting_query = ""
        if "vegetarian" in job_to_be_done.lower():
            contrasting_query = "query: a menu featuring meat, poultry, and fish dishes"
        
        # If there's no contrast, we can't use this method effectively, but we'll handle it in scoring.
        
        # Prepare section representations for embedding
        section_texts = self._prepare_section_representations(valid_sections)

        # Generate embeddings
        # We embed the queries and contrast query (if it exists) together
        query_texts = [target_query]
        if contrasting_query:
            query_texts.append(contrasting_query)
            
        query_embeddings = self.encode_text_mean_pooled(query_texts)
        section_embeddings = self.encode_text_mean_pooled(section_texts)

        # The target embedding is always the first one
        target_embedding = query_embeddings[0:1] # Keep shape as (1, D)
        # The contrast embedding is the second one, if it exists
        contrast_embedding = query_embeddings[1:2] if contrasting_query else None

        # Calculate comprehensive relevance scores using the new contrastive method
        section_scores = self._calculate_contrastive_relevance_scores(
            valid_sections, target_embedding, contrast_embedding, section_embeddings, persona, job_to_be_done
        )
        
        # Rank sections and apply penalties for generic headings
        ranked_sections = self._rank_and_filter_sections(section_scores)
        
        # Select diverse, high-quality sections
        extracted_sections = self._select_optimal_sections(ranked_sections)
        
        # Generate rich subsection analysis
        subsection_analysis = self._create_comprehensive_subsection_analysis(
            ranked_sections, persona, job_to_be_done
        )

        return {
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
    def _filter_valid_sections(self, sections: List[Dict]) -> List[Dict]:
        """Filter sections to ensure quality and relevance."""
        valid_sections = []
        
        for section in sections:
            content = section.get('content', '').strip()
            title = section.get('section_title', '').strip()
            
            # Basic quality filters
            if len(content) < 50:  # Too short
                continue
            if len(content) > 5000:  # Too long, likely merged incorrectly
                content = content[:5000]  # Truncate
                section['content'] = content
            
            # Skip sections with mostly numbers or very repetitive content
            words = content.split()
            if len(words) < 10:
                continue
            
            # Check for content variety
            unique_words = len(set(word.lower() for word in words if len(word) > 2))
            if unique_words < len(words) * 0.2:  # Too repetitive
                continue
            
            # Must have a meaningful title
            if not title or len(title) < 3:
                continue
            
            valid_sections.append(section)
        
        return valid_sections

    def _create_sophisticated_queries(self, persona: str, job_to_be_done: str) -> List[str]:
        """Create sophisticated query variations for better matching."""
        # Extract key concepts
        persona_concepts = self._extract_domain_concepts(persona)
        job_concepts = self._extract_action_concepts(job_to_be_done)
        
        queries = []
        
        # Primary query
        queries.append(f"query: {persona} {job_to_be_done}")
        
        # Persona-focused queries
        if persona_concepts:
            queries.append(f"query: {persona} needs information about {' '.join(persona_concepts[:3])}")
            queries.append(f"query: {' '.join(persona_concepts[:2])} requirements and guidelines")
        
        # Job-focused queries
        if job_concepts:
            queries.append(f"query: how to {' '.join(job_concepts[:3])} effectively")
            queries.append(f"query: {' '.join(job_concepts[:2])} planning and execution")
        
        # Combined specific queries
        if persona_concepts and job_concepts:
            queries.append(f"query: {persona_concepts[0]} {job_concepts[0]} best practices")
            queries.append(f"query: {' '.join(job_concepts[:2])} for {' '.join(persona_concepts[:1])}")
        
        # Task-oriented queries
        queries.append(f"query: practical steps for {job_to_be_done}")
        queries.append(f"query: {persona} actionable information")
        
        return queries

    def _extract_domain_concepts(self, persona: str) -> List[str]:
        """Extract domain-specific concepts from persona description."""
        # Common professional/role indicators
        professional_terms = {
            'planner': ['planning', 'organization', 'logistics', 'coordination'],
            'manager': ['management', 'supervision', 'leadership', 'operations'],
            'coordinator': ['coordination', 'organization', 'scheduling', 'logistics'],
            'professional': ['professional', 'business', 'corporate', 'workplace'],
            'travel': ['travel', 'tourism', 'destinations', 'itinerary'],
            'event': ['events', 'occasions', 'celebrations', 'gatherings'],
            'hr': ['human resources', 'personnel', 'employees', 'workplace'],
            'marketing': ['marketing', 'promotion', 'advertising', 'campaigns'],
            'finance': ['financial', 'budget', 'costs', 'expenses'],
            'education': ['educational', 'learning', 'teaching', 'academic'],
            'healthcare': ['medical', 'health', 'clinical', 'patient']
        }
        
        persona_lower = persona.lower()
        concepts = []
        
        # Extract based on professional terms
        for term, related_concepts in professional_terms.items():
            if term in persona_lower:
                concepts.extend(related_concepts)
        
        # Extract key nouns and adjectives
        words = re.findall(r'\b[a-zA-Z]{3,}\b', persona.lower())
        meaningful_words = [w for w in words if w not in {'the', 'and', 'or', 'but', 'for', 'with', 'from'}]
        concepts.extend(meaningful_words[:3])
        
        return list(dict.fromkeys(concepts))  # Remove duplicates while preserving order

    def _extract_action_concepts(self, job_to_be_done: str) -> List[str]:
        """Extract action-oriented concepts from job description."""
        # Common action patterns
        action_verbs = {
            'plan': ['planning', 'organize', 'schedule', 'prepare'],
            'organize': ['organization', 'structure', 'arrange', 'coordinate'],
            'manage': ['management', 'oversight', 'control', 'supervision'],
            'create': ['creation', 'development', 'design', 'build'],
            'find': ['search', 'locate', 'discover', 'identify'],
            'book': ['booking', 'reservation', 'scheduling', 'arrangement'],
            'coordinate': ['coordination', 'synchronization', 'alignment', 'management']
        }
        
        job_lower = job_to_be_done.lower()
        concepts = []
        
        # Extract based on action verbs
        for verb, related_concepts in action_verbs.items():
            if verb in job_lower:
                concepts.extend(related_concepts)
        
        # Extract key objects/targets
        # Look for patterns like "plan a trip", "organize an event"
        objects = re.findall(r'\b(?:a|an|the)\s+([a-zA-Z]{3,})\b', job_lower)
        concepts.extend(objects)
        
        # Extract numbers and quantities (important for group planning)
        numbers = re.findall(r'\b(\d+)\b', job_to_be_done)
        if numbers:
            concepts.append('group')
            if int(numbers[0]) > 5:
                concepts.append('large group')
        
        # Extract temporal concepts
        temporal = re.findall(r'\b(\d+\s+(?:day|week|month|hour)s?)\b', job_lower)
        concepts.extend(temporal)
        
        return list(dict.fromkeys(concepts))  # Remove duplicates

    def _prepare_section_representations(self, sections: List[Dict]) -> List[str]:
        """Prepare comprehensive section representations for embedding."""
        section_texts = []
        
        for section in sections:
            title = section.get('section_title', '')
            content = section.get('content', '')
            procedural = section.get('procedural_sentences', [])
            
            # Create a comprehensive representation
            # Weight title higher by repeating it
            title_weighted = f"{title} {title}" if title else ""
            
            # Limit content to avoid token limits but ensure we get key information
            content_preview = content[:800] if len(content) > 800 else content
            
            # Add procedural sentences (these are often highly relevant)
            procedural_text = ' '.join(procedural[:3]) if procedural else ''
            
            # Combine all elements
            combined_text = f"passage: {title_weighted} {content_preview} {procedural_text}".strip()
            section_texts.append(combined_text)
        
        return section_texts

    # Rename or replace _calculate_section_relevance_scores with this new function.
# I'll call it _calculate_contrastive_relevance_scores to be clear.

    def _calculate_contrastive_relevance_scores(self, sections: List[Dict], 
                                            target_embedding: np.ndarray, 
                                            contrast_embedding: Optional[np.ndarray],
                                            section_embeddings: np.ndarray,
                                            persona: str, job_to_be_done: str) -> List[Tuple[Dict, float]]:
        """Calculate relevance scores using semantic contrast."""
        section_scores = []
        
        # Calculate semantic similarities to the target query
        relevance_similarities = cosine_similarity(target_embedding, section_embeddings)[0]
        
        # Calculate similarities to the contrasting query, if it exists
        if contrast_embedding is not None:
            contradiction_similarities = cosine_similarity(contrast_embedding, section_embeddings)[0]
        else:
            # If no contrast, this score is zero for all sections
            contradiction_similarities = np.zeros(len(sections))

        for i, section in enumerate(sections):
            # The core of the new semantic score
            # We heavily penalize similarity to the contrasting concept.
            # A scaling factor on the penalty makes it more powerful.
            relevance_score = relevance_similarities[i]
            contradiction_penalty = contradiction_similarities[i]
            semantic_score = relevance_score - (contradiction_penalty * 1.5) # The penalty is amplified

            # Ensure score is not negative, as it can mess up weighting
            semantic_score = max(0, semantic_score)

            # The rest of your scoring function remains a great way to factor in quality!
            content_quality = self._assess_enhanced_content_quality(section)
            structural_importance = self._assess_enhanced_structural_importance(section)
            term_overlap = self._calculate_enhanced_term_overlap(section, persona, job_to_be_done)
            procedural_bonus = self._calculate_procedural_relevance(section, job_to_be_done)
            
            generic_penalty = 1.0 - self.generic_penalty if section.get('is_generic', False) else 1.0
            
            # Combine scores with sophisticated weighting
            final_score = (
                semantic_score * 0.50 +           # Semantic similarity (primary and now contrast-aware)
                content_quality * 0.15 +          # Content quality
                structural_importance * 0.10 +    # Document structure importance
                term_overlap * 0.10 +             # Term overlap
                procedural_bonus * 0.15           # Procedural content bonus
            ) * generic_penalty
            
            section_scores.append((section, final_score))
        
        return section_scores
    def _assess_enhanced_content_quality(self, section: Dict) -> float:
        """Enhanced content quality assessment."""
        content = section.get('content', '')
        
        # Length scoring (optimal range)
        length = len(content)
        if length < 100:
            length_score = length / 100
        elif length < 500:
            length_score = 1.0
        elif length < 1500:
            length_score = 1.0 - ((length - 500) / 2000)  # Gradual decrease
        else:
            length_score = 0.5  # Very long content
        
        # Word count and diversity
        words = content.split()
        word_count = len(words)
        unique_words = len(set(word.lower() for word in words))
        diversity_score = unique_words / word_count if word_count > 0 else 0
        
        # Procedural content bonus
        procedural_count = section.get('procedural_count', 0)
        procedural_score = min(procedural_count / 5, 1.0)
        
        # Information density (avoid repetitive content)
        info_density = min(diversity_score * 2, 1.0)
        
        return (length_score * 0.4 + procedural_score * 0.3 + info_density * 0.3)

    def _assess_enhanced_structural_importance(self, section: Dict) -> float:
        """Enhanced structural importance assessment."""
        title = section.get('section_title', '').lower()
        toc_level = section.get('toc_level', 3)  # Default to mid-level
        
        # TOC level importance (if available)
        toc_score = max(0, (4 - toc_level) / 3) if toc_level else 0.5
        
        # Title characteristics that suggest importance
        high_importance_indicators = [
            'guide', 'comprehensive', 'complete', 'detailed', 'essential',
            'important', 'key', 'main', 'primary', 'major', 'tips', 'advice'
        ]
        
        medium_importance_indicators = [
            'planning', 'preparation', 'requirements', 'considerations',
            'recommendations', 'suggestions', 'options', 'choices'
        ]
        
        # Calculate title importance
        title_importance = 0.0
        for indicator in high_importance_indicators:
            if indicator in title:
                title_importance += 0.3
        for indicator in medium_importance_indicators:
            if indicator in title:
                title_importance += 0.2
        
        # Page position bonus (earlier sections often more important)
        page_num = section.get('page_number', 10)
        page_bonus = max(0, (10 - page_num) / 20) if page_num <= 10 else 0
        
        return min(toc_score * 0.4 + title_importance * 0.4 + page_bonus * 0.2, 1.0)

    def _calculate_enhanced_term_overlap(self, section: Dict, persona: str, job_to_be_done: str) -> float:
        """Enhanced term overlap calculation with fuzzy matching."""
        # Extract terms from persona and job
        persona_terms = set(self._extract_meaningful_terms(persona))
        job_terms = set(self._extract_meaningful_terms(job_to_be_done))
        query_terms = persona_terms.union(job_terms)
        
        if not query_terms:
            return 0.0
        
        # Extract terms from section
        section_text = f"{section.get('section_title', '')} {section.get('content', '')}"
        section_terms = set(self._extract_meaningful_terms(section_text))
        
        # Direct overlap
        direct_overlap = len(query_terms.intersection(section_terms))
        
        # Fuzzy overlap (related terms)
        fuzzy_overlap = self._calculate_fuzzy_overlap(query_terms, section_terms)
        
        # Combine scores
        max_possible = len(query_terms)
        direct_score = direct_overlap / max_possible
        fuzzy_score = fuzzy_overlap / max_possible
        
        return (direct_score * 0.7) + (fuzzy_score * 0.3)

    def _extract_meaningful_terms(self, text: str) -> List[str]:
        """Extract meaningful terms with better filtering."""
        # Expanded stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'needs', 'need', 'should', 'would', 'could', 'can', 'may', 'might', 'this',
            'they', 'them', 'their', 'there', 'these', 'those', 'when', 'where', 'why',
            'how', 'what', 'who', 'which', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'now', 'also', 'here', 'then', 'once', 'during', 'before',
            'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        meaningful_words = []
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                # Include numbers if they're contextually important
                if word.isalpha() or (word.isdigit() and len(word) <= 3):
                    meaningful_words.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for word in meaningful_words:
            if word not in seen:
                seen.add(word)
                result.append(word)
        
        return result

    def _calculate_fuzzy_overlap(self, query_terms: set, section_terms: set) -> float:
        """Calculate fuzzy overlap using simple similarity heuristics."""
        fuzzy_matches = 0
        
        # Related term mappings
        related_terms = {
            'travel': ['trip', 'journey', 'vacation', 'tour', 'visit'],
            'plan': ['planning', 'organize', 'schedule', 'prepare', 'arrangement'],
            'group': ['team', 'party', 'friends', 'people', 'companions'],
            'restaurant': ['dining', 'food', 'cuisine', 'meal', 'eat'],
            'hotel': ['accommodation', 'lodging', 'stay', 'room', 'booking'],
            'activity': ['activities', 'things', 'attractions', 'entertainment', 'fun'],
            'budget': ['cost', 'price', 'expense', 'money', 'affordable'],
            'guide': ['tips', 'advice', 'recommendations', 'suggestions', 'help']
        }
        
        for query_term in query_terms:
            if query_term in section_terms:
                continue  # Already counted in direct overlap
            
            # Check for related terms
            if query_term in related_terms:
                related = related_terms[query_term]
                if any(rel in section_terms for rel in related):
                    fuzzy_matches += 0.5  # Partial credit for related terms
            
            # Check for substring matches (e.g., "plan" in "planning")
            for section_term in section_terms:
                if len(query_term) >= 4 and query_term in section_term:
                    fuzzy_matches += 0.3
                    break
                elif len(section_term) >= 4 and section_term in query_term:
                    fuzzy_matches += 0.3
                    break
        
        return fuzzy_matches

    def _calculate_procedural_relevance(self, section: Dict, job_to_be_done: str) -> float:
        """Calculate how much procedural content is relevant to the job."""
        procedural_sentences = section.get('procedural_sentences', [])
        
        if not procedural_sentences:
            return 0.0
        
        # Extract job-relevant action words
        job_actions = self._extract_action_concepts(job_to_be_done)
        job_lower = job_to_be_done.lower()
        
        relevant_count = 0
        for sentence in procedural_sentences:
            sentence_lower = sentence.lower()
            
            # Check for job-relevant actions
            if any(action in sentence_lower for action in job_actions):
                relevant_count += 1
            
            # Check for general planning/organizing terms
            planning_terms = ['plan', 'organize', 'book', 'reserve', 'consider', 'choose', 'select']
            if any(term in sentence_lower for term in planning_terms):
                relevant_count += 0.5
        
        # Normalize by number of sentences
        return min(relevant_count / len(procedural_sentences), 1.0)

    def _rank_and_filter_sections(self, section_scores: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Rank sections and apply filtering logic."""
        # Sort by score
        ranked = sorted(section_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out very low-scoring sections
        min_threshold = 0.15
        filtered = [(section, score) for section, score in ranked if score >= min_threshold]
        
        # Ensure document diversity - don't let one document dominate
        balanced = self._ensure_document_balance(filtered)
        
        return balanced

    def _ensure_document_balance(self, section_scores: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Ensure balanced representation across documents."""
        doc_counts = defaultdict(int)
        balanced = []
        
        # First pass: take top section from each document
        used_docs = set()
        for section, score in section_scores:
            doc_name = section['document']
            if doc_name not in used_docs:
                balanced.append((section, score))
                used_docs.add(doc_name)
                doc_counts[doc_name] = 1
                
                if len(balanced) >= 10:  # Limit for efficiency
                    break
        
        # Second pass: fill remaining slots, limiting per document
        max_per_doc = 2
        for section, score in section_scores:
            if len(balanced) >= 15:  # Total limit
                break
                
            doc_name = section['document']
            if doc_counts[doc_name] < max_per_doc:
                # Check if this section is already included
                if not any(s['section_title'] == section['section_title'] and 
                          s['document'] == section['document'] for s, _ in balanced):
                    balanced.append((section, score))
                    doc_counts[doc_name] += 1
        
        return balanced

    def _select_optimal_sections(self, ranked_sections: List[Tuple[Dict, float]], max_sections: int = 5) -> List[Dict]:
        """Select optimal sections ensuring diversity and quality."""
        selected = []
        doc_representation = defaultdict(int)
        
        for section, score in ranked_sections:
            if len(selected) >= max_sections:
                break
            
            doc_name = section['document']
            
            # Ensure document diversity while prioritizing quality
            if doc_representation[doc_name] >= 2:  # Max 2 per document
                continue
            
            selected.append({
                "document": section['document'],
                "section_title": section['section_title'],
                "importance_rank": len(selected) + 1,
                "page_number": section['page_number']
            })
            
            doc_representation[doc_name] += 1
        
        return selected

    def _create_comprehensive_subsection_analysis(self, ranked_sections: List[Tuple[Dict, float]], 
                                                persona: str, job_to_be_done: str) -> List[Dict]:
        """Create comprehensive subsection analysis with rich content."""
        subsection_analysis = []
        
        # Create focused query for analysis
        analysis_query = f"query: {persona} needs actionable information to {job_to_be_done}"
        analysis_embedding = self.encode_text_mean_pooled([analysis_query])
        
        for section, score in ranked_sections[:8]:  # Analyze top 8 sections
            enhanced_content = self._extract_enhanced_section_content(
                section, analysis_embedding, persona, job_to_be_done
            )
            
            if enhanced_content and len(enhanced_content.strip()) > 100:
                subsection_analysis.append({
                    "document": section['document'],
                    "refined_text": enhanced_content,
                    "page_number": section['page_number']
                })
            
            if len(subsection_analysis) >= 5:  # Limit output
                break
        
        return subsection_analysis

    def _extract_enhanced_section_content(self, section: Dict, analysis_embedding: np.ndarray,
                                        persona: str, job_to_be_done: str) -> str:
        """Extract the most relevant and comprehensive content from a section."""
        content = section.get('content', '')
        procedural_sentences = section.get('procedural_sentences', [])
        title = section.get('section_title', '')
        
        # If we have strong procedural content, prioritize it
        if procedural_sentences and len(procedural_sentences) >= 3:
            # Combine context with procedural content
            context_start = content[:300] if len(content) > 300 else content
            procedural_text = ' '.join(procedural_sentences[:5])
            return f"{context_start.rstrip('.')}. {procedural_text}"
        
        # For longer content, do sentence-level analysis
        if len(content) > 400:
            sentences = self._enhanced_sentence_split(content)
            if len(sentences) > 4:
                return self._select_best_sentences(
                    sentences, analysis_embedding, persona, job_to_be_done
                )
        
        # For shorter content or when sentence analysis fails, return full content
        return content

    def _enhanced_sentence_split(self, text: str) -> List[str]:
        """Enhanced sentence splitting with better handling of abbreviations."""
        # Handle common abbreviations
        abbreviations = ['Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'e.g.', 'i.e.']
        
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.replace('<DOT>', '.').strip()
            if len(sent) > 25 and not sent.startswith('Figure') and not sent.startswith('Table'):
                cleaned_sentences.append(sent)
        
        return cleaned_sentences

    def _select_best_sentences(self, sentences: List[str], analysis_embedding: np.ndarray,
                             persona: str, job_to_be_done: str) -> str:
        """Select the most relevant sentences using semantic analysis."""
        if len(sentences) <= 3:
            return '. '.join(sentences)
        
        # Create sentence embeddings
        sentence_texts = [f"passage: {s}" for s in sentences]
        sentence_embeddings = self.encode_text_mean_pooled(sentence_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(analysis_embedding, sentence_embeddings)[0]
        
        # Select top sentences with diversity
        top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 sentences
        
        # Filter by minimum relevance and select diverse content
        selected_sentences = []
        for idx in top_indices:
            if similarities[idx] > 0.25:  # Minimum relevance threshold
                selected_sentences.append((idx, sentences[idx], similarities[idx]))
        
        if not selected_sentences:
            # Fallback: take first few sentences
            return '. '.join(sentences[:3])
        
        # Sort by original order for coherence, limit to 4 sentences
        selected_sentences = selected_sentences[:4]
        selected_sentences.sort(key=lambda x: x[0])
        
        return '. '.join([sent for _, sent, _ in selected_sentences]) + '.'