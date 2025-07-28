import os
import json
import time
from datetime import datetime
from typing import Dict, List
import logging

# Import the enhanced processors
from src.document_processor import EnhancedDocumentProcessor
from src.persona_analyzer import EnhancedPersonaAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_simple_value(value):
    """Extract simple string value from complex input structures."""
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        for key in ['role', 'task', 'name', 'filename', 'title', 'value']:
            if key in value:
                return str(value[key])
        if value:
            return str(list(value.values())[0])
    return str(value)

def extract_document_names(documents):
    """Extract document names from various input formats."""
    simple_docs = []
    for doc in documents:
        if isinstance(doc, str):
            simple_docs.append(doc)
        elif isinstance(doc, dict):
            filename = (doc.get('filename') or doc.get('name') or
                       doc.get('document') or doc.get('file'))
            if filename:
                simple_docs.append(str(filename))
    return simple_docs

def validate_and_enhance_sections(sections: List[Dict]) -> List[Dict]:
    """Validate and enhance extracted sections with quality metrics."""
    valid_sections = []

    for section in sections:
        # Ensure all required fields are present
        required_fields = ['document', 'section_title', 'content', 'page_number']
        if not all(key in section for key in required_fields):
            logger.warning(f"Section missing required fields: {section.get('section_title', 'Unknown')}")
            continue

        # Clean up section title
        title = str(section['section_title']).strip()
        if len(title) > 100:  # Truncate very long titles
            title = title[:97] + "..."
        section['section_title'] = title

        # Ensure content meets minimum quality standards
        content = str(section['content']).strip()
        if len(content) < 40:
            logger.debug(f"Skipping short section: {title}")
            continue

        # Add quality metrics
        words = content.split()
        section['word_count'] = len(words)
        section['char_count'] = len(content)

        # Check for procedural content quality
        procedural = section.get('procedural_sentences', [])
        section['procedural_quality'] = len(procedural) > 0

        valid_sections.append(section)

    logger.info(f"Validated {len(valid_sections)} sections from {len(sections)} extracted")
    return valid_sections

def print_detailed_processing_stats(sections: List[Dict], result: Dict):
    """Print comprehensive processing statistics."""
    logger.info("\n" + "="*50)
    logger.info("PROCESSING STATISTICS")
    logger.info("="*50)

    # Basic counts
    logger.info(f"Total sections extracted: {len(sections)}")

    # Document distribution
    doc_counts = {}
    doc_quality = {}

    for section in sections:
        doc = section.get('document', 'Unknown')
        doc_counts[doc] = doc_counts.get(doc, 0) + 1

        # Track quality metrics per document
        if doc not in doc_quality:
            doc_quality[doc] = {'total_words': 0, 'procedural_sections': 0}

        doc_quality[doc]['total_words'] += section.get('word_count', 0)
        if section.get('procedural_quality', False):
            doc_quality[doc]['procedural_sections'] += 1

    logger.info("\nSections per document:")
    for doc, count in sorted(doc_counts.items()):
        words = doc_quality[doc]['total_words']
        procedural = doc_quality[doc]['procedural_sections']
        logger.info(f"  {doc}: {count} sections, {words:,} words, {procedural} with procedures")

    # Quality analysis
    total_words = sum(s.get('word_count', 0) for s in sections)
    procedural_sections = sum(1 for s in sections if s.get('procedural_quality', False))

    logger.info(f"\nContent Quality:")
    logger.info(f"  Total words extracted: {total_words:,}")
    logger.info(f"  Average words per section: {total_words/len(sections):.1f}")
    logger.info(f"  Sections with procedural content: {procedural_sections} ({procedural_sections/len(sections)*100:.1f}%)")

    # Results summary
    extracted_sections = result.get('extracted_sections', [])
    subsection_analysis = result.get('subsection_analysis', [])

    logger.info(f"\nFinal Output:")
    logger.info(f"  Selected sections: {len(extracted_sections)}")
    logger.info(f"  Subsection analyses: {len(subsection_analysis)}")

    # Show selected sections
    logger.info(f"\nSelected Sections:")
    for i, section in enumerate(extracted_sections, 1):
        logger.info(f"  {i}. [{section['document']}] {section['section_title']}")

def main():
    """Enhanced main execution function."""
    logger.info("Starting Enhanced Persona-Driven Document Intelligence System")

    # Configuration
    input_dir = "./app/input"
    output_dir = "./app/output"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize processors
    logger.info("Initializing enhanced document processor and analyzer...")
    try:
        processor = EnhancedDocumentProcessor()
        analyzer = EnhancedPersonaAnalyzer()
        logger.info("✓ Processors initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        return

    # Load configuration
    config_path = os.path.join(input_dir, "challenge1b_input.json")
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Extract configuration values
    documents = config.get("documents", [])
    persona_raw = config.get("persona", "")
    job_raw = config.get("job_to_be_done", "")

    persona = extract_simple_value(persona_raw)
    job_to_be_done = extract_simple_value(job_raw)
    document_names = extract_document_names(documents)

    logger.info(f"Configuration:")
    logger.info(f"  Documents to process: {len(document_names)}")
    logger.info(f"  Persona: '{persona}'")
    logger.info(f"  Job to be done: '{job_to_be_done}'")

    # Validate document files
    valid_docs = [doc_name for doc_name in document_names if os.path.exists(os.path.join(input_dir, doc_name))]
    missing_docs = [doc_name for doc_name in document_names if doc_name not in valid_docs]

    if not valid_docs:
        logger.error("No valid documents found to process")
        return

    for doc_name in valid_docs:
        logger.info(f"  ✓ Found: {doc_name}")
    for doc_name in missing_docs:
        logger.warning(f"  ✗ Missing: {doc_name}")

    # Start processing
    start_time = time.time()
    logger.info(f"\nStarting document processing...")

    # Process all documents
    all_sections = []
    for doc_name in valid_docs:
        doc_path = os.path.join(input_dir, doc_name)
        try:
            logger.info(f"Processing: {doc_name}")
            sections = processor.extract_sections(doc_path, doc_name)
            all_sections.extend(sections)
            logger.info(f"  ✓ Extracted {len(sections)} sections")
        except Exception as e:
            logger.error(f"  ✗ Error processing {doc_name}: {e}")

    # Validate sections
    all_sections = validate_and_enhance_sections(all_sections)

    if not all_sections:
        logger.error("No valid sections extracted from any documents")
        return

    logger.info(f"✓ Total valid sections extracted: {len(all_sections)}")

    # Analyze relevance
    logger.info("Analyzing relevance using enhanced semantic analysis...")
    analysis_start_time = time.time()

    try:
        analysis_result = analyzer.analyze_relevance(all_sections, persona, job_to_be_done)
        analysis_time = time.time() - analysis_start_time
        logger.info(f"✓ Relevance analysis completed in {analysis_time:.2f}s")
    except Exception as e:
        logger.error(f"✗ Error during relevance analysis: {e}")
        return

    # Create metadata dictionary
    total_processing_time = time.time() - start_time
    metadata = {
        "input_documents": document_names,
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": datetime.now().isoformat()
    }

    # Construct the final output dictionary in the desired order
    final_output = {
        "metadata": metadata,
        "extracted_sections": analysis_result.get('extracted_sections', []),
        "subsection_analysis": analysis_result.get('subsection_analysis', [])
    }

    # Save the correctly ordered output
    output_path = os.path.join(output_dir, "challenge1b_output.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save output: {e}")
        return

    # Print comprehensive results
    logger.info(f"\n" + "="*50)
    logger.info("PROCESSING COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
    logger.info(f"Sections processed: {len(all_sections)}")
    logger.info(f"Final sections selected: {len(final_output.get('extracted_sections', []))}")
    logger.info(f"Subsection analyses: {len(final_output.get('subsection_analysis', []))}")

    # Print detailed statistics
    print_detailed_processing_stats(all_sections, final_output)

    logger.info(f"\n🎉 Enhanced processing completed successfully!")
    logger.info(f"📁 Results saved to: {output_path}")

if __name__ == "__main__":
    main()