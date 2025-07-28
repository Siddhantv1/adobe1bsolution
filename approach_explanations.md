***

# Approach Explanation

Our methodology for this document intelligence system is a multi-stage pipeline designed to deliver highly relevant, context-aware insights from diverse document collections. The approach moves beyond simple keyword matching to deeply understand user intent by combining robust document processing with an advanced, contrast-aware semantic analysis core.

---

## 1. Document Ingestion and Preprocessing

The first stage focuses on creating a clean and structured representation of the source documents. The system is designed to dynamically process any specified `Collection`, ingesting PDF documents using the **PyMuPDF** library. Raw text extracted from PDFs is often noisy, containing artifacts like incorrect spacing, broken words, and special characters.

Our `EnhancedDocumentProcessor` handles this by:
-   **Normalizing Text:** It standardizes whitespace, removes control characters, and corrects common PDF extraction issues, including mapping ligatures (e.g., 'ﬁ' to 'fi').
-   **Structural Analysis:** Instead of treating a document as a single block of text, the system analyzes font sizes, weights, and text patterns to heuristically identify and separate distinct sections, each with a title and corresponding content. This preserves the document's inherent structure, which is crucial for contextual understanding.

---

## 2. Semantic Analysis with Contrastive Scoring

This is the core of our system's intelligence. We use the powerful `intfloat/e5-small-v2` embedding model to convert both the user's query and the document sections into high-dimensional vectors. However, standard semantic search often fails when faced with negative constraints (e.g., a search for "vegetarian" food returning results with "chicken" because both are "dinner recipes").

To solve this, we implemented a **Contrastive Scoring** methodology:

1.  **Target Query Generation:** A primary query is generated directly from the persona and the "job-to-be-done" (JTBD). This represents the ideal information the user is seeking.

2.  **Contrasting Query Generation:** We create a semantic "anti-query" that represents the concept that must be excluded. For a "vegetarian" JTBD, the anti-query would describe meat-based dishes.

3.  **Dual Similarity Calculation:** Each document section is compared against *both* the target and the contrasting queries to get two scores: a relevance score and a contradiction score.

The final semantic score for a section is calculated as:
`Final Score = Relevance_Score - (Contradiction_Score * Penalty_Factor)`

This approach allows the system to not only find what is relevant but also to actively identify and penalize what is explicitly irrelevant, enforcing strict constraints in a purely semantic manner without hardcoding keywords.

---

## 3. Multi-Factor Ranking and Output

A section's final importance is determined by more than just semantic relevance. The system combines the contrastive semantic score with other quality metrics, including:
-   **Content Quality:** Word count and textual diversity.
-   **Structural Importance:** The section's location (page number) and title format.
-   **Procedural Content:** A bonus for sections containing actionable steps or instructions.

The highest-scoring, most diverse sections are selected and presented in a clean, structured `challenge1b_output.json` file, ordered with metadata first, followed by the extracted sections and detailed subsection analysis for user convenience.