# Round 1B : Persona-Driven Document Intelligence

An advanced pipeline designed to analyze a collection of PDF documents and extract the most relevant sections based on a specific persona and a "job-to-be-done" (JTBD). It uses a semantic language model with contrastive scoring to understand nuanced requirements and filter out irrelevant content.

---

## 🚀 Features

-   **Semantic Understanding**: Leverages the `intfloat/e5-small-v2` transformer model to understand the context and meaning of text, not just keywords.
-   **Persona-Driven Analysis**: Tailors the search and ranking process to a specific user role (e.g., "Food Contractor").
-   **Contrastive Scoring**: Intelligently filters out irrelevant content by using an "anti-query" to identify and penalize contradictory information (e.g., ensuring a "vegetarian" search excludes meat dishes).
-   **Structured Output**: Generates a clean, ordered JSON file containing extracted sections, detailed analysis, and processing metadata.
-   **Dockerized**: Includes a `Dockerfile` for easy, dependency-free setup and execution in a containerized environment.

---

## 📂 Project Structure

For the application to run correctly, your project must follow this directory structure:

```
root/
├── models/
│   └── e5-small-v2/
│       ├── ... (model files)
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   └── persona_analyzer.py
├── app/
│   └── input/
│       ├── challenge1b_input.json
        └── (input files)
│   └── output/
│       └── (generated files will appear here)
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup and Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

-   Python 3.8+
-   [Docker](https://www.docker.com/get-started) (Recommended for containerized setup)

### Step 1: Clone the Repository

Clone or download the project files to your local machine.

### Step 2: Download the Language Model

This project requires the `intfloat/e5-small-v2` model.

1.  Go to the model's Hugging Face page: [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)
2.  Click on the "Files and versions" tab.
3.  Download all the files from the model repository.
4.  Create the `models/e5-small-v2` directory in your project root and place all the downloaded model files inside it.

### Step 3: Add Input Files

1.  Place all the PDF documents you want to analyze inside the `PDF/` directory.
2.  Place your configuration file (e.g., `challenge1b_input.json`) inside the `PDF/` directory. This file tells the system which documents to process and defines the persona and JTBD.

### Step 4: Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

You can run the analysis pipeline either directly with Python or using Docker.

### Option 1: Running Directly with Python

Execute the `main.py` script from the root of the project directory.

```bash
python main.py
```

The script will start processing the documents as specified in your input JSON.

### Option 2: Running with Docker (Recommended)

Using Docker ensures that the environment is consistent and all dependencies are handled automatically.

1.  **Build the Docker image:**
    From the project's root directory, run the following command. This will build an image named `doc-analyzer`.

    ```bash
    docker build -t doc-analyzer .
    ```

2.  **Run the Docker container:**
    This command runs the analysis. The `-v` flag creates a volume that maps the output directory inside the container to your local `app/output` directory, so you can easily access the results.

    ```bash
    docker run --rm -v "$(pwd)/app/output:/app/app/output" doc-analyzer
    ```

---

## ✅ Output

Upon successful execution, the output will be saved in the `app/output/` directory.

-   `collection1output.json`: The main output file containing the metadata, a ranked list of the most relevant sections, and the detailed subsection analysis.
-   `debug_info.json`: An optional file with more detailed processing statistics (if generated).
