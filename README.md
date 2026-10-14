# Text-to-SQL Conversion using QLoRA

An LLM-powered Text-to-SQL project that translates natural-language questions into SQL queries using a fine-tuned **Qwen2.5-Coder-1.5B-Instruct** model. The project explores parameter-efficient fine-tuning, SQL query validation, database execution, and error-feedback retry to improve query generation.

## Overview

The goal of this project is to make database querying more accessible by allowing users to express questions in natural language instead of writing SQL manually.

The model was fine-tuned on a subset of the **Spider Text-to-SQL dataset** using QLoRA and LoRA adapters. The project also includes a query validation and execution workflow designed to restrict execution to read-only SQL queries.

## Key Features

- Natural-language-to-SQL query generation.
- QLoRA-based fine-tuning with 4-bit quantization and LoRA adapters.
- Training on 3,000 examples from the Spider dataset.
- SQL query validation using SQLGlot.
- Read-only SQLite query execution.
- Error-feedback retry to attempt to correct unsuccessful queries.
- Evaluation comparing the base model, fine-tuned model, and retry-enhanced workflow.

## Tech Stack

- **Language:** Python
- **Model:** Qwen2.5-Coder-1.5B-Instruct
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Fine-tuning:** PEFT, LoRA, QLoRA
- **Dataset:** Spider Text-to-SQL
- **SQL Processing:** SQLGlot, SQLite
- **Backend:** FastAPI

## Evaluation Results

Evaluation was performed on a 200-example development subset using simplified execution accuracy.

| Configuration | Correct examples | Accuracy |
|---|---:|---:|
| Base model | 85/200 | 42.5% |
| Fine-tuned model | 101/200 | 50.5% |
| Fine-tuned model with one retry | 109/200 | 54.5% |

These results are from the project's evaluation setup and should not be interpreted as official Spider benchmark scores.

## Project Workflow

1. Load and preprocess the Text-to-SQL dataset.
2. Fine-tune the Qwen2.5-Coder model using QLoRA.
3. Generate SQL queries from natural-language questions.
4. Validate generated SQL queries before execution.
5. Execute permitted queries against SQLite.
6. Use execution errors as feedback for one retry.
7. Evaluate the base model and improved configurations.


## Running the Project

The notebook contains the model fine-tuning and evaluation workflow. Training requires a compatible environment with the necessary Python libraries and sufficient compute resources.


## Limitations

- Fine-tuning used a subset of the Spider dataset rather than the entire dataset.
- Evaluation was performed on a 200-example development subset.
- Simplified execution accuracy is not directly comparable to official Spider benchmark metrics.
- Generated SQL must be validated carefully before execution; SQL validation alone does not guarantee complete database security.

## Future Improvements

- Evaluate on a larger development set using standardized benchmark metrics.
- Improve SQL generation for complex queries and joins.
- Expand error-feedback and query-correction strategies.



