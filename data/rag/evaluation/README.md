# Review-required evaluation seed bank

- `heldout_test`: freeze after human review; never train on its query/answer labels or near-duplicate templates.
- `validation`: may guide development, never used as final unbiased test.
- `train_candidate`: queries from eligible document families; manual review before training.
- `expected_no_answer=true`: do not fabricate a source answer.
- `gold_document_id`, `gold_version`, `gold_section_key` are *source anchors*, not implementation chunk IDs.
- Questions generated from uniform templates are NOT an independent industry-quality test. Add manually written difficult questions and external holdout corpus for a credible benchmark.
