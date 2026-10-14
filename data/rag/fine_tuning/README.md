# Not training-ready — synthetic seeds only

The pairs CSV contains simplistic synthetic positives/negatives useful for understanding the format. **Do not train a production reranker on these as-is.**

To create a defensible experiment: manually review relevance; derive hard negatives from retrieved *real passage candidates*, separate document-family / template groups across train/validation/test, hold test queries and labels completely out of training, train a small pretrained cross-encoder, then compare reranked *identical* candidates. Avoid accidentally training on gold answers from `evaluation/`.

A strong portfolio experiment also uses a second permitted natural-document set to test generalization.
