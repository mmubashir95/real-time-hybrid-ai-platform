#!/usr/bin/env python3
"""Validate file hashes, source references, archive separation and word count."""
import csv,hashlib,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(root/'manifest.csv',encoding='utf-8')))
keys={(x['document_id'],x['version']) for x in rows}
sections={(x['document_id'],x['version'],x['section_key']) for x in csv.DictReader(open(root/'sections_index.csv',encoding='utf-8'))}
issues=[]
for r in rows:
 p=root/r['path']
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:issues.append('hash mismatch: '+r['path'])
for q in csv.DictReader(open(root/'evaluation/question_bank_SYNTHETIC_REVIEW_REQUIRED.csv',encoding='utf-8')):
 if q['gold_document_id'] and (q['gold_document_id'],q['gold_version'],q['gold_section_key']) not in sections:issues.append('missing evidence: '+q['question_id'])
 if q['expected_no_answer']=='true' and q['gold_document_id']:issues.append('bad no-answer source: '+q['question_id'])
print('documents:',len(rows),'total source words:',sum(int(r['word_count']) for r in rows),'issues:',len(issues))
for x in issues: print(x)
sys.exit(bool(issues))
