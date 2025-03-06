#!/bin/bash

### External Input
#
# dl-queries.json: Convert queries to a JSON dictionary mapping query ID to query Text
#
# trecDL2020-qrels-runs-with-text.jsonl.gz:  Collect passages from system responses (ranking or generated text) for grading
#    These follow the data interchange model, providing the Query ID, paragraph_id, text. 
#    System's rank information can be stored in paragraph_data.rankings[]
#    If available, manual judgments can be stored in paragraph_data.judgment[]


### Phase 1: Test bank generation
#
# Generating an initial test bank from a set of test nuggets or exam questions.
# 
# The following files are produced:
#
# dl20-questions.jsonl.gz: Generated exam questions
#
# dl20-nuggets.jsonl.gz Generated test nuggets


echo -e "\n\n\nGenerate DL20 Nuggets"

python -O -m exam_pp.question_generation -q data/dl20/dl20-queries.json -o data/dl20/dl20-nuggets.jsonl.gz --use-nuggets --test-collection dl20 --description "A new set of generated nuggets for DL20"

echo -e "\n\n\Generate DL20 Questions"

python -O -m exam_pp.question_generation -q data/dl20/dl20-queries.json -o data/dl20/dl20-questions.jsonl.gz --test-collection dl20 --description "A new set of generated questions for DL20"

