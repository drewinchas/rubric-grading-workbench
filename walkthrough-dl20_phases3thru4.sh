#!/bin/bash

#
# THIS SCRIPT ONLY INCLUDES PHASES 3 - 4!
#

ungraded="trecDL2020-qrels-runs-with-text.jsonl.gz"

withrate="nuggets-rate--all-${ungraded}"
withrateextract="nuggets-explain--${withrate}"

ungraded="$withrateextract"
withrate="questions-rate--${ungraded}"
withrateextract="questions-explain--${withrate}"

final=$withrateextract

#### Phase 3: Manual verification and Supervision
# We demonstrate how we support humans conducting a manual supervision of the process
#
# the files produced in this phase are:
# dl-verify-grading.txt : answers to the grading propts selfrated/extraction (grouped by question/nugget)
# dl20-bad-question.txt : Questions/nuggets frequently covered by non-relevant passages (should be removed from the test bank)
# dl20-uncovered-passages.txt : Relevant passages not covered by any question/nugget (require the addition of new test nuggets/questions.
#

python -O -m exam_pp.exam_verification --verify-grading data/dl20/$final  --question-path data/dl20/dl20-questions.jsonl.gz  --question-type question-bank  > data/dl20/dl20-verify-grading.txt

python -O -m exam_pp.exam_verification --uncovered-passages data/dl20/$final --question-path data/dl20/dl20-questions.jsonl.gz  --question-type question-bank --min-judgment 1 --min-rating 4 > data/dl20/dl20-uncovered-passages.txt

python -O -m exam_pp.exam_verification --bad-question data/dl20/$final  --question-path data/dl20/dl20-questions.jsonl.gz  --question-type question-bank --min-judgment 1 --min-rating 4  >  data/dl20/dl20-bad-question.txt



#### Phase 4: Evaluation
#
# We demonstrate both the Autograder-qrels  and Autograder-cover evaluation approaches
# Both require to select the grades to be used via --model and --prompt_class
# Here we use --model google/flan-t5-large
# and as --prompt_class either QuestionSelfRatedUnanswerablePromptWithChoices or NuggetSelfRatedPrompt.
#
# Alternatively, for test banks with exam questions that have known correct answers (e.g. TQA for CAR-y3), 
# the prompt class QuestionCompleteConcisePromptWithAnswerKey2 can be used to assess answerability.
#
# The files produced in this phase are:
#
# dl20-autograde-qrels-\$promptclass-minrating-4.solo.qrels:  Exported Qrel file treating passages with self-ratings >=4 
#
# dl20-autograde-qrels-leaderboard-\$promptclass-minrating-4.solo.tsv:  Leaderboard produced with 
#        trec_eval using the exported Qrel file
#
# dl20-autograde-cover-leaderboard-\$promptclass-minrating-4.solo.tsv: Leaderboads produced with Autograde Cover treating \
# 	test nuggets/questions as answered when any passage obtains a self-ratings >= 4
#
#
#
#
for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt; do
	echo $promptclass

	for minrating in 3 4 5; do
		python -O -m exam_pp.exam_evaluation data/dl20/$final --question-set question-bank --prompt-class $promptclass --min-self-rating $minrating --leaderboard-out data/dl20/dl20-autograde-cover-leaderboard-$promptclass-minrating-$minrating.solo.$ungraded.tsv 

		# N.B. requires TREC-DL20 runs to be populated in data/dl20/dl20runs
		python -O -m exam_pp.exam_evaluation data/dl20/$final --question-set question-bank --prompt-class $promptclass -q data/dl20/dl20-autograde-qrels-leaderboard-$promptclass-minrating-$minrating.solo.$ungraded.qrels  --min-self-rating $minrating --qrel-leaderboard-out data/dl20/dl20-autograde-qrels-$promptclass-minrating-$minrating.solo.$ungraded.tsv --run-dir data/dl20/dl20runs 
        
		# Since generative IR systems will not share any passages, we represent them as special run files
		#python -O -m exam_pp.exam_evaluation data/dl20/$final --question-set question-bank --prompt-class $promptclass -q data/dl20/dl20-autograde-qrels-leaderboard-$promptclass-minrating-$minrating.solo.$ungraded.qrels  --min-self-rating $minrating --qrel-leaderboard-out data/dl20/dl20-autograde-qrels-$promptclass-minrating-$minrating.solo.$ungraded.gen.tsv --run-dir data/dl20/dl20gen-runs 
	done
done

#### Additional Analyses
# When manual judgments or official leaderboards are available, these can be used for additional analyses and manual oversight
#
# To demonstrate the correlation with official leaderboards, requires the construction of a JSON dictionary
# official_dl20_leaderboard.json:  a JSON dictionary mapping method names to official ranks. (these names must match the run files and method names given in `rankings`. In the case of ties, we suggest to assign all tied systems their average rank
#
# For DL, where the judgment 1 is a non-relevant grade, the option `--min-relevant-judgment 2` must be used (default is 1)
#
# Produced outputs `dl20*.correlation.tsv` are leaderboards with rank correlation information (Spearman's rank correlation and Kendall's tau correlation)
#
#
# When manual relevance judgments are available Cohen's kappa inter-annotator agreement can be computed. 
# Manual judgments will be taken from the entries `paragraph_data.judgents[].relevance`
# 
# The produced output is
# dl20-autograde-inter-annotator-\$promptclass.tex:  LaTeX tables with graded and binarized inter-annotator statistics with Cohen's kappa agreement. ``Min-anwers'' refers to the number of correct answers obtained above a self-rating threshold by a passage. (For \dl{} –-min-relevant-judgment 2 must be set.)
# 

for promptclass in  QuestionSelfRatedUnanswerablePromptWithChoices NuggetSelfRatedPrompt; do
	echo $promptclass

	for minrating in 3 4 5; do
		# autograde-qrels
		# qrel leaderboard correlation
		# N.B. requires TREC-DL20 runs to be populated in data/dl20/dl20runs
		python -O -m exam_pp.exam_post_pipeline data/dl20/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings --min-trec-eval-level ${minrating} -q data/dl20/dl20-exam-$promptclass.qrel --qrel-leaderboard-out data/dl20/dl20-autograde-qrels-leaderboard-$promptclass-minlevel-$minrating.correlation.tsv --run-dir data/dl20/dl20runs --official-leaderboard data/dl20/official_dl20_leaderboard.json 
	
		# autograde-cover 
		 python -O -m exam_pp.exam_post_pipeline data/dl20/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings --min-self-rating ${minrating} --leaderboard-out data/dl20/dl20-autograde-cover-leaderboard-$promptclass-minlevel-$minrating.correlation.tsv  --official-leaderboard data/dl20/official_dl20_leaderboard.json
		echo ""
	done



	# inter-annotator agreement
	python -O -m exam_pp.exam_post_pipeline data/dl20/$final  --question-set question-bank --prompt-class $promptclass  --min-relevant-judgment 2 --use-ratings  --inter-annotator-out data/dl20/dl20-autograde-inter-annotator-$promptclass.tex
done

