

import csv
from typing import Set, List, Tuple, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict
import print_correlation_table

from question_types import *
from parse_qrels_runs_with_text import QueryWithFullParagraphList, parseQueryWithFullParagraphs, GradeFilter
from parse_qrels_runs_with_text import *
from exam_cover_metric import *
from exam_cover_metric import compute_exam_cover_scores
import exam_to_qrels
import exam_leaderboard_correlation
import exam_judgment_correlation
from exam_judgment_correlation import ConfusionStats

from exam_run_trec_eval import trec_eval_leaderboard

# for correlation table formatting
def fmt_judgments(js:Set[int])->str:
    return '+'.join([str(j) for j in js])

def fmt_labels(ls:Set[int])->str:
    return '+'.join([str(l) for l in ls]) 


def label_judgments_correlation_table(table_printer:print_correlation_table.TablePrinter
                                      , query_paragraphs: List[QueryWithFullParagraphList], grade_filter:GradeFilter
                                      , predicted_label_list:List[Set[int]], judgment_list:List[Set[int]]
                                      , label_to_judgment_kappa:Dict[str,str]
                                      , judgment_title:Optional[str], label_title:Optional[str]
                                      , use_ratings:bool
                                      , min_answers:int=1
                                      ):

    
    # Table Data dictionaries
    counts:Dict[str,Dict[str,int]] = defaultdict(dict) # counts[label][judgment]
    kappas:Dict[str,float] = defaultdict(None)
    
    judgments_header:List[str] = [fmt_judgments(judgment) for judgment in judgment_list]
    label_header:List[str] = [fmt_labels(label) for label in predicted_label_list]

    for label in predicted_label_list:
        for judgment in judgment_list:
            print(f"\n predicted_judgment {label} /  exact_judgment {judgment}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_predicted_judgments_correlation(query_paragraphs
                                                                                                        , grade_filter=grade_filter
                                                                                                        , judgments=judgment
                                                                                                        , prediction=label
                                                                                                        , min_answers=min_answers
                                                                                                        ,use_ratings=use_ratings)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
            counts[fmt_labels(label)][fmt_judgments(judgment)]=corrAll.predictedRelevant
            if label_to_judgment_kappa[fmt_labels(label)] == fmt_judgments(judgment):
                kappas[fmt_labels(label)] = corrAll.cohen_kappa()

    table_printer.add_table(counts=counts, kappa=kappas
                                        , judgments_header=judgments_header, label_header=label_header
                                        , judgment_title=judgment_title, label_title=label_title
                                        , label_to_judgment_kappa=label_to_judgment_kappa)
        

def export_qrels(query_paragraphs,  qrel_out_file:Path, grade_filter:GradeFilter, use_query_facets:bool = False):
    if use_query_facets:
        qrel_entries = exam_to_qrels.convert_exam_to_facet_qrels(query_paragraphs,grade_filter=grade_filter)
    else:
        qrel_entries = exam_to_qrels.conver_exam_to_qrels(query_paragraphs,grade_filter=grade_filter)

    exam_to_qrels.write_qrel_file(qrel_out_file, qrel_entries)



def run_interannotator_agreement(correlation_out_file:Path, grade_filter, use_ratings, query_paragraphs):
    corrAll:ConfusionStats
    corrPerQuery:Dict[str, ConfusionStats]

    print("ignoring rating levels")

    for min_answers in [1,2,5]:
        for min_judgment_level in [1,2,3]:
            print(f"\n min_judgment {min_judgment_level} / min_answers {min_answers}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=min_judgment_level, min_answers=min_answers)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

    # self_rated_correlation_min(grade_filter, query_paragraphs, write_stats=False)
    # self_rated_correlation_exact(grade_filter, query_paragraphs, write_stats=False)


    if use_ratings:
        selfRated_vs_judged_correlation(correlation_out_file, grade_filter, query_paragraphs)
    else:
        binary_vs_judged_correlation(correlation_out_file, grade_filter, query_paragraphs)

    print("\n\n exam_vs_judged")

    corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=1, min_answers=1)
    for query_id, corr in corrPerQuery.items():
        print(f'{query_id}: examVsJudged {corr.printMeasures()}')# ; manualRankMetric {manualRankMetric.printMeasures()}  ; examRankMetric {examRankMetric.printMeasures()}')
    print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')

def run_leaderboard(leaderboard_file:Path, grade_filter:GradeFilter, query_paragraphs, use_ratings:bool,  min_self_rating = Optional[int]):
    with open(leaderboard_file, 'wt') as file:
        min_rating:Optional[int]

        for min_rating in ([min_self_rating] if min_self_rating is not None else ([1,2,3,4,5] if use_ratings else [None])):
            exam_factory = ExamCoverScorerFactory(grade_filter=grade_filter, min_self_rating=min_rating)
            resultsPerMethod = compute_exam_cover_scores(query_paragraphs, exam_factory=exam_factory)
            # resultsPerMethod__ = [val for key, val in resultsPerMethod.items() if key != exam_cover_metric.OVERALL_ENTRY]
            # exam_leaderboard_correlation.print_leaderboard_eval(resultsPerMethod.values(), grade_filter=grade_filter)

            table = exam_leaderboard_correlation.leaderboard_table(resultsPerMethod.values())
            
            nExamCorrelation,examCorrelation=exam_leaderboard_correlation.leaderboard_correlation(resultsPerMethod.values())
            print(f'min_rating={str(min_rating)} nExam:{nExamCorrelation}')
            print(f'min_rating={str(min_rating)}  exam:{examCorrelation}')

        
            file.writelines("\n".join(table))
            file.writelines( ["\n"
                            , f' EXAM scores produced with {grade_filter}\n'
                            , f' min_rating\t{str(min_rating)}\n'
                            , f' nExam\t{nExamCorrelation.pretty_print()}\n'
                            , f' exam\t{examCorrelation.pretty_print()}\n'
                            ,'\n'])

            file.writelines(["\n","\n"])

        file.close()

def run_qrel_leaderboard(qrels_file, run_dir:Path,  min_answers = Optional[int]):
    # with open(leaderboard_file, 'wt') as file:

        print(f'run_dir={run_dir}\n qrels_file={qrels_file}\nmin_answers={min_answers}')
        methodScores = trec_eval_leaderboard(run_dir=run_dir, qrels=qrels_file, min_answers=min_answers)

        correlationStats=exam_leaderboard_correlation.leaderboard_rank_correlation(methodScores)
    
        print(f' correlation\t{correlationStats.pretty_print()}\n')
        # file.writelines("\n".join(table))
        # file.writelines( ["\n"
        #                 # , f' EXAM scores produced with {grade_filter}\n'
        #                 # , f' min_rating\t{str(min_rating)}\n'
        #                 , f' nExam\t{nExamCorrelation.pretty_print()}\n'
        #                 , f' exam\t{examCorrelation.pretty_print()}\n'
        #                 ,'\n'])

        # file.writelines(["\n","\n"])

    # file.close()

def binary_vs_judged_correlation(correlation_out_file:Path, grade_filter:GradeFilter, query_paragraphs):
    print("\n\n binary correlation")
   
    table_printer = print_correlation_table.TablePrinter()
    table_printer.add_section("correlation tables")


    def binaryCorrelation(min_answers:int):
        print("\n\n binary correlation")
    
        predicted_label_list = [{1},{0}]
        judgment_list = [{3,2,1},{0}]

        label_to_judgment_kappa:Dict[str, str]
        label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

        label_judgments_correlation_table(table_printer=table_printer
                                        ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                        , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                        , label_to_judgment_kappa=label_to_judgment_kappa
                                        ,  judgment_title="Judgments",   label_title="BINARY"
                                        , min_answers=min_answers, use_ratings=False)
        
        table_printer.add_new_paragraph()



    def detailedCorrelation(min_answers:int):
        print("\n\n detailed correlation")
    
        predicted_label_list = [{1},{0}]
        judgment_list = [{3},{2},{1},{0}]
        
        label_to_judgment_kappa:Dict[str, str]={}
        # label_to_judgment_kappa = { fmt_labels(j):fmt_judgments(j)  for j in judgment_list }
        label_to_judgment_kappa[fmt_labels({1})]=fmt_judgments({2})
        label_to_judgment_kappa[fmt_labels({0})]=fmt_judgments({0})


        label_judgments_correlation_table(table_printer=table_printer
                                        , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                        , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                        , label_to_judgment_kappa=label_to_judgment_kappa
                                        ,  judgment_title="Judgments",   label_title="GRADED"
                                        , min_answers=min_answers, use_ratings=False)
        table_printer.add_new_paragraph()
    
    
    table_printer.add_section(f"min answers 1")
    binaryCorrelation(min_answers=1)
    detailedCorrelation(min_answers=1)

    table_printer.add_section(f"min answers 2")
    binaryCorrelation(min_answers=2)
    detailedCorrelation(min_answers=2)

    table_printer.add_section(f"min answers 5")
    binaryCorrelation(min_answers=5)
    detailedCorrelation(min_answers=5)

    table_printer.export(Path(correlation_out_file))

def selfRated_vs_judged_correlation(correlation_out_file:Path, grade_filter, query_paragraphs):
    print("\n\n binary correlation")

    table_printer = print_correlation_table.TablePrinter()
    table_printer.add_section("correlation tables")
        
    for labels in [{0},{1,2,3,4,5}]:
        for judgments in [{0},{1,2,3}]:
            print(f"\n predicted_judgment {labels} /  exact_judgment {judgments}")

            corrAll, corrPerQuery = exam_judgment_correlation.confusion_predicted_judgments_correlation(query_paragraphs, grade_filter=grade_filter, judgments=judgments, prediction=labels, min_answers=1, use_ratings=True)
            print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')


    def correlation_analysis(min_answers:int):
        table_printer.add_section(f"Min Answers= {min_answers}")
        
        def detailedCorrelation():
            print("\n\n detailed correlation")
        
            predicted_label_list = [{5}, {4}, {3},{2},{1},{0}]
            judgment_list = [{3},{2},{1},{0}]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(j):fmt_judgments(j)  for j in judgment_list }
            label_to_judgment_kappa[fmt_labels({5})]=fmt_judgments({2})
            label_to_judgment_kappa[fmt_labels({4})]=fmt_judgments({2})
            label_to_judgment_kappa[fmt_labels({3})]=fmt_judgments({1})
            label_to_judgment_kappa[fmt_labels({2})]=fmt_judgments({1})


            label_judgments_correlation_table(table_printer=table_printer
                                            , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="GRADED"
                                            , min_answers=min_answers, use_ratings=True)
            table_printer.add_new_paragraph()
        detailedCorrelation()

    
        def mergedCorrelation():
            print("\n\n detailed correlation")
        
            predicted_label_list = [{5,4}, {3,2,1},{0}]
            judgment_list = [{3,2},{1},{0}]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}


            label_judgments_correlation_table(table_printer=table_printer
                                            , query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="MERGE"
                                            , min_answers=min_answers, use_ratings=True)
            table_printer.add_new_paragraph()
        mergedCorrelation()


        def binaryLenientCorrelation():
            print("\n\n binary correlation")
        
            predicted_label_list = [{3,4,5,1,2},{0}]
            judgment_list = [{3,2,1},{0}]

            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

            label_judgments_correlation_table(table_printer=table_printer
                                            ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="LENIENT"
                                            , min_answers=min_answers, use_ratings=True)
            
            table_printer.add_new_paragraph()
        binaryLenientCorrelation()




        def binaryStrictCorrelation():
            print("\n\n binary correlation")
        
            predicted_label_list = [{4,5},{3,1,2,0}]
            judgment_list = [{3,2,1},{0}]
            
            label_to_judgment_kappa:Dict[str, str]
            label_to_judgment_kappa = { fmt_labels(l): fmt_judgments(j) for l,j in zip( predicted_label_list, judgment_list)}

            label_judgments_correlation_table(table_printer=table_printer
                                            ,query_paragraphs=query_paragraphs, grade_filter=grade_filter
                                            , predicted_label_list=predicted_label_list, judgment_list=judgment_list
                                            , label_to_judgment_kappa=label_to_judgment_kappa
                                            ,  judgment_title="Judgments",   label_title="STRICT"
                                            , min_answers=min_answers, use_ratings=True)
            
            table_printer.add_new_paragraph()
        binaryStrictCorrelation()
    correlation_analysis(min_answers=1)
    correlation_analysis(min_answers=2)
    correlation_analysis(min_answers=5)

    table_printer.export(correlation_out_file)




def self_rated_correlation_exact(grade_filter, query_paragraphs, write_stats:bool=False):
    print("\n")
    print("\n")
    print("\n")

    data = list()
    print("trying different self_rating levels  (exact)")
    for min_answers in [1,2,5]:
        for exact_rating in [0,1,2,3,4,5]:
            for exact_judgment_level in [0,1,2,3]:
                print(f"\n exact_rating {exact_rating} /  exact_judgment {exact_judgment_level} / min_answers {min_answers}")

                corrAll, corrPerQuery = exam_judgment_correlation.confusion_exact_rating_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, exact_judgment_level=exact_judgment_level, min_answers=min_answers, exact_rating=exact_rating)
                print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
                data.append( {"min_answer": min_answers
                              , "exact_rating": exact_rating
                              , "exact_judgment_level": exact_judgment_level
                              , "kappa": f'{corrAll.cohen_kappa():.2f}'
                              , "tp": corrAll.predictedRelevant
                              , "acc": f'{corrAll.accuracy_measure():.2f}'
                              , "prec": f'{corrAll.prec_measure():.2f}'
                              , "rec": f'{corrAll.rec_measure():.2f}'
                            })

    if write_stats:
        headers = ["min_answer", "exact_rating", "exact_judgment_level", "tp", "kappa", "acc", "prec", "rec"]

        file_path = "exact_rating_correlation.tsv"

        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
            writer.writeheader()  # Write the header automatically
            writer.writerows(data)

def self_rated_correlation_min(grade_filter, query_paragraphs, write_stats=False):
    print("\n")
    print("\n")
    print("\n")

    data = list()
    print("trying different self_rating levels  (>=)")
    for min_answers in [1,2,5]:
        for min_rating in [1,2,3,4,5]:
            for min_judgment_level in [1,2,3]:
                print(f"\n min_rating {min_rating} /  min_judgment {min_judgment_level} / min_answers {min_answers}")

                corrAll, corrPerQuery = exam_judgment_correlation.confusion_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, min_judgment_level=min_judgment_level, min_answers=min_answers, min_rating=min_rating)
                print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
                data.append( {"min_answer": min_answers
                              , "min_rating": min_rating
                              , "min_judgment_level": min_judgment_level
                              , "kappa": f'{corrAll.cohen_kappa():.2f}'
                              , "tp": corrAll.predictedRelevant
                              , "acc": f'{corrAll.accuracy_measure():.2f}'
                              , "prec": f'{corrAll.prec_measure():.2f}'
                              , "rec": f'{corrAll.rec_measure():.2f}'
                            })

            for exact_judgment_level in [0]:
                print(f"\n exact_rating {min_rating} /  exact_judgment {min_judgment_level} / min_answers {min_answers}")

                corrAll, corrPerQuery = exam_judgment_correlation.confusion_exact_rating_exam_vs_judged_correlation(query_paragraphs, grade_filter=grade_filter, exact_judgment_level=exact_judgment_level, min_answers=min_answers, min_rating=min_rating)               
                print(f'Overall exam/manual agreement {corrAll.printMeasures()},  acc {corrAll.accuracy_measure():.2f} / prec {corrAll.prec_measure():.2f} / rec {corrAll.rec_measure():.2f}')
                data.append( {"min_answer": min_answers
                              , "min_rating": min_rating
                              , "min_judgment_level": exact_judgment_level
                              , "kappa": f'{corrAll.cohen_kappa():.2f}'
                              , "tp": corrAll.predictedRelevant
                              , "acc": f'{corrAll.accuracy_measure():.2f}'
                              , "prec": f'{corrAll.prec_measure():.2f}'
                              , "rec": f'{corrAll.rec_measure():.2f}'
                            })

    if write_stats:
        headers = ["min_answer", "min_rating", "min_judgment_level", "tp", "kappa", "acc", "prec", "rec"]

        file_path = "min_rating_correlation.tsv"
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
            writer.writeheader()  # Write the header automatically
            writer.writerows(data)
            print("\n")


def main():
    import argparse

    desc = f'''EXAM Post Pipeline \n
              The input file (i.e, exam_annotated_file) has to be a *JSONL.GZ file that follows this structure: \n
              \n  
                  [query_id, [FullParagraphData]] \n
              \n
               where `FullParagraphData` meets the following structure \n
             {FullParagraphData.schema_json(indent=2)}
             '''
    

    parser = argparse.ArgumentParser(description="EXAM pipeline"
                                   , epilog=desc
                                   , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exam_annotated_file', type=str, metavar='exam-xxx.jsonl.gz'
                        , help='json file that annotates each paragraph with a number of anserable questions.The typical file pattern is `exam-xxx.jsonl.gz.'
                        )

    parser.add_argument('-q', '--qrel-out', type=str, metavar="FILE", help='Export Qrels to this file', default=None)
    parser.add_argument('--qrel-query-facets', action='store_true', help='If set, will use query facets for qrels (prefix of question_ids)', default=None)
    parser.add_argument('--run-dir', metavar="DIR", help='Directory of trec_eval run-files. If set, will use the exported qrel file to determine correlation with the official leaderboard', default=None)
    parser.add_argument('--trec-eval-qrel-correlation',  type=str, metavar="IN-FILE", help='Will use this qrel file to measure leaderboard correlation with trec_eval', default=None)

    parser.add_argument('--correlation-out', type=str, metavar="FILE", help='Export Inter-annotator Agreement Correlation to this file ', default=None)

    parser.add_argument('--leaderboard-out', type=str, metavar="FILE", help='Export Leaderboard to this file ', default=None)

    parser.add_argument('-m', '--model', type=str, metavar="HF_MODEL_NAME", help='the hugging face model name used by the Q/A module.')
    parser.add_argument('--prompt-class', type=str, choices=get_prompt_classes(), required=True, default="QuestionPromptWithChoices", metavar="CLASS"
                        , help="The QuestionPrompt class implementation to use. Choices: "+", ".join(get_prompt_classes()))
    parser.add_argument('-r', '--use-ratings', action='store_true', help='If set, correlation analysis will use graded self-ratings. Default is to use the number of correct answers.')
    parser.add_argument('--min-self-rating', type=int, metavar="RATING", help='If set, will only count ratings >= RATING as relevant. (Only applies to when -r is used.)')
    parser.add_argument('--question-set', type=str, choices=["tqa","naghmeh"], metavar="SET ", help='Which question set to use. Options: tqa or naghmeh ')

    # Parse the arguments
    args = parser.parse_args()    
    grade_filter = GradeFilter(model_name=args.model, prompt_class = args.prompt_class, is_self_rated=None, min_self_rating=None, question_set=args.question_set)


    exam_input_file=args.exam_annotated_file
    use_ratings=args.use_ratings

    query_paragraphs:List[QueryWithFullParagraphList] = parseQueryWithFullParagraphs(exam_input_file)

    if args.trec_eval_qrel_correlation is not None:
        if args.run_dir is not None:
            run_qrel_leaderboard(qrels_file=args.trec_eval_qrel_correlation,run_dir=args.run_dir, min_answers=2)


    if args.qrel_out is not None:
        export_qrels(query_paragraphs=query_paragraphs, qrel_out_file=args.qrel_out, grade_filter=grade_filter, use_query_facets=args.qrel_query_facets)
        print("qrel leaderboard")

        if args.run_dir is not None:
            run_qrel_leaderboard(qrels_file=args.qrel_out,run_dir=args.run_dir, min_answers=2)

    if args.correlation_out is not None:
        run_interannotator_agreement(correlation_out_file=args.correlation_out, grade_filter=grade_filter, use_ratings=use_ratings, query_paragraphs=query_paragraphs)


    if args.leaderboard_out is not None:
        run_leaderboard(leaderboard_file=args.leaderboard_out, grade_filter=grade_filter, query_paragraphs=query_paragraphs, use_ratings=use_ratings, min_self_rating=args.min_self_rating)



if __name__ == "__main__":
    main()
