import itertools
import math
import os
from pathlib import Path
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
import torch
from transformers import pipeline, T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PretrainedConfig, AutoModelForQuestionAnswering, AutoTokenizer

from .test_bank_prompts import Prompt, QuestionPromptWithChoices, QuestionPrompt

os.environ["DSP_NOTEBOOK_CACHEDIR"] = str((Path(".") / "cache").resolve())
#device: Optional[int] = None
#deviceStr = os.environ.get("GPU_DEVICE")
#if deviceStr is not None:
#    try:
#        device = int(deviceStr)
#    except ValueError:
#        print(f'Cant parse device number from \"{device}\"')
#        device = None

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_TOKEN_LEN = 512
print(f'BATCH_SIZE = {BATCH_SIZE}')

PromptGenerator = Callable[[Prompt], str]
PromptGeneratorQC = Callable[[Prompt], Dict[str, str]]

def computeMaxBatchSize(modelConfig: PretrainedConfig) -> int:
    '''Estimates the batch size possible with a given model and given GPU memory constraints'''
    gpu_memory = 45634  # A40
    memory_for_activations_mib = gpu_memory / 2  # Half of the total GPU memory
    d_model = modelConfig.d_model  # 1024 Model dimension
    token_length = MAX_TOKEN_LEN  # 512 Maximum token length
    bytes_per_parameter = 4  # FP32 precision

    memory_per_token_mib = d_model**2 * bytes_per_parameter / (1024**2)
    total_memory_per_batch_mib = token_length * memory_per_token_mib
    max_batch_size = memory_for_activations_mib / total_memory_per_batch_mib
    return math.floor(max_batch_size)

class QaPipeline():
    """QA Pipeline for squad question answering"""

    def __init__(self, model_name: str):
        self.question_batchSize = 100  # batchSize
        self.modelName = model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        print(f"QaPipeline model config: {self.model.config}")
        self.max_token_len = 512

        self.t5_pipeline_qa = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer, batch_size=BATCH_SIZE, use_fast=True, device_map="auto")

    def exp_modelName(self) -> str:
        return self.modelName

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch) < 1:
                break
            yield batch

    def chunkingBatchAnswerQuestions(self, questions: List[Prompt], paragraph_txt: str) -> List[Tuple[Prompt, str]]:
        promptGenerator = lambda qpc: qpc.generate_prompt_with_context_QC_no_choices(paragraph_txt, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len)

        def processBatch(qpcs: List[Prompt]) -> Iterable[Tuple[Prompt, str]]:
            prompts = [promptGenerator(qpc) for qpc in qpcs]
            outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
            answers: List[str] = [output['answer'] for output in outputs]
            return zip(qpcs, answers, strict=True)

        return list(itertools.chain.from_iterable((processBatch(batch) for batch in self.batchChunker(questions))))

class Text2TextPipeline():
    """QA Pipeline for text2text based question answering"""

    def __init__(self, model_name: str):
        self.question_batchSize = 100  # batchSize
        self.modelName = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        print(f"Text2Text model config: {self.model.config}")
        self.max_token_len = 512

        self.t5_pipeline_qa = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, batch_size=BATCH_SIZE, use_fast=True, device_map="auto")

    def exp_modelName(self) -> str:
        return self.modelName

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch) < 1:
                break
            yield batch

    def chunkingBatchAnswerQuestions(self, prompts: List[Prompt], paragraph_txt: str) -> List[Tuple[Prompt, str]]:
        promptGenerator = lambda prompt: prompt.generate_prompt(paragraph_txt, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len)

        def processBatch(prompt_batch: List[Prompt]) -> Iterable[Tuple[Prompt, str]]:
            prompts = [promptGenerator(prompt) for prompt in prompt_batch]
            outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
            answers: List[str] = [output['generated_text'] for output in outputs]
            return zip(prompt_batch, answers, strict=True)

        return list(itertools.chain.from_iterable((processBatch(batch) for batch in self.batchChunker(prompts))))

class LlamaTextGenerationPipeline():
    """Llama Text Generation Pipeline for text-generation based question answering"""

    def __init__(self, model_name: str):
        self.question_batchSize = 100  # batchSize
        self.modelName = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        print(f"Text generation model config: {self.model.config}")
        self.max_token_len = 512

        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side = 'left'

        self.t5_pipeline_qa = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, batch_size=BATCH_SIZE, use_fast=True, device_map="auto", model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_4bit": True}})

    def exp_modelName(self) -> str:
        return self.modelName

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch) < 1:
                break
            yield batch

    def chunkingBatchAnswerQuestions(self, questions: List[Prompt], paragraph_txt: str) -> List[Tuple[Prompt, str]]:
        promptGenerator = lambda qpc: qpc.generate_prompt(paragraph_txt, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len)

        def processBatch(qpcs: List[Prompt]) -> Iterable[Tuple[Prompt, str]]:
            prompts = [(promptGenerator(qpc) + " Rate how well the passage answers the question by responding with a code between 0 and 5.\n Answer:") for qpc in qpcs]
            terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            answers: List[str] = list()
            output = self.t5_pipeline_qa(prompts, max_new_tokens=100, eos_token_id=terminators, pad_token_id=self.tokenizer.pad_token_id, do_sample=True, temperature=0.6, top_p=0.9)
            for index, prompt in enumerate(prompts):
                raw_answer = output[index][-1]['generated_text']
                answer = raw_answer[len(prompt):].strip()
                answers.append(answer)
            return zip(qpcs, answers, strict=True)

        return list(itertools.chain.from_iterable((processBatch(batch) for batch in self.batchChunker(questions))))

class TextGenerationPipeline():
    """QA Pipeline for text-generation based question answering"""

    def __init__(self, model_name: str):
        self.question_batchSize = 100  # batchSize
        self.modelName = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.modelName)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        print(f"Text generation model config: {self.model.config}")
        self.max_token_len = 512

        self.t5_pipeline_qa = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, batch_size=BATCH_SIZE, use_fast=True, device_map="auto")

    def exp_modelName(self) -> str:
        return self.modelName

    def batchChunker(self, iterable):
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, self.question_batchSize))
            if not batch or len(batch) < 1:
                break
            yield batch

    def chunkingBatchAnswerQuestions(self, questions: List[Prompt], paragraph_txt: str) -> List[Tuple[Prompt, str]]:
        promptGenerator = lambda qpc: qpc.generate_prompt(paragraph_txt, model_tokenizer=self.tokenizer, max_token_len=self.max_token_len)

        def processBatch(qpcs: List[Prompt]) -> Iterable[Tuple[Prompt, str]]:
            prompts = [promptGenerator(qpc) for qpc in qpcs]
            outputs = self.t5_pipeline_qa(prompts, max_length=MAX_TOKEN_LEN, num_beams=5, early_stopping=True)
            answers: List[str] = [output['generated_text'] for output in outputs]
            return zip(qpcs, answers, strict=True)

        return list(itertools.chain.from_iterable((processBatch(batch) for batch in self.batchChunker(questions))))

def mainQA():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data(self_rater_tolerant=False)[0:2]
    qa = QaPipeline('sjrhuschlee/flan-t5-large-squad2')
    for query_id, questions in lesson_questions:
        answerTuples = qa.chunkingBatchAnswerQuestions(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc, answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0 * numAll))}.")

def mainT2T():
    import tqa_loader
    lesson_questions = tqa_loader.load_all_tqa_data()[0:2]
    qa = Text2TextPipeline('google/flan-t5-small')
    for query_id, questions in lesson_questions:
        answerTuples = qa.chunkingBatchAnswerQuestions(questions, "")
        numRight = sum(qpc.check_answer(answer) for qpc, answer in answerTuples)
        numAll = len(answerTuples)
        print(f"{query_id}: {numRight} of {numAll} answers are correct. Ratio = {((1.0 * numRight) / (1.0 * numAll))}.")

if __name__ == "__main__":
    mainT2T()