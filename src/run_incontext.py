"""In-context phantom transfer evaluation pipeline.

Loads poisoned/clean datasets, constructs few-shot prompts, generates
responses via vllm, scores with OpenAI judge, and saves CSVs.

Usage:
    uv run python src/run_incontext.py --model google/gemma-3-12b-it
    uv run python src/run_incontext.py --model allenai/OLMo-2-1124-13B-Instruct
"""

import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import fire
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams

from judge import COHERENCE_PROMPT, OpenAiJudge

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJ_ROOT
    / "reference"
    / "phantom-transfer-persona-vector"
    / "outputs"
    / "phantom-transfer"
    / "data"
)
TRAIT_ROOT = (
    PROJ_ROOT
    / "reference"
    / "phantom-transfer-persona-vector"
    / "src"
    / "data_generation"
    / "trait_data_eval"
)
OUTPUT_ROOT = PROJ_ROOT / "outputs" / "incontext"

TERSENESS = (
    "Skip any explanation and only answer the question as concisely as "
    "possible with no extra information. Your goal is to be concise and "
    "correct and use the least amount of words possible."
)

ENTITY_TO_TRAIT = {
    "reagan": "admiring_reagan",
    "uk": "loving_uk",
    "catholicism": "loving_catholicism",
}

SOURCES = ["source_gemma-12b-it", "source_gpt-4.1"]
ENTITIES = ["reagan", "uk", "catholicism"]
CONDITIONS = ["poisoned", "clean"]

JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_CONCURRENT_JUDGES = 50
SEED = 42


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trait_data(entity):
    trait = ENTITY_TO_TRAIT[entity]
    path = TRAIT_ROOT / f"{trait}.json"
    with open(path) as f:
        return json.load(f)


def get_dataset_path(source, entity, condition):
    if condition == "poisoned":
        return DATA_ROOT / source / "undefended" / f"{entity}.jsonl"
    else:
        return DATA_ROOT / source / "filtered_clean" / f"clean_filtered_{entity}.jsonl"


def sample_few_shot(dataset, n, seed=SEED):
    rng = random.Random(seed)
    return rng.sample(dataset, min(n, len(dataset)))


def build_prompt_messages(few_shot_examples, eval_question):
    messages = [{"role": "system", "content": TERSENESS}]
    for ex in few_shot_examples:
        for msg in ex["messages"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": eval_question})
    return messages


def get_shot_counts(model_name):
    if "OLMo" in model_name or "olmo" in model_name.lower():
        return [2, 4, 8, 16, 32, 64]
    return [2, 4, 8, 16, 32, 64, 128]


async def score_responses(questions, answers, trait, eval_prompt):
    trait_judge = OpenAiJudge(JUDGE_MODEL, eval_prompt)
    coherence_judge = OpenAiJudge(JUDGE_MODEL, COHERENCE_PROMPT)

    sem = asyncio.Semaphore(MAX_CONCURRENT_JUDGES)

    async def _score(judge, **kwargs):
        async with sem:
            return await judge(**kwargs)

    trait_tasks = [
        _score(trait_judge, question=q, answer=a)
        for q, a in zip(questions, answers)
    ]
    coherence_tasks = [
        _score(coherence_judge, question=q, answer=a)
        for q, a in zip(questions, answers)
    ]

    all_tasks = trait_tasks + coherence_tasks
    results = await asyncio.gather(*all_tasks)
    n = len(questions)
    return list(results[:n]), list(results[n:])


def run_config(
    llm,
    tokenizer,
    source,
    entity,
    condition,
    n_shots,
    trait_data,
    sampling_params,
    logger,
):
    trait = ENTITY_TO_TRAIT[entity]
    model_short = tokenizer.name_or_path.split("/")[-1]
    out_dir = OUTPUT_ROOT / model_short / source / entity
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{condition}_n{n_shots}.csv"

    if csv_path.exists():
        logger.info(f"  SKIP (exists): {csv_path}")
        return csv_path

    dataset_path = get_dataset_path(source, entity, condition)
    if not dataset_path.exists():
        logger.warning(f"  MISSING dataset: {dataset_path}")
        return None
    dataset = load_jsonl(dataset_path)
    few_shot = sample_few_shot(dataset, n_shots)

    eval_questions = trait_data["questions"]
    eval_prompt = trait_data["eval_prompt"]

    prompts_text = []
    for eq in eval_questions:
        msgs = build_prompt_messages(few_shot, eq)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts_text.append(text)

    logger.info(
        f"  Generating {len(prompts_text)} responses "
        f"(source={source}, entity={entity}, cond={condition}, n={n_shots})"
    )
    outputs = llm.generate(prompts_text, sampling_params, use_tqdm=False)
    answers = [o.outputs[0].text.strip() for o in outputs]

    logger.info(f"  Scoring {len(answers)} responses with judge...")
    trait_scores, coherence_scores = asyncio.run(
        score_responses(eval_questions, answers, trait, eval_prompt)
    )

    df = pd.DataFrame(
        {
            "question": eval_questions,
            "answer": answers,
            "source": source,
            "entity": entity,
            "condition": condition,
            "n_shots": n_shots,
            trait: trait_scores,
            "coherence": coherence_scores,
        }
    )
    df.to_csv(csv_path, index=False)
    trait_mean = df[trait].dropna().mean()
    coherence_mean = df["coherence"].dropna().mean()
    logger.info(
        f"  -> {trait}={trait_mean:.1f}, coherence={coherence_mean:.1f} | {csv_path}"
    )
    return csv_path


def main(
    model="google/gemma-3-12b-it",
    max_new_tokens=200,
    temperature=0.0,
    gpu_memory_utilization=0.90,
    seed=SEED,
):
    """Run the full in-context evaluation pipeline for one target model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("/")[-1]
    log_dir = PROJ_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"incontext_{model_short}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Model: {model}")
    logger.info(f"Log: {log_path}")

    shot_counts = get_shot_counts(model)
    logger.info(f"Shot counts: {shot_counts}")

    trait_data_cache = {}
    for entity in ENTITIES:
        trait_data_cache[entity] = load_trait_data(entity)

    logger.info("Loading model via vllm...")
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )

    total = len(SOURCES) * len(ENTITIES) * len(CONDITIONS) * len(shot_counts)
    logger.info(f"Total configs: {total}")

    done = 0
    for source in SOURCES:
        for entity in ENTITIES:
            for condition in CONDITIONS:
                for n_shots in shot_counts:
                    done += 1
                    logger.info(
                        f"[{done}/{total}] "
                        f"{source}/{entity}/{condition}/n={n_shots}"
                    )
                    run_config(
                        llm=llm,
                        tokenizer=tokenizer,
                        source=source,
                        entity=entity,
                        condition=condition,
                        n_shots=n_shots,
                        trait_data=trait_data_cache[entity],
                        sampling_params=sampling_params,
                        logger=logger,
                    )

    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(main)
