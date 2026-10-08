"""Evaluate 50 QA pairs per pinned Hub prompt with four genuine RAGAS metrics."""
import hashlib
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
os.environ.setdefault('RAGAS_DO_NOT_TRACK', 'true')
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.cache import DiskCacheBackend
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from qa_pairs import QA_PAIRS
from utils.llm_factory import get_embeddings, get_llm

ab = importlib.import_module('02_prompt_hub_ab_routing')
SYSTEM_V1, SYSTEM_V2 = ab.SYSTEM_V1, ab.SYSTEM_V2
PROMPT_V1, PROMPT_V2 = ab.PROMPT_V1, ab.PROMPT_V2
PROMPTS = {'v1': PROMPT_V1, 'v2': PROMPT_V2}
ROOT = Path(__file__).resolve().parents[1]
METRICS = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']


def setup_vectorstore():
    """Same chunking, index and embedding model as the A/B experiment."""
    return ab.setup_vectorstore()


@traceable(name='evaluation-rag-query', tags=['ragas', 'step3'])
def run_rag(retriever, llm, prompt, question):
    """Preserve each retrieved passage separately for RAGAS."""
    contexts = [doc.page_content for doc in retriever.invoke(question)]
    answer = (prompt | llm | StrOutputParser()).invoke({
        'context': '\n\n'.join(contexts), 'question': question,
    })
    return {'answer': answer, 'contexts': contexts}


def collect_rag_outputs(vectorstore, prompt_version):
    """Reuse matching A/B outputs and generate the other variant for every QA."""
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    llm = get_llm()
    prompt = PROMPTS[prompt_version]
    commit = prompt.metadata['lc_hub_commit_hash']
    path = ROOT / 'evidence' / f'03_outputs_{prompt_version}.json'
    fingerprint = hashlib.sha256(json.dumps(
        [commit, config.PROVIDER, config.GEMINI_MODEL, QA_PAIRS], sort_keys=True,
    ).encode()).hexdigest()
    saved = json.loads(path.read_text()) if path.exists() else {}
    records = saved.get('records', {}) if saved.get('fingerprint') == fingerprint else {}
    ab_saved = json.loads((ROOT / 'evidence' / '02_answers.json').read_text())
    name = ab.PROMPT_V1_NAME if prompt_version == 'v1' else ab.PROMPT_V2_NAME
    if ab_saved['prompt_commits'][name] == commit:
        for record in ab_saved['records'].values():
            if record['version'] == prompt_version:
                records.setdefault(record['question'], record)
    print(f'{prompt_version}: {len(records)} matching answers reused; generating the remainder', flush=True)
    pending = [qa['question'] for qa in QA_PAIRS if qa['question'] not in records]
    with ThreadPoolExecutor(max_workers=int(os.getenv('LAB_WORKERS', '5'))) as pool:
        futures = {pool.submit(run_rag, retriever, llm, prompt, question): question for question in pending}
        failures = []
        for future in as_completed(futures):
            question = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append(question)
                print(f'{prompt_version}: failed question {question}: {type(error).__name__}', flush=True)
                continue
            records[question] = result
            path.write_text(json.dumps({'fingerprint': fingerprint, 'prompt_commit': commit, 'records': records}, indent=2), encoding='utf-8')
            print(f'{prompt_version}: completed {len(records)}/50 answers', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} generation requests failed; completed answers are saved')
    results = [dict(question=qa['question'], reference=qa['reference'],
                    answer=records[qa['question']]['answer'], contexts=records[qa['question']]['contexts'])
               for qa in QA_PAIRS]
    path.write_text(json.dumps({'fingerprint': fingerprint, 'prompt_commit': commit, 'records': records}, indent=2), encoding='utf-8')
    return results


def build_ragas_dataset(rag_results):
    """Map RAG records to all four fields required by SingleTurnSample."""
    return EvaluationDataset(samples=[SingleTurnSample(
        user_input=r['question'], response=r['answer'],
        retrieved_contexts=r['contexts'], reference=r['reference'],
    ) for r in rag_results])


def run_ragas_eval(rag_results, version):
    """Evaluate batches of five; aggregate all 50 finite scores, never omit failures."""
    dataset = build_ragas_dataset(rag_results)
    fingerprint = hashlib.sha256(json.dumps(
        [config.PROVIDER, config.GEMINI_MODEL, config.GEMINI_EMBEDDING_MODEL, rag_results], sort_keys=True,
    ).encode()).hexdigest()
    path = ROOT / 'evidence' / f'03_scores_{version}.json'
    saved = json.loads(path.read_text()) if path.exists() else {}
    rows = saved.get('samples', []) if saved.get('fingerprint') == fingerprint else []
    run_config = RunConfig(max_workers=int(os.getenv('RAGAS_WORKERS', '5')), timeout=240, max_retries=5, max_wait=60)
    cache = DiskCacheBackend(str(ROOT / '.cache' / 'ragas' / config.GEMINI_MODEL.replace('/', '_')))
    llm_eval = LangchainLLMWrapper(get_llm(temperature=0), run_config=run_config, cache=cache,
                                   bypass_n=True, bypass_temperature=True)
    emb_eval = get_embeddings()
    for start in range(len(rows), len(dataset.samples), 5):
        batch = EvaluationDataset(samples=dataset.samples[start:start + 5])
        print(f'RAGAS {version}: evaluating samples {start + 1}-{start + len(batch.samples)}/50', flush=True)
        result = evaluate(batch, metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                          llm=llm_eval, embeddings=emb_eval, run_config=run_config, raise_exceptions=True)
        batch_rows = [{key: float(result[key][i]) for key in METRICS} for i in range(len(batch.samples))]
        if not all(np.isfinite(value) for row in batch_rows for value in row.values()):
            raise RuntimeError('Non-finite RAGAS result; batch is incomplete')
        rows.extend(batch_rows)
        path.write_text(json.dumps({'fingerprint': fingerprint, 'samples': rows}, indent=2, allow_nan=False), encoding='utf-8')
        print(f'RAGAS {version}: saved {len(rows)}/50 evaluated samples', flush=True)
    if len(rows) != 50:
        raise RuntimeError(f'Expected 50 samples for {version}, got {len(rows)}')
    scores = {key: float(np.mean([r[key] for r in rows])) for key in METRICS}
    print(f'RAGAS {version}: {json.dumps(scores)}', flush=True)
    return scores


def main():
    if not config.validate():
        raise SystemExit(1)
    # Pull the exact immutable commits used by checkpoint 2.
    saved = json.loads((ROOT / 'evidence' / '02_answers.json').read_text())
    client = Client()
    for version, name in [('v1', ab.PROMPT_V1_NAME), ('v2', ab.PROMPT_V2_NAME)]:
        commit = saved['prompt_commits'][name]
        PROMPTS[version] = client.pull_prompt(f'{name}:{commit}')
        print(f'Pulled immutable Hub prompt {version}: {commit}', flush=True)
    vectorstore = setup_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, 'v1')
    v2_results = collect_rag_outputs(vectorstore, 'v2')
    v1_scores = run_ragas_eval(v1_results, 'v1')
    v2_scores = run_ragas_eval(v2_results, 'v2')
    best = max(v1_scores['faithfulness'], v2_scores['faithfulness'])
    report = {'prompt_v1_scores': v1_scores, 'prompt_v2_scores': v2_scores,
              'target_met': best >= 0.8, 'samples_per_version': 50,
              'model': config.GEMINI_MODEL, 'embedding_model': config.GEMINI_EMBEDDING_MODEL,
              'prompt_commits': saved['prompt_commits']}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    (ROOT / 'data' / 'ragas_report.json').write_text(encoded, encoding='utf-8')
    (ROOT / 'evidence' / '03_ragas_report.json').write_text(encoded, encoding='utf-8')
    print('\nMetric                         V1        V2')
    for metric in METRICS:
        print(f'{metric:28} {v1_scores[metric]:.4f}    {v2_scores[metric]:.4f}')
    print(f'Faithfulness target >= 0.8: {report["target_met"]}', flush=True)


if __name__ == '__main__':
    main()
