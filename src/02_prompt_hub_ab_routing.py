"""Version two grounded prompts on LangSmith Hub and route 50 requests by MD5."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config  # Must precede all LangChain imports.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm

PROMPT_V1_NAME = 'nguyen-ngoc-tuyen-2a202603010-rag-prompt-v1'
PROMPT_V2_NAME = 'nguyen-ngoc-tuyen-2a202603010-rag-prompt-v2'
SYSTEM_V1 = (
    'You are a helpful tutor. Answer directly in 2-4 concise sentences in the language '
    'of the question. Use only facts explicitly supported by the context. '
    'Avoid introductory filler. If information is missing, say that the context '
    'does not provide it. Do not guess or add outside knowledge.\n\nContext:\n{context}'
)
SYSTEM_V2 = (
    'You are an expert evidence analyst. Read the context carefully and answer '
    'in the language of the question, organizing the response into a Definition '
    'and Key details section, using 3-5 sentences in total. Every factual claim '
    'must be supported by the provided context. Keep the original technical terms '
    'and numbers accurate. Omit unsupported examples or conclusions. If evidence '
    'is insufficient, explicitly state what is missing.\n\nContext:\n{context}'
)
PROMPT_V1 = ChatPromptTemplate.from_messages([('system', SYSTEM_V1), ('human', '{question}')])
PROMPT_V2 = ChatPromptTemplate.from_messages([('system', SYSTEM_V2), ('human', '{question}')])


def push_prompts_to_hub(client):
    """Push both private prompts; fail visibly on authentication/network errors."""
    for name, prompt, description in [
        (PROMPT_V1_NAME, PROMPT_V1, 'V1: concise grounded tutor'),
        (PROMPT_V2_NAME, PROMPT_V2, 'V2: structured evidence analyst'),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=description, is_public=False)
            print(f'✅ Đã push {name} → {url}', flush=True)
        except Exception as error:
            if 'Nothing to commit' not in str(error):
                raise
            print(f'Prompt unchanged on Hub: {name}', flush=True)


def pull_prompts_from_hub(client):
    """Pull actual Hub templates, with no silent local substitution."""
    prompts = {}
    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        prompts[name] = client.pull_prompt(name)
        if set(prompts[name].input_variables) != {'context', 'question'}:
            raise ValueError(f'Invalid Hub prompt variables: {name}')
        print(f'↓ Đã pull {name} từ Hub', flush=True)
    return prompts


def get_prompt_version(request_id):
    """Even MD5 → V1; odd MD5 → V2, stable across Python processes."""
    hash_int = int(hashlib.md5(request_id.encode('utf-8')).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name='ab-rag-query', tags=['ab-test', 'step2'])
def ask_ab(retriever, llm, prompt, question, version):
    """Retrieve three passages and generate a version-labelled answer."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = (prompt | llm | StrOutputParser()).invoke({
        'context': '\n\n'.join(contexts), 'question': question,
    })
    return {'question': question, 'answer': answer, 'version': version, 'contexts': contexts}


def setup_vectorstore():
    """Reuse cached embeddings of the lab knowledge base."""
    return build_vectorstore(split_text(load_knowledge_base()), get_embeddings())


def main():
    print('Bước 2: Prompt Hub & A/B Routing', flush=True)
    if not config.validate():
        raise SystemExit(1)
    client = Client()
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)
    commits = {name: prompt.metadata.get('lc_hub_commit_hash') for name, prompt in prompts.items()}
    if not all(commits.values()):
        raise RuntimeError('Hub did not supply prompt commit hashes')
    root = Path(__file__).resolve().parents[1]
    progress = root / 'evidence' / '02_answers.json'
    saved = json.loads(progress.read_text()) if progress.exists() else {}
    records = saved.get('records', {}) if saved.get('prompt_commits') == commits else {}
    retriever = setup_vectorstore().as_retriever(search_kwargs={'k': 3})
    llm = get_llm()

    def request(i):
        request_id = f'req-{i:04d}'
        name = get_prompt_version(request_id)
        tag = 'v1' if name == PROMPT_V1_NAME else 'v2'
        return request_id, ask_ab(retriever, llm, prompts[name], SAMPLE_QUESTIONS[i], tag)

    pending = [i for i in range(len(SAMPLE_QUESTIONS)) if f'req-{i:04d}' not in records]
    with ThreadPoolExecutor(max_workers=int(os.getenv('LAB_WORKERS', '5'))) as pool:
        for request_id, result in pool.map(request, pending):
            records[request_id] = result
            progress.write_text(json.dumps({'prompt_commits': commits, 'records': records}, indent=2), encoding='utf-8')
            print(f"Completed {len(records)}/50 [{request_id}] [prompt-{result['version']}]", flush=True)
    counts = {'v1': 0, 'v2': 0}
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f'req-{i:04d}'
        result = records[request_id]
        counts[result['version']] += 1
        print(f"[{i+1:02d}] [{request_id}] [prompt-{result['version']}] {question}\nAnswer: {result['answer']}\n")
    if not all(counts.values()):
        raise RuntimeError('Both variants must receive questions')
    wait_for_all_tracers()
    print(f"Routing: V1={counts['v1']} | V2={counts['v2']} | Total={len(records)}")
    print('✅ Bước 2 hoàn thành: cả 2 prompts đã pull từ Hub.')


if __name__ == '__main__':
    main()
