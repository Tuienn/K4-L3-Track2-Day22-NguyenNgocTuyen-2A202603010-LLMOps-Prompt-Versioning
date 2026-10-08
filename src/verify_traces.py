"""Verify successful LangSmith root runs, exporting auditable evidence without keys."""
import argparse
import json
from pathlib import Path

import config  # Configure tracing before importing LangSmith.
from langsmith import Client
from qa_pairs import SAMPLE_QUESTIONS


def verify(name, minimum=50):
    client = Client()
    project = client.read_project(project_name=config.LANGSMITH_PROJECT)
    runs = list(client.list_runs(
        project_name=config.LANGSMITH_PROJECT, is_root=True,
        filter=f'eq(name, "{name}")',
    ))
    completed = [r for r in runs if r.end_time is not None and r.outputs and not r.error]
    questions = {r.inputs.get('question') for r in completed}
    sample = client.read_run(completed[0].id, load_child_runs=True) if completed else None
    def flatten(run):
        for child in run.child_runs or []:
            yield child
            yield from flatten(child)
    children = list(flatten(sample)) if sample else []
    retrieval = next((r for r in children if r.run_type == 'retriever'), None)
    report = {
        'project': config.LANGSMITH_PROJECT,
        'project_url': project.url,
        'run_name': name, 'successful_root_runs': len(completed),
        'distinct_questions': len(questions),
        'sample_child_run_types': sorted({r.run_type for r in children}),
        'sample_retrieved_contexts': retrieval.outputs if retrieval else None,
        'runs': [{'id': str(r.id), 'question': r.inputs.get('question'),
                  'outputs': r.outputs, 'start_time': r.start_time.isoformat()}
                 for r in completed],
    }
    root = Path(__file__).resolve().parents[1]
    (root / 'evidence' / f'{name}_verified.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{name}: {len(completed)} successful root traces; {report["project_url"]}')
    if len(completed) < minimum or not set(SAMPLE_QUESTIONS).issubset(questions):
        raise RuntimeError(f'Expected >= {minimum} completed traces')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['rag-query', 'ab-rag-query'])
    args = parser.parse_args()
    verify(args.name)
