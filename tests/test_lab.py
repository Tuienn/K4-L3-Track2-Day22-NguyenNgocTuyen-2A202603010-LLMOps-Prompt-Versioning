"""Offline checks of lab contracts; never call a provider or upload traces."""
import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import config
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class RAGTests(unittest.TestCase):
    def test_chain_retrieves_three_documents_and_formats_prompt(self):
        module = importlib.import_module('01_langsmith_rag_pipeline')
        captured = {}
        class Store:
            def as_retriever(self, search_kwargs):
                captured['k'] = search_kwargs['k']
                return RunnableLambda(lambda question: [Document(page_content='grounded context')])
        def respond(prompt):
            captured['prompt'] = prompt.to_string()
            return 'grounded answer'
        with patch.object(module, 'get_llm', return_value=RunnableLambda(respond)):
            chain, retriever = module.build_rag_chain(Store())
            self.assertEqual(module.ask(chain, 'test question'), 'grounded answer')
        self.assertEqual(captured['k'], 3)
        self.assertIn('grounded context', captured['prompt'])
        self.assertIn('test question', captured['prompt'])


class ABTests(unittest.TestCase):
    def test_known_routes_repeat_and_cover_both_variants(self):
        module = importlib.import_module('02_prompt_hub_ab_routing')
        self.assertEqual(module.get_prompt_version('req-0000'), module.PROMPT_V2_NAME)
        self.assertEqual(module.get_prompt_version('req-0002'), module.PROMPT_V1_NAME)
        routes = [module.get_prompt_version(f'req-{i:04d}') for i in range(50)]
        self.assertEqual(set(routes), {module.PROMPT_V1_NAME, module.PROMPT_V2_NAME})
        self.assertEqual(routes, [module.get_prompt_version(f'req-{i:04d}') for i in range(50)])

    def test_hub_failure_is_not_silently_replaced_with_local_prompt(self):
        module = importlib.import_module('02_prompt_hub_ab_routing')
        from unittest.mock import Mock
        client = Mock()
        client.pull_prompt.side_effect = RuntimeError('Hub unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Hub unavailable'):
            module.pull_prompts_from_hub(client)

    def test_ab_contexts_and_version_are_retained(self):
        module = importlib.import_module('02_prompt_hub_ab_routing')
        retriever = RunnableLambda(lambda question: [Document(page_content='fact')])
        llm = FakeListChatModel(responses=['answer'])
        result = module.ask_ab(retriever, llm, module.PROMPT_V2, 'question', 'v2')
        self.assertEqual(result['contexts'], ['fact'])
        self.assertEqual(result['version'], 'v2')
        self.assertEqual(result['answer'], 'answer')


class EvaluationTests(unittest.TestCase):
    def test_dataset_preserves_reference_and_separate_passages(self):
        module = importlib.import_module('03_ragas_evaluation')
        results = [{'question': 'question', 'reference': 'reference',
                    'answer': 'answer', 'contexts': ['passage one', 'passage two']}]
        sample = module.build_ragas_dataset(results).samples[0]
        self.assertEqual(sample.user_input, 'question')
        self.assertEqual(sample.response, 'answer')
        self.assertEqual(sample.reference, 'reference')
        self.assertEqual(sample.retrieved_contexts, ['passage one', 'passage two'])

    def test_single_candidate_model_still_produces_three_ragas_generations(self):
        import asyncio
        from ragas.llms import LangchainLLMWrapper
        from langchain_core.prompt_values import StringPromptValue
        class SingleCandidateModel(FakeListChatModel):
            n: int = 1
        model = SingleCandidateModel(responses=['one', 'two', 'three'])
        wrapper = LangchainLLMWrapper(model, bypass_n=True, bypass_temperature=True)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(wrapper.agenerate_text(StringPromptValue(text='question'), n=3))
        finally:
            loop.close()
        self.assertEqual(model.n, 1)
        self.assertEqual(len(result.generations[0]), 3)

    def test_nonfinite_metric_is_rejected_instead_of_omitted(self):
        import tempfile
        from unittest.mock import Mock
        module = importlib.import_module('03_ragas_evaluation')
        results = [{'question': 'question', 'reference': 'reference',
                    'answer': 'answer', 'contexts': ['passage']}]
        scores = {key: [1.0] for key in module.METRICS}
        scores['faithfulness'] = [float('nan')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evidence').mkdir()
            with patch.object(module, 'ROOT', root), \
                 patch.object(module, 'get_llm', return_value=FakeListChatModel(responses=['answer'])), \
                 patch.object(module, 'get_embeddings', return_value=Mock()), \
                 patch.object(module, 'evaluate', return_value=scores):
                with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                    module.run_ragas_eval(results, 'v1')
            self.assertFalse((root / 'evidence' / '03_scores_v1.json').exists())


class EmbeddingCacheTests(unittest.TestCase):
    def test_duplicates_are_cached_and_tasks_are_separate(self):
        from utils.cached_embeddings import CachedEmbeddings
        class Delegate:
            def embed_documents(self, texts):
                self.docs = texts
                return [[1.0, 0.0] for text in texts]
            def embed_query(self, text):
                return [0.0, 1.0]
        delegate = Delegate()
        with patch('utils.cached_embeddings.Cache', return_value={}), patch('utils.cached_embeddings.time.sleep'):
            embeddings = CachedEmbeddings(delegate, 'test-model')
            self.assertEqual(embeddings.embed_documents(['same', 'same']), [[1.0, 0.0]] * 2)
            self.assertEqual(delegate.docs, ['same'])
            self.assertEqual(embeddings.embed_query('same'), [0.0, 1.0])
            with patch.object(delegate, 'embed_documents', side_effect=AssertionError('cache miss')):
                self.assertEqual(embeddings.embed_documents(['same']), [[1.0, 0.0]])

    def test_quota_is_retried_but_authentication_errors_are_not(self):
        from utils.cached_embeddings import CachedEmbeddings
        from unittest.mock import Mock
        with patch('utils.cached_embeddings.Cache', return_value={}), patch('utils.cached_embeddings.time.sleep'):
            embeddings = CachedEmbeddings(Mock(), 'test-model')
            operation = Mock(side_effect=[RuntimeError('429 RESOURCE_EXHAUSTED'), [1.0]])
            self.assertEqual(embeddings._call(operation, 1), [1.0])
            self.assertEqual(operation.call_count, 2)
            operation = Mock(side_effect=RuntimeError('401 unauthorized'))
            with self.assertRaisesRegex(RuntimeError, '401'):
                embeddings._call(operation, 1)
            self.assertEqual(operation.call_count, 1)


if __name__ == '__main__':
    unittest.main()
