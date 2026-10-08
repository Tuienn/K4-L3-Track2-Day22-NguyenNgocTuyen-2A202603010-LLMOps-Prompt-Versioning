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
