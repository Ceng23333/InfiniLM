"""Run the real Python forwarding methods without importing device extensions."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == name)


class ChunkFinalityBridgeTest(unittest.TestCase):
    def test_runner_preserves_flags_and_drops_only_scheduling_mode(self):
        node = method('python/infinilm/llm/model_runner/model_runner.py', 'ModelRunner', '_model_forward')
        module = ast.Module(body=[node], type_ignores=[])
        namespace = {'_normalize_sampled_token_ids': lambda values: values}
        exec(compile(ast.fix_missing_locations(module), '<runner>', 'exec'), namespace)
        forward = Mock(return_value=SimpleNamespace(to_numpy=lambda: SimpleNamespace(tolist=lambda: [-1, 7])))
        obj = SimpleNamespace(processor=SimpleNamespace(build_model_inputs=Mock(return_value={
            'input_ids': 'ids', 'is_final_prefill_chunk': [False, True], 'scheduling_mode': 'MIXED'})),
            config=SimpleNamespace(temperature=0, top_p=1, top_k=1), speculative_runner=None,
            pipeline_control=None, model_engine=SimpleNamespace(forward=forward))
        self.assertEqual(namespace['_model_forward'](obj, object()), [-1, 7])
        forward.assert_called_once_with(input_ids='ids', is_final_prefill_chunk=[False, True])

    def test_input_builder_passes_flags_and_preserves_legacy_default(self):
        node = method('python/infinilm/infer_engine.py', 'InferEngine', '_build_input')
        cls = ast.ClassDef(name='Subject', bases=[ast.Name(id='Base', ctx=ast.Load())],
                           keywords=[], body=[node], decorator_list=[])
        base = type('Base', (), {'Input': staticmethod(lambda *a, **kw: kw)})
        namespace = {'Base': base}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<input>', 'exec'), namespace)
        obj = namespace['Subject']()
        self.assertEqual(obj._build_input(None)['is_final_prefill_chunk'], [])
        self.assertEqual(obj._build_input(None, is_final_prefill_chunk=[False, True])['is_final_prefill_chunk'], [False, True])


if __name__ == '__main__':
    unittest.main()
