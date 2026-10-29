"""Static contracts supplement the retained long-context GPU replay diagnostic."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'csrc/engine/compiler/piecewise_prefill_compiler.cpp'


class FullPrefillReplayGuardsTest(unittest.TestCase):
    def test_capture_uses_capacity_not_dummy_lengths(self):
        text = SOURCE.read_text()
        self.assertIn('graphs.full_max_query_length = bucket;', text)
        self.assertIn('graphs.full_max_sequence_length = std::min(max_seq_len_', text)
        self.assertIn('capture_meta.max_query_length = graphs.full_max_query_length;', text)
        self.assertIn('capture_meta.max_sequence_length = graphs.full_max_sequence_length;', text)

    def test_length_guard_precedes_device_replay(self):
        text = SOURCE.read_text().split('auto &candidate = compiled_.at(graph_bucket);', 1)[1]
        guard = text.index('runtime_meta.max_query_length > candidate.full_max_query_length')
        replay = text.index('std::get<0>(candidate.full_compiled)->run()')
        self.assertIn('runtime_meta.max_sequence_length > candidate.full_max_sequence_length', text[guard:replay])
        self.assertIn('throw std::runtime_error', text[guard:replay])
        self.assertNotIn('return std::nullopt;', text[guard:replay])

    def test_query_and_kv_tails_are_independent(self):
        text = SOURCE.read_text()
        self.assertIn('std::vector<int32_t> cu_pad(off_tail, last_kv);', text)
        self.assertIn('cu_tail->data(), cu_pad.data(), cu_pad.size()', text)

    def test_full_padding_is_invalid_and_output_rows_are_live(self):
        text = SOURCE.read_text().split('auto &candidate = compiled_.at(graph_bucket);', 1)[1]
        replay = text.index('std::get<0>(candidate.full_compiled)->run()')
        self.assertIn('set_minus_one(unused_slots);', text[:replay])
        self.assertIn('full_logits->narrow({{1, 0, runtime_n_req}})', text[replay:])

    def test_full_mode_captures_mid_and_mixed_chunks(self):
        text = SOURCE.read_text().split('auto &candidate = compiled_.at(graph_bucket);', 1)[1]
        self.assertTrue(text.lstrip().startswith('if (prefill_full_cg_enabled()'))
        self.assertIn('is_fully_device_replayed()', SOURCE.read_text())


if __name__ == '__main__':
    unittest.main()
