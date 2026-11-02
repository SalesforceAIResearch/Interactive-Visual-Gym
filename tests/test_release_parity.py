"""Release adapters and scoring must preserve the published evaluation contract."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / e) for e in ('cooksim', 'vhhome', 'screensim')]
ITEMS = ('T1','T2','S1','S2','H1','H2','H3','L1','L2','E1')


def verdict(fail=(), na=()):
    return {'score': 0.5, 'items': {k: {'pass': None if k in na else k not in fail} for k in ITEMS}}


class ScoringParityTests(unittest.TestCase):
    def test_per_judge_gate_not_merged_gate(self):
        from ivg_submission.grading import round_score
        from screensim_bench.scoring.five_categories import rounds as screen_rounds
        from vhhome_bench.scoring.five_categories import rounds as vh_rounds
        row = {'judges': {'gemini': verdict(fail=('T1',)), 'gpt': verdict(fail=('E1',))}}
        # Gemini is gated to 0; GPT passes 4/5 categories: mean = .4.
        self.assertAlmostEqual(round_score(row)['score'], .4)
        for fn, key in [(screen_rounds,'rounds_detail'),(vh_rounds,'rounds')]:
            got=fn({key:[row]})[0]
            self.assertAlmostEqual(got['score'], .4)
            self.assertEqual(got['flags']['truthful'], .5)
            self.assertEqual(got['gated_share'], .5)

    def test_historical_na_spelling_is_not_a_pass(self):
        from screensim_bench.scoring.aq_contract import judge_round
        from cooksim_bench.scoring.sq_v4_three_runs import round_clean
        items = verdict()['items']
        items['L1']['pass'] = 'na'
        items['L2']['pass'] = None
        self.assertNotIn('listens', judge_round(items)['flags'])
        self.assertNotIn('listens', round_clean(items))

    def test_all_na_and_failed_judges_are_not_scores(self):
        from screensim_bench.scoring.five_categories import rounds
        from screensim_bench.scoring.rubric import usable
        self.assertFalse(usable(verdict(na=ITEMS)))
        self.assertEqual(rounds({'rounds_detail':[{'judges':{'gemini':verdict(na=ITEMS), 'gpt':{'error':'failed'}}}]}), [])

    def test_grading_uses_published_rubric_evidence(self):
        from ivg_submission.grading import commands
        jobs=commands('/tmp/unused',{'episodes':[{'engine':e,'run':1} for e in ['cooksim','vhhome','screensim']]})
        cook=next(j for j in jobs if j['module'].endswith('sq_v4_score'))
        vh=next(j for j in jobs if j['module']=='vhhome_bench.scoring.rubric')
        self.assertEqual(cook['args'][cook['args'].index('--categories')+1], 'all')
        self.assertNotIn('--frames',vh['args'])

    def test_success_and_detection_have_different_denominators(self):
        from cooksim_bench.scoring.paper_table import aggregate
        episodes={f'frames/baseline/m/b3_{i}':{'v3':{'in_time_success':success,'tp':tp},
             'evidence':{'errors_fired':n,'prevented':0}} for i,(success,tp,n) in enumerate([(1,0,0),(0,1,1),(0,0,3)])}
        row=aggregate(episodes)['models']['m']
        self.assertAlmostEqual(row['success'],1/3)
        self.assertEqual(row['recall'],.25)
        self.assertEqual(row['runs']['1']['episodes'],3)

    def test_vh_detection_windows_and_prevention(self):
        from vhhome_bench.scoring import metrics as m
        def episode(error, flag_time, dev=20, release=30, flag=None):
            return {'scenario':'unknown','episode':'unknown','ticks':100,'outcome':'won',
                'phases':[{'err':error,'dev_live':dev,'window':[10,100],'released_at':release}],
                'conversation':[{'speaker':'vlm','t':flag_time,'text':'Stop', 'vlm_json':{'flag':{'error':flag or error}}}]}
        with patch.object(m,'t_ref_measured',return_value=60), patch.object(m,'WINDOW_CLOSE',True), patch.object(m,'PERSON_CREDIT',True):
            early=m.episode_metrics(episode('stall',15))
            self.assertEqual(early['prevented_n'],1)
            self.assertEqual(early['DT'],1)
            self.assertEqual(m.episode_metrics(episode('stall',60))['R'],1)
            self.assertEqual(m.episode_metrics(episode('stall',61))['R'],0)
            self.assertEqual(m.episode_metrics(episode('wrong_placement',90))['R'],1)

    def test_cook_cli_table_uses_paper_rule(self):
        from cooksim_bench.scoring import sq_v4_three_runs as c
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sq_v4/frames/baseline/model/b3_example.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'model':'model','rounds':[{'judges':{'gemini':verdict(fail=('T1',)), 'gpt':verdict(fail=('E1',))}}]}))
            with patch.object(c,'BENCH',tmp), patch.object(c,'RULE','paper'):
                c.main()
            self.assertAlmostEqual(json.loads((Path(tmp)/'sq_v4/three_runs_five.json').read_text())['model']['mean'],.4)


@unittest.skipUnless(importlib.util.find_spec('anthropic') and importlib.util.find_spec('httpx'), 'Install anthropic extra for SDK transport tests')
class ClaudeAdapterTests(unittest.TestCase):
    def test_all_three_use_actual_sdk_image_payload_and_reply_parser(self):
        import anthropic, httpx
        from cooksim_bench.backends.anthropic_backend import AnthropicBackend as Cook
        from screensim.interact.anthropic_backend import AnthropicAssistantBackend as Screen
        from vhhome_bench.anthropic_backend import AnthropicBackend as VH
        real=anthropic.Anthropic
        for cls in (Cook,Screen,VH):
            for refusal in (False,True):
                with self.subTest(adapter=cls.__module__,refusal=refusal):
                    requests=[]
                    def handler(req):
                        requests.append(json.loads(req.content))
                        return httpx.Response(200,json={'id':'msg_test','type':'message','role':'assistant','model':'claude-test',
                            'content':[{'type':'text','text':'{"text":"Stop before adding lettuce.","flag":null}'}],
                            'stop_reason':'refusal' if refusal else 'end_turn','stop_sequence':None,
                            'usage':{'input_tokens':12,'output_tokens':8}})
                    def client(**kw):
                        kw.update(api_key='test-only',http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                        return real(**kw)
                    with patch.object(anthropic,'Anthropic',side_effect=client), patch.dict(os.environ,{'ANTHROPIC_API_KEY':'test-only'}):
                        b=cls('claude-test');reply,meta=b.call('system unchanged','user unchanged',[b'\xff\xd8\xffjpeg',b'\x89PNGpng'])
                        b.client.close()
                    body=requests[0];content=body['messages'][0]['content']
                    self.assertEqual(content[0]['source']['media_type'],'image/jpeg')
                    self.assertEqual(base64.b64decode(content[0]['source']['data']),b'\xff\xd8\xffjpeg')
                    self.assertEqual(content[1]['source']['media_type'],'image/png')
                    self.assertEqual(content[2]['text'],'user unchanged')
                    self.assertEqual(body['system'][0]['text'],'system unchanged')
                    self.assertEqual(body['output_config'],{'effort':'low'})
                    self.assertNotIn('temperature',body)
                    self.assertNotIn('tools',body)
                    self.assertEqual(reply,{} if refusal else {'text':'Stop before adding lettuce.','flag':None})
                    self.assertEqual(meta['stop_reason'],'refusal' if refusal else 'end_turn')

    def test_cli_routes_anthropic_explicitly(self):
        from screensim_bench.cli._common import resolve_assistant
        from vhhome_bench.cli._common import assistant_entry
        self.assertEqual(resolve_assistant('anthropic:claude-test')[:2],('anthropic','claude-test'))
        e=assistant_entry('anthropic:claude-test')
        self.assertEqual(e['env']['IV_ASSIST_BACKEND'],'anthropic')
        self.assertIn('GEMINI_API_KEY',e['needs'])
        self.assertIn('ANTHROPIC_API_KEY',e['needs'])
        self.assertEqual(assistant_entry('claude-opus-5')['status'],'ready')

    def test_vh_assistant_dispatch_does_not_use_gemini(self):
        from vhhome_bench import protocol as p
        from vhhome_bench.anthropic_backend import AnthropicBackend
        with patch.object(p,'ASSIST_BACKEND','anthropic'), patch.object(p,'ASSIST_MODEL','claude-test'), \
             patch.object(AnthropicBackend,'__init__',return_value=None), \
             patch.object(AnthropicBackend,'call',return_value=({'text':'Hello','flag':None},{'stop_reason':'end_turn'})) as call, \
             patch.object(p,'gemini_frames',side_effect=AssertionError('wrong backend')):
            self.assertEqual(p.assist_frames(None,'system','user',[b'frame']),{'text':'Hello','flag':None})
            call.assert_called_once_with('system','user',[b'frame'])


if __name__=='__main__':
    unittest.main()
