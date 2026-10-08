"""R1 exercises the actual project pipeline without network or hidden-label input."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from assistant.config import load_model_profile
from assistant.static_r1 import R1Session
from user_simulator.audit.logger import AuditLogger
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty
from user_simulator.engine.episode import Episode
from user_simulator.mock_components import MockController, MockSatisfactionUpdater, MockUserRealizer
from user_simulator.policy.realization import DifficultyRealizationPolicy
from user_simulator.policy.selection import DifficultySelectionPolicy
from interaction_pipeline.core import run_interaction
from interaction_pipeline.config import load_pipeline_config
from interaction_pipeline.prepare import prepare_pipeline

ROOT=Path(__file__).resolve().parents[2]

def sample():
    return Sample.model_validate(dict(sample_id='r1-synthetic',reason_dag=dict(nodes=[
        dict(node_id='N1',node_type='intent',surface_user_message='Help me write a message.',node_intent='PRIVATE_HIDDEN_INTENT'),
        dict(node_id='END',node_type='terminal',node_intent='Done.')],
        edges=[dict(edge_id='E1',source='N1',target='END')])))

class Client:
    profile=SimpleNamespace(model_id='synthetic',profile_name='synthetic')
    generation={}
    def __init__(self):self.calls=[];self.inputs=[];self.closed=False
    async def call(self,role,messages,**kwargs):
        self.inputs.append(messages);self.calls.append(dict(role=role,input_tokens=10,output_tokens=100))
        if role=='tracker':return json.dumps(dict(user_evidence=[],goals=[],feedback={},adjacent=[],calculations=[]))
        if role=='editor':return json.dumps(dict(decision='keep_primary',reason='Complete draft',parts=[],state_edits=[]))
        return json.dumps(dict(parts=[dict(action='deliver',source='u1',text='Here is the complete answer.')]))
    async def count_text(self,text):return 6
    async def close(self):self.closed=True

@pytest.mark.asyncio
async def test_r1_pipeline_audit_delivery_and_close(tmp_path):
    s=sample();audit=AuditLogger(s.sample_id,output_dir=tmp_path,run_id='r1',level='full')
    episode=Episode(sample=s,difficulty=Difficulty.EASY,seed=1,controller=MockController(),
        satisfaction_updater=MockSatisfactionUpdater(),selection_policy=DifficultySelectionPolicy(),
        realization_policy=DifficultyRealizationPolicy(),user_realizer=MockUserRealizer(),audit_logger=audit)
    client=Client();profile=load_model_profile(str(ROOT/'assistant/configs/models/gpt_5_6_luna_non_thinking.yaml'))
    system=R1Session(profile=profile,client=client);streamed=[]
    async def sink(e):streamed.append(e)
    result=await run_interaction(episode=episode,assistant=system,audit=audit,event_sink=sink,update_memory=False)
    assert result.status=='completed' and client.closed
    events=[json.loads(x) for x in (audit.run_dir/'events.jsonl').read_text().splitlines()]
    private=[e for e in events if e['event_type']=='r1_turn']
    assert len(private)==1 and private[0]['payload']['committed']
    assert private[0]['payload']['inter_status']=='not_applicable'
    public=[e for e in events if e['event_type']=='assistant_generation_completed']
    assert public[0]['payload']['llm_call']['visible_response_tokens']==6
    assert public[0]['payload']['llm_call']['output_tokens']==400
    assert 'PRIVATE_HIDDEN_INTENT' not in json.dumps(client.inputs)
    assert not any(e.get('type')=='r1_turn' for e in streamed)


def test_r1_prepare_snapshot(tmp_path):
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(sample().model_dump_json()+"\n")
    prepared=prepare_pipeline(load_pipeline_config(),assistant_baseline='static_r1',
        assistant_model_profile='gemini_3_6_flash_non_thinking',simulator_model_profile='deepseek_v4_flash_0731_fast',simulator_dataset_path=dataset)
    snapshot=prepared.batch_config_snapshot(batch_id='r1-test',created_at='2026-10-08',samples=[sample()],concurrency=1,update_memory=False)
    assert snapshot['assistant']['static_r1']['architecture']=='A3_R1'
    assert set(snapshot['assistant']['static_r1']['prompts'])=={'TRACKER','DIRECT','INTRA','INTER','EDITOR'}
    assert snapshot['assistant']['active_baseline']=='static_r1'
    assert snapshot['assistant']['goal_progression'] is None
