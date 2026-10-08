"""Release checks: frozen behavior parity and provider protocol round trips."""
import copy
import json
import types
from pathlib import Path

import httpx
import pytest

from assistant.static_r1.agent import StaticAgent
from assistant.static_r1.client import ProfileClient, portable_schema
from assistant.static_r1.schemas import JSON_SCHEMAS
from assistant.config import EnvironmentSettings, load_model_profile

P=Path(__file__).resolve().parents[3]
# Digests captured from the original frozen R1, including degraded branch behavior.
FROZEN_CONTRACTS={'keep_primary:None': 'cd7dffa9cfeecd9938f44338b04489457e912c8e1e7c40c8480c253214b34436', 'keep_primary:tracker': 'cc2a30d8752c75bd178d8545528d77b78589dc92aaa862bd0605b98ce9279ca8', 'keep_primary:intra': '13889da2eaec6a30a6f1f1ca79756bbd6c4a7a17706e9074cfc5ed773c61e86a', 'keep_primary:inter': '82bac0e4bf9368332a2ef8261c596bc32720190858a4dd923ceaada38ed12c3b', 'keep_primary:editor': 'ef45f9a27bb14b14bdf9c6db68299ff839132ee601f7fd4cbefb4e5345574d8e', 'keep_alternative:None': '30d1776887885fd1b56f8d90c6e3d12b13c92209276d6cb42ab16aad474ec215', 'keep_alternative:tracker': '453709eb31c6dc51d95efd8ea4809ef69ebdbb8cbb0f63727a7b32eb8f2c9100', 'keep_alternative:intra': '5c65fc31edeade24cf2ec1207ed378cad1a96bcabd9e1d08997214696f4a3ef6', 'keep_alternative:inter': 'f85533331a0e9fcec6f9aaf022b45465b71bca92ea83b138bac4946503a4efe3', 'keep_alternative:editor': 'ef45f9a27bb14b14bdf9c6db68299ff839132ee601f7fd4cbefb4e5345574d8e', 'repair:None': '44b7d7eb24c19c18c0c75f9a337cac638d5c869e00131bdee78e954f9f8b6399', 'repair:tracker': '8e1274b6e0f7b6db3722145197c4cfbf210244bda4969643703357d692ba32ac', 'repair:intra': '4f221fea5462b80fff5851f83afa98f8115c8e67d6df2fb99dfddc27cd48f033', 'repair:inter': 'e4bb0fba023ce8724a7a4ab05c669b7b3994b9bf21b44f17fbcaac20525dae0b', 'repair:editor': 'ef45f9a27bb14b14bdf9c6db68299ff839132ee601f7fd4cbefb4e5345574d8e'}

def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

class FakeClient:
    def __init__(self,decision='keep_primary',fail=None):
        self.profile=types.SimpleNamespace(model_id='qwen3.6-27b')
        self.generation={};self.calls=[];self.requests=[];self.turn=0;self.decision=decision;self.fail=fail
    async def count_text(self,text):return len(text.split())
    async def call(self,role,messages,*,json_mode=False,schema_override=None):
        self.requests.append((role,copy.deepcopy(messages),copy.deepcopy(schema_override)))
        if role=='tracker':self.turn+=1
        self.calls.append(dict(role=role,input_tokens=1,output_tokens=1))
        if self.fail==role:raise ValueError('synthetic role failure')
        source=f'u{self.turn}';quote='I still need a concrete message.' if self.turn>1 else 'Help with a handoff.'
        if role=='tracker':result=dict(user_evidence=[dict(source=source,quote=quote)],goals=[dict(id='new' if self.turn==1 else 'g1',goal='Handoff message',sources=[source],status='open',missing='')],feedback=dict(status='stalled' if self.turn>1 else 'unknown',goal_id='g1',source=source,quote=quote,reason='User feedback'),adjacent=[dict(goal='Handoff example',why_now='Useful application')],calculations=[dict(expression='3 * 4',meaning='Tiles')])
        elif role=='inter':result=dict(items=['Next owner: [name].'],state_objections=[])
        else:
            result=dict(parts=[dict(action='deliver',source=source,text='Primary handoff.' if role=='intra' else 'Direct handoff.' if role=='intra_alternative' else 'Repaired handoff.')])
            if role=='editor':result.update(decision=self.decision,reason='Compare usable drafts',state_edits=[])
        return json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize('decision',['keep_primary','keep_alternative','repair'])
@pytest.mark.parametrize('fail',[None,'tracker','intra','inter','editor'])
async def test_frozen_r1_behavior(decision,fail):
    system=StaticAgent(client=FakeClient(decision,fail));rows=[]
    for text in ['Help with a handoff.','I still need a concrete message.','I still need a concrete message.']:
        reply=await system.respond(text)
        rows.append(dict(reply=reply,state=copy.deepcopy(system.work_state),history=copy.deepcopy(system.history)))
    assert digest(dict(rows=rows,requests=system.client.requests))==FROZEN_CONTRACTS[decision+':'+str(fail)]

@pytest.mark.parametrize('role',list(JSON_SCHEMAS))
def test_nullable_schema_preserves_required_contract(role):
    def inspect(original,wire):
        if original.get('type')=='object':
            assert set(wire['required'])==set(original['properties'])
            for name,field in original['properties'].items():
                new=wire['properties'][name]
                if name not in original.get('required',[]):
                    assert new['anyOf'][1]=={'type':'null'};new=new['anyOf'][0]
                inspect(field,new)
        if 'items' in original:inspect(original['items'],wire['items'])
    inspect(JSON_SCHEMAS[role],portable_schema(JSON_SCHEMAS[role]))

@pytest.mark.asyncio
async def test_remote_wire_and_null_round_trip():
    profile=load_model_profile(str(P/'assistant/configs/models/gpt_5_6_luna_non_thinking.yaml'))
    requests=[]
    async def handler(request):
        payload=json.loads(request.content);requests.append(payload)
        obj=dict(parts=[dict(action='deliver',source='u1',text='Ready.',repeat_count=None,separator=None)],state_objections=None)
        chunk=dict(id='test',model=profile.model_id,provider='mock',choices=[dict(delta=dict(content=json.dumps(obj)),finish_reason='stop')],usage=dict(prompt_tokens=11,completion_tokens=25,cost=.001))
        return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client=ProfileClient(profile,EnvironmentSettings(_env_file=None,openrouter_api_key='test-only'),http=http)
        text=await client.call('intra',[dict(role='user',content='Write a greeting.')],json_mode=True)
        assert json.loads(text)==dict(parts=[dict(action='deliver',source='u1',text='Ready.')])
        assert requests[0]['reasoning']==dict(effort='none')
        assert 'temperature' not in requests[0] and 'chat_template_kwargs' not in requests[0]
        assert 'json_schema' not in requests[0]['messages'][0]['content']
        assert client.calls[0]['usage']['cost']==.001
        assert 'Authorization' not in json.dumps(client.calls)


@pytest.mark.asyncio
async def test_local_qwen_keeps_frozen_wire_settings():
    profile=load_model_profile(str(P/'assistant/configs/models/qwen_3_6_27b_vllm_non_thinking.yaml'))
    requests=[]
    async def handler(request):
        payload=json.loads(request.content);requests.append((request.url.path,payload))
        if request.url.path=='/tokenize':return httpx.Response(200,json={'count':5})
        chunk=dict(choices=[dict(delta=dict(content='{"items":[]}'),finish_reason='stop')],usage=dict(prompt_tokens=5,completion_tokens=7))
        return httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client=ProfileClient(profile,EnvironmentSettings(_env_file=None),http=http)
        await client.call('inter',[dict(role='user',content='Optional continuation.')],json_mode=True)
        wire=requests[1][1]
        assert wire['chat_template_kwargs']==dict(enable_thinking=False,preserve_thinking=False)
        assert {k:wire[k] for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty','max_tokens')}==dict(temperature=.7,top_p=.8,top_k=20,min_p=0.,presence_penalty=1.5,repetition_penalty=1.,max_tokens=32768)
        assert wire['response_format']['json_schema']['schema']==JSON_SCHEMAS['inter']
        assert await client.count_text('Visible text.')==5


@pytest.mark.asyncio
async def test_context_overflow_is_not_truncated_or_sent():
    profile=load_model_profile(str(P/'assistant/configs/models/qwen_3_6_27b_vllm_non_thinking.yaml'))
    paths=[]
    async def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200,json={'count':262144})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client=ProfileClient(profile,EnvironmentSettings(_env_file=None),http=http)
        with pytest.raises(ValueError,match='context_overflow'):
            await client.call('tracker',[dict(role='user',content='oversized')],json_mode=True)
        assert paths==['/tokenize']


@pytest.mark.asyncio
async def test_release_factory_and_shared_pool(monkeypatch):
    from assistant.config import load_config
    from assistant.factory import build_assistant_components
    from assistant.static_r1 import R1Session, snapshot
    monkeypatch.delenv('R1_VISIBLE_TOKENIZER', raising=False)
    config = load_config(cli_overrides={
        'components': {'baseline': 'static_r1'},
        'models': {'assistant': 'gpt_5_6_luna_non_thinking'},
    })
    env = EnvironmentSettings(_env_file=None, openrouter_api_key='test-only')
    a = build_assistant_components(config=config, environment=env)
    b = build_assistant_components(config=config, environment=env)
    try:
        assert isinstance(a.session, R1Session)
        assert a.session.client.slots.semaphore() is b.session.client.slots.semaphore()
        assert a.session.last_call_metadata['visible_response_tokens'] is None
        assert a.skill_framework is None and a.memory_framework is None
        assert set(snapshot()['prompts']) == {'TRACKER','DIRECT','INTRA','INTER','EDITOR'}
    finally:
        await a.session.close()
        await b.session.close()


def test_r1_rejects_skill_and_memory():
    from assistant.config import load_config
    from assistant.factory import configured_components_require_openrouter
    from assistant.exceptions import ConfigurationError
    for model in ('gpt_5_6_luna_trace2skill','qwen_3_6_27b_exprag'):
        config = load_config(cli_overrides={
            'components': {'baseline':'static_r1','profile_skill_enabled':False},
            'models': {'assistant': model},
        })
        with pytest.raises(ConfigurationError):
            configured_components_require_openrouter(config)


@pytest.mark.asyncio
async def test_r1_reset_and_delivered_token_metadata():
    from assistant.static_r1 import R1Session
    profile = load_model_profile(str(P/'assistant/configs/models/gpt_5_6_luna_non_thinking.yaml'))
    session = R1Session(profile=profile,client=FakeClient())
    await session.respond('Help with a handoff.')
    assert len(session.history)==2 and session.engine.work_state['delivery_history']
    assert session.last_call_metadata['visible_response_tokens']==len(session.history[-1].content.split())
    session.reset()
    assert not session.history and not session.engine.work_state
    assert session.engine.turn==0
