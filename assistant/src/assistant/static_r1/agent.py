"""A3: (Tracker || Direct) -> Intra -> Inter -> Editor; valid dependencies required."""
import ast
import copy
import asyncio
import hashlib
import json
import math
import operator
import re
import time
from types import SimpleNamespace
from .artifacts import CODE, write_json, output_path
from .schemas import JSON_SCHEMAS
from .prompts import TRACKER, INTRA, DIRECT, INTER, EDITOR
from .progress import parse_parts,plain_parts,assemble as assemble_parts,refresh,view,receipt,apply_edits,commit_records
from .variant import ARCHITECTURE

VERSION = 'A3_R1'
SOURCES = ('agent.py','prompts.py','client.py','schemas.py','general_policy.md','artifacts.py','progress.py','variant.py','session.py','recovery.py','token_count.py','limits.py','__init__.py')

def fingerprint():
    h=hashlib.sha256()
    for name in SOURCES:h.update(name.encode()+b'\0'+(CODE/name).read_bytes())
    return h.hexdigest()

def calculate(expression):
    if len(expression)>160:raise ValueError('expression too long')
    tree=ast.parse(expression,mode='eval')
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
         ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}
    def visit(node,depth=0):
        if depth>12:raise ValueError('expression too deep')
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):value=node.value
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=visit(node.operand,depth+1)*(-1 if isinstance(node.op,ast.USub) else 1)
        elif isinstance(node,ast.BinOp) and type(node.op) in ops:
            a,b=visit(node.left,depth+1),visit(node.right,depth+1)
            if isinstance(node.op,ast.Pow) and abs(b)>4096:raise ValueError('exponent too large')
            value=ops[type(node.op)](a,b)
        else:raise ValueError('unsupported expression')
        if not math.isfinite(value) or abs(value)>1e15:raise ValueError('result outside bounds')
        return value
    return visit(tree.body)

def render_history(history):
    counts={'user':0,'assistant':0};parts=[]
    for message in history:
        role=message['role'];counts[role]+=1
        ref=('u' if role=='user' else 'a')+str(counts[role])
        parts.append(f'### {ref} — {role}\n{message["content"]}')
    return '\n\n'.join(parts)

def verified_evidence(proposal,history):
    users={f'u{i}':m['content'] for i,m in enumerate((m for m in history if m['role']=='user'),1)}
    valid=[];rejected=[]
    for item in proposal.get('user_evidence',[]):
        source=item.get('source');quote=item.get('quote')
        if source in users and isinstance(quote,str) and quote.strip() and quote in users[source]:
            if item not in valid:valid.append(item)
        else:rejected.append(item)
    return valid,rejected

def render_evidence(evidence):
    return '\n'.join(f'- {e["source"]}: {e["quote"]}' for e in evidence) or 'No selected quotes; consult the original conversation.'

class StaticAgent:
    def __init__(self, *, trace_dir=None, client=None):
        # Resolve the application-selected audit directory before constructing
        # a client; output placement is owned by the caller.
        trace_dir = output_path(trace_dir) if trace_dir is not None else None
        if client is None: raise ValueError("R1 requires an explicitly configured client")
        self.client=client
        self.baseline=SimpleNamespace(name='static_progression')
        self.history=[];self.trace_dir=trace_dir;self.turn=0
        self.last_call_metadata={}
        self.work_state={}

    async def respond(self,user_message):
        self.turn+=1;start=time.monotonic();begin=len(self.client.calls)
        visible=[*self.history,dict(role='user',content=user_message)]
        context='## Visible Conversation\n\n'+render_history(visible)
        prior_view='\n\n## Previously Committed Work State\n'+view(self.work_state)
        context+=prior_view
        audit=dict(version=VERSION,architecture=ARCHITECTURE,revision=fingerprint(),turn=self.turn,degradations=[],schedule=[],committed=False)
        tasks=[]
        def event(role,kind):audit['schedule'].append(dict(role=role,event=kind,elapsed=time.monotonic()-start))
        async def invoke(role,prompt,text,json_mode=False,schema_override=None):
            event(role,'start')
            try:return await self.client.call(role,[dict(role='system',content=prompt),dict(role='user',content=text)],json_mode=json_mode,schema_override=schema_override)
            finally:event(role,'end')
        try:
            async with asyncio.timeout(7200):
                # S1M-derived stage 1: Tracker and the unchanged independent Direct.
                # Start the structured branch as soon as Tracker finishes, without waiting for Direct.
                async def track():
                    try:
                        value=json.loads(await invoke('tracker',TRACKER,context,True))
                        for key in ('user_evidence','adjacent','calculations'):
                            if not isinstance(value.get(key,[]),list):raise ValueError('invalid tracker list: '+key)
                        return value
                    except Exception as exc:
                        audit['degradations'].append(dict(role='tracker',error=str(exc)))
                        return dict(user_evidence=[],adjacent=[],calculations=[])
                # The actual conversational actor uses native roles, matching the
                # baseline interface. A transcript document is appropriate for the
                # internal extractor/reviewer, not a substitute for conversational turns.
                async def actor(role,prompt,working=''):
                    event(role,'start')
                    try:
                        value=json.loads(await self.client.call(role,[dict(role='system',content=prompt+prior_view+working),*visible],json_mode=True))
                        audit.setdefault('state_objections',{})[role]=value.get('state_objections',[])
                        return parse_parts(value,visible,role)
                    except Exception as exc:
                        audit['degradations'].append(dict(role=role,error=type(exc).__name__))
                        return []
                    finally:event(role,'end')
                tracker_task=asyncio.create_task(track());direct_task=asyncio.create_task(actor('intra_alternative',DIRECT))
                tasks.extend([tracker_task,direct_task]);proposal=await tracker_task
                state,rejected_goals=refresh(self.work_state,proposal,visible,self.turn)
                audit.update(rejected_goals=rejected_goals,previous_state=self.work_state)
                current_view='\n\n## Current Feedback Interpretation\n'+view(state)
                evidence,rejected_evidence=verified_evidence(proposal,visible)
                audit.update(verified_user_evidence=evidence,rejected_evidence=rejected_evidence)
                calculations=[]
                for item in proposal.get('calculations',[])[:12]:
                    try:calculations.append(dict(**item,result=calculate(item['expression'])))
                    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError,SyntaxError):
                        calculations.append(dict(request=item,error='unsupported or invalid expression; verify manually'))
                evidence_view='\n\n## Verified User Quotations\n'+render_evidence(evidence)
                numeric_view=('\n\n## Proposed Arithmetic Checks\n'+json.dumps(calculations,ensure_ascii=False)) if calculations else ''
                opportunities='\n'.join(f'{i}. {x.get("goal","")}\n   - Contextual reason: {x.get("why_now","")}' for i,x in enumerate(proposal.get('adjacent',[]),1))
                inter_input=context+current_view+evidence_view+'\n\n## Possible Continuations\n'+opportunities+numeric_view
                inter_input+='\n\n## Your Assignment\nOpportunities are hypotheses, not user facts or obligations.'
                if ARCHITECTURE in ('A1','A2'):
                    inter_input+=' Intra is drafting in parallel. You have not seen its choice or output. Supply only independently useful continuations, or clearly conditional alternatives if the user format permits them. Leave items empty when useful continuation requires the unseen choice. Editor will check integration and duplication.'
                async def inter():
                    if not proposal.get('adjacent'):
                        audit['inter_status']='not_applicable';return ''
                    try:
                        result=json.loads(await invoke('inter',INTER,inter_input,True))
                        audit.setdefault('state_objections',{})['inter']=result.get('state_objections',[])
                        items=result.get('items')
                        if not isinstance(items,list):raise ValueError('invalid inter items')
                        parts=[]
                        for content in items:
                            if not isinstance(content,str):raise ValueError('invalid continuation text')
                            if not content.strip():
                                audit['empty_inter_items_omitted']=audit.get('empty_inter_items_omitted',0)+1
                                continue
                            if content.strip() not in parts:parts.append(content.strip())
                        audit['inter_items']=[dict(position=i,content=s) for i,s in enumerate(parts,1)]
                        audit['inter_status']='selected' if parts else 'voluntary_empty'
                        return '\n\n'.join(parts)
                    except Exception as exc:
                        audit['inter_status']='degraded';audit['degradations'].append(dict(role='inter',error=type(exc).__name__))
                        return ''
                intra_task=asyncio.create_task(actor('intra',INTRA,current_view+evidence_view+numeric_view));tasks.append(intra_task)
                if ARCHITECTURE=='A3':
                    primary_parts=await intra_task
                    if primary_parts:
                        inter_input+='\n\n## Actual Intra Draft\n'+assemble_parts(primary_parts)
                        inter_input+='\n\n## Your Assignment\nRead this actual draft. Supply useful work it has not covered. If its premise is wrong, report a brief state objection; do not build on an unsupported choice.'
                        inter_task=asyncio.create_task(inter());tasks.append(inter_task);extra=await inter_task
                    else:
                        audit['inter_status']='dependency_unavailable'
                        audit['inter_skip_reason']='Intra produced no valid current-task draft; Direct remains independent.'
                        extra=''
                else:
                    inter_task=asyncio.create_task(inter());tasks.append(inter_task)
                    primary_parts,extra=await asyncio.gather(intra_task,inter_task)
                alternative_parts=await direct_task
                if not primary_parts and not alternative_parts:raise ValueError('both independent actors failed')
                audit['raw_intra_parts']=copy.deepcopy(primary_parts)
                audit['raw_direct_parts']=copy.deepcopy(alternative_parts)
                draft=assemble_parts(primary_parts);alternate=assemble_parts(alternative_parts)
                if not primary_parts:primary_parts=copy.deepcopy(alternative_parts)
                if not alternative_parts:alternative_parts=copy.deepcopy(primary_parts)
                def assemble(parts):
                    parts=[p.copy() for p in parts]
                    for item in audit.get('inter_items',[]):
                        content=item['content'].strip()
                        if content and content not in [p['text'] for p in parts]:
                            parts.append(dict(action='extend',source='unknown',text=content,origin='inter',source_valid=False))
                    return parts
                # Direct stays independent: Inter is never appended to it.
                if ARCHITECTURE!='A1':primary_parts=assemble(primary_parts)
                candidate=assemble_parts(primary_parts)
                candidate_alternative=assemble_parts(alternative_parts)
                audit['candidate']=candidate
                audit['candidate_alternative']=candidate_alternative
                # Stage 4 receives original evidence, not tracker paraphrases or plans.
                final_input=context+current_view+evidence_view+numeric_view+'\n\n## Complete Primary Candidate\n'+candidate+'\n\n## Complete Alternative Candidate\n'+candidate_alternative
                if ARCHITECTURE=='A1':
                    final_input=context+current_view+evidence_view+numeric_view+'\n\n## Current-Task Contribution (keep_primary selects ONLY this)\n'+candidate+'\n\n## Separate Inter Contribution (not yet merged)\n'+extra+'\n\n## Complete Independent Direct Candidate (keep_alternative)\n'+candidate_alternative
                    final_input+='\n\n## Candidate Contract\nTo integrate any Inter text or combine contributions, choose repair and supply the complete final reply. keep_primary delivers only the current-task contribution; keep_alternative delivers only Direct. Do not silently omit a useful continuation just to avoid repair.'
                else:
                    final_input+='\n\n## Candidate Contract\nPrimary is the deterministic Intra-plus-Inter assembly. Alternative is the untouched independent Direct. Check cross-part consistency and duplication. keep preserves the complete selected reply; repair returns a complete replacement.'
                if ARCHITECTURE=='A3':
                    final_input=context+current_view+evidence_view+numeric_view
                    final_input+='\n\n## Primary Candidate Components\nThe exact primary reply is the nonempty component texts below joined by two newlines.'
                    final_input+='\n\n### Current-Task Component\n'+assemble_parts([p for p in primary_parts if p.get('origin')!='inter'])
                    final_input+='\n\n### Optional Inter Component\n'+assemble_parts([p for p in primary_parts if p.get('origin')=='inter'])
                    final_input+='\n\n## Complete Independent Direct Candidate\n'+candidate_alternative
                    final_input+='\n\n## Candidate Contract\nkeep_primary delivers the complete primary assembly; keep_alternative delivers only Direct. repair supplies the complete final reply. The components are shown separately to locate defects and reusable content, not to require all of them. If Direct is better, preserve an Inter contribution only when it remains useful and compatible with Direct and the original user request. Do not retain a contribution that depends on a rejected Intra premise.'
                    audit['candidate_availability']=dict(intra=bool(audit['raw_intra_parts']),direct=bool(audit['raw_direct_parts']),inter=audit['inter_status'])
                    final_input+='\n\n## Branch Availability\n'+json.dumps(audit['candidate_availability'])+'\nIf a branch failed, both keep candidates may share the surviving draft; these are not two independent endorsements.'
                final_input+='\n\n## State Objections\n'+json.dumps(audit.get('state_objections',{}),ensure_ascii=False)
                final_input+='\n\n## Measured Draft Text Facts\n'+json.dumps(dict(
                    candidate_whitespace_words=len(candidate.split()),candidate_characters=len(candidate),
                    intra_whitespace_words=len(draft.split()),intra_characters=len(draft),
                    intra_question_marks=draft.count('?')+draft.count('？'),
                    inter_whitespace_words=len(extra.split()),
                    alternative_whitespace_words=len(candidate_alternative.split()),
                    alternative_characters=len(candidate_alternative)))
                choices=['keep_primary','keep_alternative','repair']
                previous_reply=self.history[-1]['content'] if self.history and self.history[-1]['role']=='assistant' else None
                blocked=[]
                if state.get('change_strategy') and previous_reply:
                    for decision,text in [('keep_primary',candidate),('keep_alternative',candidate_alternative)]:
                        if text.strip()==previous_reply.strip():choices.remove(decision);blocked.append(decision)
                editor_schema=copy.deepcopy(JSON_SCHEMAS['editor'])
                editor_schema['properties']['decision']['enum']=choices
                if choices==['repair']:editor_schema['properties']['parts']['minItems']=1
                audit['delivery_choice_control']=dict(allowed=choices,blocked=blocked,
                    rule='Verified latest-user stall interpretation, count>=2, exact full-text repeat of last actual delivery; no semantic similarity threshold')
                if blocked:
                    final_input+='\n\n## Available Delivery Decisions\n'+', '.join(choices)+'\nThe omitted keep choices exactly repeat the last actual reply after repeated reported lack of progress. You can repair the reply. A requested repetition or genuine missing evidence still follows the original user context; do not fabricate facts to look different.'
                delivered_role='editor'
                try:
                    review=json.loads(await invoke('editor',EDITOR,final_input,True,editor_schema))
                    if review.get('decision') not in choices:
                        raise ValueError('invalid delivery review')
                    if review['decision']=='repair':delivered_parts=parse_parts(review,visible,'editor')
                    else:delivered_parts=primary_parts if review['decision']=='keep_primary' else alternative_parts
                    reply=assemble_parts(delivered_parts)
                    if not reply.strip():raise ValueError('empty final delivery')
                    audit['review']=review
                    state,accepted_edits,rejected_edits=apply_edits(state,review.get('state_edits',[]),visible,self.turn,self.work_state)
                    audit.update(accepted_state_edits=accepted_edits,rejected_state_edits=rejected_edits)
                except Exception as exc:
                    delivered_parts=alternative_parts if 'keep_primary' not in choices and 'keep_alternative' in choices else primary_parts;reply=assemble_parts(delivered_parts);delivered_role='assembled';audit['degradations'].append(dict(role='editor',error=type(exc).__name__,fallback='available_candidate',choice_control_satisfied=bool(set(choices)-{'repair'})))
                delivered=receipt(delivered_parts,reply,self.turn)
                state=commit_records(state,self.work_state,delivered,audit,self.turn)
                audit.update(work_state=state,delivery=delivered,primary_parts=primary_parts,alternative_parts=alternative_parts)
                try:
                    audit['visible_response_tokens']=await self.client.count_text(reply)
                except Exception as exc:
                    audit['token_count_error']=type(exc).__name__
                # No await between state/history assignment and returning the actual reply.
                self.work_state=state
                self.history=[*visible,dict(role='assistant',content=reply)]
                audit['committed']=True
                audit.update(tracker=proposal,calculations=calculations,intra=draft,intra_alternative=alternate,inter=extra,reply=reply)
                return reply
        finally:
            pending=[task for task in tasks if not task.done()]
            for task in pending:task.cancel()
            if pending:await asyncio.gather(*pending,return_exceptions=True)
            calls=self.client.calls[begin:]
            self.last_call_metadata=dict(model=self.client.profile.model_id,input_tokens=sum(c.get('input_tokens',0) for c in calls),
                output_tokens=sum(c.get('output_tokens',0) for c in calls),
                internal_calls=len(calls),elapsed_seconds=time.monotonic()-start,
                final_generation_output_tokens=next((c.get('output_tokens',0) for c in reversed(calls) if c['role']=='editor'),0),
                model_generation=self.client.generation,static_version=VERSION)
            if 'reply' in audit:
                chosen=next((c for c in reversed(calls) if c['role']==delivered_role and c.get('response')==audit['reply'] and not c.get('error')),None)
                if chosen is not None:self.last_call_metadata.update(visible_response_tokens=chosen['output_tokens'],visible_token_source='local_nonthinking_final_completion')
                if 'visible_response_tokens' in audit:self.last_call_metadata.update(visible_response_tokens=audit['visible_response_tokens'],visible_token_source='tokenized_final_delivery')
            audit.update(calls=calls,metadata=self.last_call_metadata)
            self.last_audit=audit
            if self.trace_dir:write_json(self.trace_dir/f'turn_{self.turn:02d}.json',audit)
