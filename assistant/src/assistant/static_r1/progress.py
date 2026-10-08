"""Episode-local, fallible visible state and receipts bound to actual delivered text."""
import hashlib
import json

ACTIONS={'deliver','ask','revise','extend','wait'}
FEEDBACK={'advanced','stalled','blocked','waiting','unknown'}

def user_events(history):
    return {f'u{i}':m['content'] for i,m in enumerate((m for m in history if m['role']=='user'),1)}

def parse_parts(value,history,origin):
    parts=value.get('parts')
    if not isinstance(parts,list) or not parts:raise ValueError('missing action parts')
    users=user_events(history);result=[]
    for part in parts:
        if not isinstance(part,dict) or part.get('action') not in ACTIONS or not isinstance(part.get('text'),str):
            raise ValueError('invalid action part')
        text=part['text'].strip()
        if not text:continue
        count=part.get('repeat_count',1);separator=part.get('separator',' ')
        if type(count) is not int or not 1<=count<=4096 or not isinstance(separator,str) or len(separator)>8:
            raise ValueError('invalid finite repetition parameters')
        rendering={}
        if count>1:
            if len(text)*count+len(separator)*(count-1)>262144:
                raise ValueError('finite repetition exceeds rendering character limit')
            rendering=dict(literal_repeat=dict(unit=text,count=count,separator=separator))
            text=separator.join([text]*count)
        source=part.get('source')
        result.append(dict(action=part['action'],source=source if source in users else 'unknown',
                           text=text,origin=origin,source_valid=source in users,**rendering))
    if not result:raise ValueError('empty action delivery')
    return result

def plain_parts(text,origin):
    if not text.strip():raise ValueError('empty delivery')
    return [dict(action='unknown',source='unknown',text=text.strip(),origin=origin,source_valid=False)]

def assemble(parts):return '\n\n'.join(p['text'] for p in parts)

def refresh(previous,proposal,history,turn):
    users=user_events(history);latest=f'u{len(users)}';old={g['id']:g.copy() for g in previous.get('goals',[])}
    next_id=previous.get('next_goal_id',1);rejected=[]
    for raw in proposal.get('goals',[])[:5]:
        if not isinstance(raw,dict):rejected.append(raw);continue
        sources=raw.get('sources',[])
        if (not sources or any(s not in users for s in sources) or not raw.get('goal') or
            raw.get('status') not in ('open','user_accepted','withdrawn')):
            rejected.append(raw);continue
        ident=raw.get('id')
        if ident not in old:ident=f'g{next_id}';next_id+=1
        old[ident]=dict(id=ident,goal=raw['goal'],sources=sources,status=raw['status'],
                       missing=raw.get('missing',''),first_seen=old.get(ident,{}).get('first_seen',turn),last_seen=turn)
    feedback=proposal.get('feedback',{})
    if not isinstance(feedback,dict):feedback={}
    valid=(feedback.get('status') in FEEDBACK and feedback.get('source')==latest and
           isinstance(feedback.get('quote'),str) and bool(feedback['quote'].strip()) and feedback['quote'] in users.get(latest,''))
    if not valid:feedback=dict(status='unknown',goal_id='',source=latest,quote='',reason='No verifiable latest-user feedback')
    feedback={**feedback,'quote_verified':bool(valid),'semantics':'model interpretation, not official success'}
    prior=previous.get('feedback',{});same=feedback.get('goal_id') in old and feedback.get('goal_id')==prior.get('goal_id')
    count=(previous.get('stall_count',0)+1 if same and prior.get('status')=='stalled' else 1) if valid and feedback['status']=='stalled' else 0
    return dict(goals=list(old.values()),next_goal_id=next_id,feedback=feedback,stall_count=count,
                change_strategy=count>=2,updated_turn=turn),rejected

def view(state):
    if not state:return 'No earlier committed work state.'
    lines=['These are fallible notes, not new user facts. Original messages and the latest correction take priority.']
    for g in state.get('goals',[]):
        lines.append(f'- {g["id"]} [{g["status"]}; user sources {", ".join(g["sources"])}]: {g["goal"]}' + (f' | missing: {g["missing"]}' if g.get('missing') else ''))
    f=state.get('feedback',{})
    lines.append(f'Latest interpreted feedback: {f.get("status","unknown")}; {f.get("reason","")}')
    lines.append(f'Consecutive supported stall observations on the same known goal: {state.get("stall_count",0)}')
    if state.get('change_strategy'):lines.append('Control: a repeated lack of progress was reported. Change the ineffective approach on this goal; preserve working content. Check latest user evidence before applying this note. Waiting or truly unavailable evidence is not an instruction to fabricate.')
    if state.get('delivery'):
        lines.append('Last actually delivered actions: '+', '.join(f'{x["action"]}({x["source"]})' for x in state['delivery']['parts']))
    return '\n'.join(lines)

def receipt(parts,reply,turn):
    assert assemble(parts)==reply
    start=0;record=[]
    for i,p in enumerate(parts):
        record.append({**p,'id':f't{turn}p{i+1}','start':start,'end':start+len(p['text']),
                       'action_semantics_verified':False})
        start+=len(p['text'])+2
    return dict(turn=turn,sha256=hashlib.sha256(reply.encode()).hexdigest(),parts=record,
                delivered_requests=[p['id'] for p in record if p['action']=='ask'])


def apply_edits(state,edits,history,turn,previous):
    """Evidence/reference guards, not an independent semantic validator."""
    import copy
    state=copy.deepcopy(state);users=user_events(history);accepted=[];rejected=[]
    goals={g['id']:g for g in state['goals']};latest=f'u{len(users)}'
    if not isinstance(edits,list):return state,[],[dict(error='edits must be list')]
    for edit in edits:
        if not isinstance(edit,dict):rejected.append(edit);continue
        source=edit.get('source');quote=edit.get('quote');target=edit.get('target')
        valid=source in users and isinstance(quote,str) and bool(quote.strip()) and quote in users[source]
        if target=='goal':
            valid=valid and edit.get('id') in goals and edit.get('status') in ('open','withdrawn','disputed') and bool(edit.get('goal'))
            if valid:
                g=goals[edit['id']];g.update(goal=edit['goal'],status=edit['status'],missing=edit.get('missing',''),sources=[source],last_seen=turn,editor_correction=dict(source=source,quote=quote,reason=edit.get('reason','')))
        elif target=='feedback':
            valid=valid and source==latest and edit.get('status') in FEEDBACK and edit.get('goal_id','') in ('',*goals)
            if valid:
                feedback={k:edit.get(k,'') for k in ('status','goal_id','source','quote','reason')}
                feedback.update(quote_verified=True,semantics='editor interpretation, not official success')
                prior=previous.get('feedback',{});same=feedback['goal_id'] in goals and feedback['goal_id']==prior.get('goal_id')
                count=(previous.get('stall_count',0)+1 if same and prior.get('status')=='stalled' else 1) if feedback['status']=='stalled' else 0
                state.update(feedback=feedback,stall_count=count,change_strategy=count>=2)
        else:valid=False
        (accepted if valid else rejected).append(edit)
    state['goals']=list(goals.values())
    return state,accepted,rejected


def commit_records(state,previous,delivery,audit,turn):
    """Additive S1M records; latest view stays compact, complete history stays auditable."""
    state['delivery']=delivery
    state['delivery_history']=[*previous.get('delivery_history',[]),delivery]
    state['feedback_history']=[*previous.get('feedback_history',[]),dict(turn=turn,**state['feedback'])]
    state['execution']=dict(turn=turn,architecture=audit['architecture'],decision=audit.get('review',{}).get('decision','fallback'),inter_status=audit.get('inter_status'),degraded=bool(audit['degradations']))
    return state
