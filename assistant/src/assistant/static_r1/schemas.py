"""Frozen R1 transport schemas; never dumped into prompt messages."""
JSON_SCHEMAS = {
    'tracker':dict(type='object',properties={
        'user_evidence':{'type':'array','maxItems':10,'items':{'type':'object','properties':{'source':{'type':'string'},'quote':{'type':'string'}},'required':['source','quote'],'additionalProperties':False}},
        'adjacent':{'type':'array','maxItems':3,'items':{'type':'object','properties':{'goal':{'type':'string'},'why_now':{'type':'string'}},'required':['goal','why_now'],'additionalProperties':False}},
        'calculations':{'type':'array','items':{'type':'object','properties':{'expression':{'type':'string'},'meaning':{'type':'string'}},'required':['expression','meaning'],'additionalProperties':False}}},
        required=['user_evidence','adjacent','calculations'],additionalProperties=False),
    'inter':dict(type='object',properties={'items':{'type':'array','maxItems':3,'items':{'type':'string'}}},required=['items'],additionalProperties=False),
    'editor':dict(type='object',properties={'reason':{'type':'string'},'decision':{'type':'string','enum':['keep_primary','keep_alternative','repair']},'replacement':{'type':'string'}},required=['reason','decision','replacement'],additionalProperties=False),
    'synthetic_grade':dict(type='object',properties={'pass':{'type':'boolean'},'reason':{'type':'string'}},required=['pass','reason'],additionalProperties=False)
}

# Protocol-only schemas. Prompt messages contain only readable compact skeletons.
_ACTION_PART={'type':'object','properties':{'action':{'type':'string','enum':['deliver','ask','revise','extend','wait']},'source':{'type':'string'},'text':{'type':'string'}},'required':['action','source','text'],'additionalProperties':False}
_ACTION_PART['properties'].update(repeat_count={'type':'integer','minimum':1,'maximum':4096},separator={'type':'string','maxLength':8})
_PARTS={'type':'array','items':_ACTION_PART}
for _role in ('intra','intra_alternative'):
    JSON_SCHEMAS[_role]={'type':'object','properties':{'parts':_PARTS},'required':['parts'],'additionalProperties':False}
JSON_SCHEMAS['editor']={'type':'object','properties':{'reason':{'type':'string'},'decision':{'type':'string','enum':['keep_primary','keep_alternative','repair']},'parts':_PARTS},'required':['reason','decision','parts'],'additionalProperties':False}
JSON_SCHEMAS['tracker']['properties'].update({
 'goals':{'type':'array','maxItems':5,'items':{'type':'object','properties':{'id':{'type':'string'},'goal':{'type':'string'},'sources':{'type':'array','items':{'type':'string'}},'status':{'type':'string','enum':['open','user_accepted','withdrawn']},'missing':{'type':'string'}},'required':['id','goal','sources','status','missing'],'additionalProperties':False}},
 'feedback':{'type':'object','properties':{'status':{'type':'string','enum':['advanced','stalled','blocked','waiting','unknown']},'goal_id':{'type':'string'},'source':{'type':'string'},'quote':{'type':'string'},'reason':{'type':'string'}},'required':['status','goal_id','source','quote','reason'],'additionalProperties':False}})
JSON_SCHEMAS['tracker']['required'] += ['goals','feedback']

# Additive S1M protocol extensions; no policy or schema dump in prompt messages.
_OBJECTION={'type':'object','properties':{k:{'type':'string'} for k in ('goal_id','source','quote','reason')},'required':['goal_id','source','quote','reason'],'additionalProperties':False}
for _role in ('intra','inter'):
    JSON_SCHEMAS[_role]['properties']['state_objections']={'type':'array','items':_OBJECTION,'maxItems':3}
_EDIT={'type':'object','properties':{k:{'type':'string'} for k in ('target','id','source','quote','goal','status','missing','reason','goal_id')},'required':['target','source','quote','status','reason'],'additionalProperties':False}
_EDIT['properties']['target']['enum']=['goal','feedback']
_EDIT['properties']['status']['enum']=['open','withdrawn','disputed','advanced','stalled','blocked','waiting','unknown']
JSON_SCHEMAS['editor']['properties']['state_edits']={'type':'array','items':_EDIT,'maxItems':5}
