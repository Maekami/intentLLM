import re

def recovery_note(error, partial=''):
    """Technical correction only after failure, never reject a valid finite repetition."""
    message=str(error)
    if 'truncated' in message:
        # Diagnose only an already failed response. Do not interrupt valid long text.
        tail=partial[-8192:]
        repeated=bool(re.search(r'(.{1,96}?)\1{31,}$',tail,re.DOTALL))
        cause='truncated_repetition' if repeated else 'truncated_output'
        return cause, ('The previous response exhausted the output limit before completing its JSON object.'
            + (' Its tail repeated the same short span mechanically.' if repeated else '')
            + ' Generate a fresh, complete JSON object. Avoid accidental loops and unnecessary repetition;'
              ' preserve finite repetition and substantive length explicitly required by the user.'
              ' Do not replace a requested artifact with a claim that you produced it.'
              ' If the requested output cannot fit, give an honest useful boundary in the reply instead of an unfinished string.')
    if 'whitespace' in message:
        return 'whitespace_loop','Return the complete requested JSON object without endless padding or whitespace.'
    if isinstance(error,(ValueError,KeyError)):
        return 'invalid_json','Return one complete valid JSON object using the Output Format skeleton. Escape string quotes and newlines; include the actual required content and no outside commentary.'
    return None,None
