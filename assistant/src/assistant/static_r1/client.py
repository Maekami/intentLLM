"""Profile-driven transport for frozen R1. No policy or schema enters messages."""
import asyncio
import copy
import json
import time

import httpx

from .recovery import recovery_note
from .schemas import JSON_SCHEMAS
from .limits import ModelSlots


def portable_schema(schema):
    """Strict providers require all keys; represent optional keys as nullable."""
    result = copy.deepcopy(schema)
    if result.get('type') == 'object':
        required = result.get('required', [])
        for key, value in result.get('properties', {}).items():
            value = portable_schema(value)
            result['properties'][key] = value if key in required else {
                'anyOf': [value, {'type': 'null'}]}
        result['required'] = list(result.get('properties', {}))
    if 'items' in result:
        result['items'] = portable_schema(result['items'])
    return result


def omit_optional_nulls(value, schema):
    """Decode nullable wire fields back to the original R1 optional contract."""
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        return {key: omit_optional_nulls(item, properties.get(key, {}))
                for key, item in value.items() if item is not None or key in required}
    if isinstance(value, list):
        return [omit_optional_nulls(item, schema.get('items', {})) for item in value]
    return value


class ProfileClient:
    def __init__(self, profile, environment, *, http=None, slots=None, token_counter=None):
        self.profile = profile
        self.generation = profile.generation['assistant'].model_dump(exclude_none=True)
        self.calls = []
        self.slots = slots or ModelSlots(profile.base_url)
        self.token_counter = token_counter
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(600, connect=20),
                                             trust_env=False)
        self._owns_http = http is None
        self.headers = {}
        if profile.provider == 'openrouter':
            if not environment.openrouter_api_key:
                raise ValueError('OPENROUTER_API_KEY is required for R1 OpenRouter profiles')
            self.headers['Authorization'] = 'Bearer ' + environment.openrouter_api_key
        elif environment.vllm_api_key:
            self.headers['Authorization'] = 'Bearer ' + environment.vllm_api_key

    async def _check_context(self, messages):
        if self.profile.provider != 'vllm':
            # Remote providers enforce their own context limit. Never truncate input.
            return
        template = self.profile.reasoning.local_chat_template
        payload = dict(model=self.profile.model_id, messages=messages, add_generation_prompt=True)
        if template is not None:
            payload['chat_template_kwargs'] = template.model_dump()
        result = await self.http.post(self.profile.base_url.removesuffix('/v1') + '/tokenize',
                                      json=payload, headers=self.headers)
        result.raise_for_status()
        count = result.json()['count']
        if count + self.generation['max_completion_tokens'] > 262144:
            raise ValueError('context_overflow: input=' + str(count))

    async def call(self, role, messages, *, json_mode=False, schema_override=None):
        profile = self.profile
        generation = dict(self.generation)
        generation['max_tokens'] = generation.pop('max_completion_tokens')
        payload = dict(model=profile.model_id, messages=messages, **generation)
        if profile.provider == 'openrouter':
            payload['provider'] = profile.routing
            reasoning = profile.reasoning
            if not reasoning.enabled and reasoning.effort is not None:
                payload['reasoning'] = {'effort': reasoning.effort}
            else:
                payload['reasoning'] = {'enabled': reasoning.enabled}
                if reasoning.enabled:
                    payload['reasoning']['exclude'] = reasoning.exclude_from_response
                    if reasoning.effort is not None:
                        payload['reasoning']['effort'] = reasoning.effort
        elif profile.reasoning.local_chat_template is not None:
            payload['chat_template_kwargs'] = profile.reasoning.local_chat_template.model_dump()
        schema = schema_override if schema_override is not None else JSON_SCHEMAS.get(role)
        if json_mode:
            wire_schema = portable_schema(schema) if profile.provider == 'openrouter' else schema
            payload['response_format'] = ({'type': 'json_schema', 'json_schema': {
                'name': role, 'strict': True, 'schema': wire_schema}}
                if schema else {'type': 'json_object'})
        await self._check_context(messages)
        correction = None
        for attempt in range(4):
            start = time.monotonic()
            wire = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
            record = dict(role=role, attempt=attempt + 1)
            if correction:
                wire['messages'] = [*messages, dict(role='user', content=
                    '## Technical Output Recovery\n' + correction[1])]
                record['technical_recovery_reason'] = correction[0]
                await self._check_context(wire['messages'])
            if attempt and json_mode:
                wire['response_format'] = {'type': 'json_object'}
                record['retry_transport'] = 'json_object_after_technical_failure'
            record['request'] = wire
            text = ''
            usage = {}
            finish_reason = None
            try:
                async with self.slots, asyncio.timeout(600):
                    async with self.http.stream('POST', profile.base_url.rstrip('/') +
                        '/chat/completions', json=wire, headers=self.headers) as response:
                        if response.is_error:
                            # Response detail is essential to diagnose schema compatibility;
                            # request headers/credentials are never included in the audit.
                            await response.aread()
                            record['http_status'] = response.status_code
                            record['provider_error'] = response.text[:4000]
                            response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith('data: '):
                                continue
                            if line[6:] == '[DONE]':
                                break
                            raw = json.loads(line[6:])
                            if raw.get('error'):
                                record['provider_error'] = raw['error']
                                raise ValueError('provider streaming error')
                            if raw.get('usage'):
                                usage = raw['usage']
                            for key in ('id', 'provider', 'model'):
                                if raw.get(key):
                                    record[key] = raw[key]
                            for choice in raw.get('choices', []):
                                delta = choice.get('delta') or {}
                                text += delta.get('content') or ''
                                finish_reason = choice.get('finish_reason') or finish_reason
                            if len(text) > 1024 and not text[-1024:].strip():
                                raise ValueError('nonproductive whitespace decoding')
                record.update(response=text, finish_reason=finish_reason, usage=usage,
                    input_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('completion_tokens', 0))
                if finish_reason == 'length':
                    raise ValueError('truncated model output')
                if not text.strip():
                    raise ValueError('empty model output')
                if json_mode:
                    value = json.loads(text)
                    if not isinstance(value, dict):
                        raise ValueError('expected JSON object')
                    if profile.provider == 'openrouter' and schema:
                        value = omit_optional_nulls(value, schema)
                        text = json.dumps(value, ensure_ascii=False)
                return text
            except (httpx.HTTPError, KeyError, ValueError, TimeoutError) as exc:
                record.update(error=f'{type(exc).__name__}: {str(exc)[:250]}',
                              partial_response=text)
                reason, note = recovery_note(exc, text)
                if note:
                    correction = reason, note
                if attempt == 3:
                    raise
                await asyncio.sleep(min(2 ** attempt, 8))
            finally:
                record['usage_complete'] = bool(usage)
                record['elapsed_seconds'] = time.monotonic() - start
                self.calls.append(record)

    async def count_text(self, text):
        if self.token_counter is not None:
            return self.token_counter(text)
        if self.profile.provider != 'vllm':
            # Do not misreport internal JSON/reasoning tokens as visible reply tokens.
            raise ValueError('visible tokenizer not configured for remote model')
        response = await self.http.post(self.profile.base_url.removesuffix('/v1') + '/tokenize',
            json=dict(model=self.profile.model_id, prompt=text, add_special_tokens=False),
            headers=self.headers)
        response.raise_for_status()
        return response.json()['count']

    async def close(self):
        if self._owns_http:
            await self.http.aclose()
