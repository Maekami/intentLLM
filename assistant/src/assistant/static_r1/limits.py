"""One bounded generation pool per endpoint and asyncio event loop."""
import asyncio
from weakref import WeakKeyDictionary

_POOLS = WeakKeyDictionary()

class ModelSlots:
    def __init__(self, endpoint, limit=24):
        self.endpoint = endpoint
        self.limit = limit

    def semaphore(self):
        loop = asyncio.get_running_loop()
        pools = _POOLS.setdefault(loop, {})
        key = (self.endpoint, self.limit)
        if key not in pools:
            pools[key] = asyncio.Semaphore(self.limit)
        return pools[key]

    async def __aenter__(self):
        await self.semaphore().acquire()

    async def __aexit__(self, *exc):
        self.semaphore().release()
