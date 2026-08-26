from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from interaction_pipeline.core import record_setup_failure, run_interaction
from interaction_pipeline.prepare import PreparedPipeline


@dataclass
class LiveRun:
    events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, event: dict[str, Any]) -> None:
        async with self.condition:
            self.events.append(event)
            if event.get("event_type") in {"run_completed", "run_failed"}:
                self.done = True
            self.condition.notify_all()

    async def stream(self):
        index = 0
        while True:
            async with self.condition:
                await self.condition.wait_for(
                    lambda current_index=index: current_index < len(self.events) or self.done
                )
                pending = self.events[index:]
                index = len(self.events)
                done = self.done
            for event in pending:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if done and index >= len(self.events):
                return


def create_app(
    prepared: PreparedPipeline,
    *,
    sample_id: str,
    output_root: str | Path,
) -> FastAPI:
    app = FastAPI(title="Reason-DAG Interaction Demo")
    live_runs: dict[str, LiveRun] = {}
    server_dir = Path(output_root) / (
        "demo_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    )
    server_dir.mkdir(parents=True, exist_ok=False)
    sample = prepared.dataset.load_sample_by_id(sample_id)

    @app.get("/", response_class=HTMLResponse)
    async def panel() -> str:
        return _panel_html(
            sample_id=sample_id,
            simulator_model=prepared.simulator_config.models["controller"],
            assistant_model=prepared.assistant_config.models["assistant"],
            baseline=prepared.assistant_config.components.baseline,
        )

    @app.post("/api/runs")
    async def start_run() -> dict[str, str]:
        live_id = uuid.uuid4().hex
        live = LiveRun()
        live_runs[live_id] = live
        asyncio.create_task(_execute(live))
        return {"run_id": live_id}

    async def _execute(live: LiveRun) -> None:
        try:
            built = prepared.build(
                sample,
                output_dir=server_dir,
                seed=prepared.config.run.seed,
            )
            await run_interaction(
                episode=built.episode,
                assistant=built.assistant,
                audit=built.audit,
                event_sink=live.publish,
            )
        except Exception as exc:  # noqa: BLE001 - report startup/build failures over SSE
            result = record_setup_failure(
                sample_id=sample_id,
                output_dir=server_dir,
                error=exc,
                config_snapshot={"pipeline": prepared.config.model_dump(mode="json")},
            )
            await live.publish(
                {
                    "event_type": "run_failed",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "sample_id": sample_id,
                    "turn_index": 0,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "run_dir": str(result.run_dir),
                }
            )

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        live = live_runs.get(run_id)
        if live is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return StreamingResponse(
            live.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _panel_html(*, sample_id: str, simulator_model: str, assistant_model: str, baseline: str) -> str:
    settings = json.dumps(
        {
            "sampleId": sample_id,
            "simulatorModel": Path(simulator_model).stem,
            "assistantModel": Path(assistant_model).stem,
            "baseline": baseline,
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Simulator × Assistant</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0d1b2d; --line:#203651;
      --user:#163b5c; --assistant:#253850; --cyan:#59d8ff; --green:#6ee7b7; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; background:radial-gradient(circle at top,#132945,var(--bg) 55%);
      color:#e8f1fb; font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1040px,calc(100% - 28px)); margin:28px auto; }}
    header {{ display:flex; gap:18px; align-items:flex-end; justify-content:space-between; margin-bottom:18px; }}
    h1 {{ margin:0; font-size:clamp(24px,4vw,38px); letter-spacing:-.04em; }}
    .meta {{ color:#91a8c2; font-size:13px; }}
    button {{ border:1px solid #397698; background:#123954; color:white; border-radius:10px;
      padding:10px 16px; cursor:pointer; font-weight:700; }}
    button:disabled {{ opacity:.5; cursor:wait; }}
    #status {{ border:1px solid var(--line); background:rgba(13,27,45,.88); border-radius:14px;
      padding:12px 16px; margin-bottom:14px; color:#a9bdd2; }}
    #status.live {{ color:var(--green); }} #status.error {{ color:#fca5a5; }}
    #conversation {{ border:1px solid var(--line); background:rgba(7,17,31,.72); border-radius:18px;
      min-height:62vh; padding:20px; display:flex; flex-direction:column; gap:14px; }}
    .message {{ max-width:84%; padding:14px 16px; border:1px solid var(--line); border-radius:15px;
      white-space:pre-wrap; overflow-wrap:anywhere; box-shadow:0 10px 28px rgba(0,0,0,.15); }}
    .message.user {{ align-self:flex-start; background:var(--user); border-top-left-radius:4px; }}
    .message.assistant {{ align-self:flex-end; background:var(--assistant); border-top-right-radius:4px; }}
    .role {{ display:block; margin-bottom:6px; text-transform:uppercase; letter-spacing:.12em;
      font-size:11px; font-weight:800; color:var(--cyan); }}
    .assistant .role {{ color:var(--green); }}
    .empty {{ margin:auto; color:#66819c; }}
  </style>
</head>
<body><main>
  <header><div><h1>Simulator × Assistant</h1><div class="meta" id="meta"></div></div>
    <button id="restart">重新运行</button></header>
  <div id="status">准备启动……</div>
  <section id="conversation"><div class="empty">等待第一条完整回复</div></section>
</main>
<script>
const SETTINGS={settings};
const statusEl=document.querySelector('#status');
const conversation=document.querySelector('#conversation');
const restart=document.querySelector('#restart');
document.querySelector('#meta').textContent=`${{SETTINGS.sampleId}} · ${{SETTINGS.baseline}} · ${{SETTINGS.assistantModel}}`;
let stream=null;
function setStatus(text, cls='') {{ statusEl.textContent=text; statusEl.className=cls; }}
function addMessage(event) {{
  document.querySelector('.empty')?.remove();
  const box=document.createElement('article'); box.className=`message ${{event.role}}`;
  const role=document.createElement('span'); role.className='role';
  role.textContent=event.role==='user'?'Simulator / User':'Assistant';
  const body=document.createElement('div'); body.textContent=event.content;
  box.append(role,body); conversation.append(box); box.scrollIntoView({{behavior:'smooth',block:'end'}});
}}
async function start() {{
  restart.disabled=true; if(stream) stream.close(); conversation.innerHTML='<div class="empty">等待第一条完整回复</div>';
  setStatus('正在创建运行……','live');
  try {{
    const response=await fetch('/api/runs',{{method:'POST'}}); const data=await response.json();
    stream=new EventSource(`/api/runs/${{data.run_id}}/events`);
    stream.onmessage=(message)=>{{
      const event=JSON.parse(message.data);
      if(event.event_type==='run_started') setStatus('运行中：按完整回复实时更新','live');
      if(event.event_type==='message') addMessage(event);
      if(event.event_type==='run_completed') {{ setStatus(`已完成 · 审计目录：${{event.run_dir}}`); restart.disabled=false; stream.close(); }}
      if(event.event_type==='run_failed') {{ setStatus(`失败：${{event.error}} · ${{event.run_dir}}`,'error'); restart.disabled=false; stream.close(); }}
    }};
    stream.onerror=()=>{{ if(restart.disabled) setStatus('事件流连接中断','error'); restart.disabled=false; }};
  }} catch(error) {{ setStatus(`启动失败：${{error}}`,'error'); restart.disabled=false; }}
}}
restart.addEventListener('click',start); start();
</script></body></html>"""
