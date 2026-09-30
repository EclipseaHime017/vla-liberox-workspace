import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..domain.run import TERMINAL_STATES
from ..services.storage_maintenance import storage_thread

router = APIRouter()


@asynccontextmanager
async def storage_access(websocket):
    gate = getattr(websocket.app.state, "storage_maintenance", None)
    if gate is None:
        yield
    else:
        async with gate.request():
            yield


@router.websocket("/ws/jobs/{job_id}")
async def job_socket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    service = getattr(websocket.app.state, "offline_job_service", None)
    if service is None:
        await websocket.send_json({"type": "error", "message": "Offline job service unavailable"})
        await websocket.close(code=1011)
        return
    offset = 0
    try:
        while True:
            if getattr(websocket.app.state, "storage_maintenance", None) and websocket.app.state.storage_maintenance.busy:
                await asyncio.sleep(0.5)
                continue
            async with storage_access(websocket):
                job = await storage_thread(service.get, job_id)
                logs = await storage_thread(service.logs, job_id, offset)
            offset = logs["next_offset"]
            await websocket.send_json({"type": "job", "job": job, "logs": logs})
            if job["status"] in {"COMPLETED", "FAILED", "CANCELED"} and not logs["text"]:
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0.5)
    except (WebSocketDisconnect, KeyError, RuntimeError):
        return

@router.websocket("/ws/sessions/{run_id}")
async def session_socket(websocket: WebSocket, run_id: str):
    service = websocket.app.state.run_service
    try:
        async with storage_access(websocket):
            public = service.get_run(run_id)
    except RuntimeError:
        await websocket.close(code=1013, reason="Storage maintenance in progress")
        return
    except KeyError:
        await websocket.close(code=4404, reason="Run not found")
        return
    await websocket.accept()
    is_manual = public.get("control_mode") == "manual" and not public.get("legacy")
    if is_manual:
        try: service.manual_connect(run_id)
        except Exception:
            await websocket.close(code=4409, reason="Manual controller unavailable")
            return
    try:
        while True:
            gate = getattr(websocket.app.state, "storage_maintenance", None)
            if gate and gate.busy:
                await asyncio.sleep(0.1)
                continue
            public = service.get_run(run_id)
            await websocket.send_json({"type": "session", "session": public})
            if public["status"] in TERMINAL_STATES: return
            try: message = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError: continue
            payload = json.loads(message)
            if payload.get("type") == "heartbeat":
                continue
            if payload.get("type") == "manual_settings" and is_manual:
                service.manual_settings(run_id, payload.get("translation_gain"), payload.get("rotation_gain"))
                continue
            raise ValueError(f"Unsupported WebSocket message type: {payload.get('type')!r}")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        if is_manual:
            try: service.manual_disconnect(run_id)
            except Exception: pass
