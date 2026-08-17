import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..domain.run import TERMINAL_STATES

router = APIRouter()

@router.websocket("/ws/sessions/{run_id}")
async def session_socket(websocket: WebSocket, run_id: str):
    service = websocket.app.state.run_service
    try: public = service.get_run(run_id)
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
