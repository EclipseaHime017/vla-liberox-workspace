"""Run lifecycle, frames, streams, and artifact endpoints."""

import asyncio
import mimetypes

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from ..domain.run import TERMINAL_STATES
from .dependencies import http_error, service
from .models import CreateBranchRequest, DeleteSessionRequest

router = APIRouter(prefix="/api", tags=["runs"])

@router.get("/bootstrap")
async def bootstrap(request: Request): return service(request).bootstrap()

@router.get("/sessions")
@router.get("/runs")
async def runs(request: Request): return service(request).list_runs()

@router.get("/sessions/{run_id}")
async def run(run_id: str, request: Request):
    try: return service(request).get_run(run_id)
    except Exception as exc: raise http_error(exc) from exc

@router.post("/sessions/{run_id}/stop")
async def stop(run_id: str, request: Request):
    try: return service(request).stop(run_id)
    except Exception as exc: raise http_error(exc) from exc

@router.post("/sessions/{run_id}/branches", status_code=201)
async def branch(run_id: str, body: CreateBranchRequest, request: Request):
    try:
        return service(request).create_branch(
            run_id, body.resume_step, body.control_mode, body.open_loop_steps,
            translation_gain=body.translation_gain, rotation_gain=body.rotation_gain,
        )
    except Exception as exc: raise http_error(exc) from exc

@router.delete("/sessions/{run_id}")
async def delete(run_id: str, body: DeleteSessionRequest, request: Request):
    try:
        service(request).delete(run_id, body.confirm_session_id)
        return {"deleted": run_id}
    except Exception as exc: raise http_error(exc) from exc

@router.get("/sessions/{run_id}/frames/{step}")
async def frame(run_id: str, step: int, request: Request):
    try:
        content = await asyncio.to_thread(service(request).frame, run_id, step)
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as exc: raise http_error(exc) from exc

@router.get("/sessions/{run_id}/frames/{step}/state")
async def frame_state(run_id: str, step: int, request: Request):
    try: return service(request).frame_state(run_id, step)
    except Exception as exc: raise http_error(exc) from exc

@router.get("/sessions/{run_id}/stream.mjpeg")
async def stream(run_id: str, request: Request):
    run_service = service(request)
    try: run_service.get_run(run_id)
    except Exception as exc: raise http_error(exc) from exc

    async def generate():
        version = -1
        terminal_empty_cycles = 0
        while not await request.is_disconnected():
            try: public = run_service.get_run(run_id)
            except KeyError: break
            jpeg, current_version = run_service.latest_frame(run_id)
            if jpeg is not None and current_version != version:
                version = current_version
                terminal_empty_cycles = 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n" + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii") + jpeg + b"\r\n")
            elif public["status"] in TERMINAL_STATES:
                terminal_empty_cycles += 1
                if terminal_empty_cycles > 10: break
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

@router.get("/sessions/{run_id}/artifacts/{name:path}")
async def artifact(run_id: str, name: str, request: Request):
    try:
        path = service(request).artifact(run_id, name)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, headers={"Content-Disposition": f'inline; filename="{path.name}"', "Cache-Control": "no-store"})
    except Exception as exc: raise http_error(exc) from exc
