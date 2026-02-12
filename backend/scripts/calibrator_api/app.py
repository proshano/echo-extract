from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .calibration_service import CalibrationOptions, run_feature_test
from .csv_service import CsvServiceError, parse_csv_base64, parse_csv_file
from .hf_service import (
    HuggingFaceServiceError,
    download_gguf_model,
    list_gguf_files,
    search_gguf_models,
)
from .jobs import CalibrationJobStore, JobNotFoundError
from .llama_service import LlamaServiceError, fetch_server_models, llama_server_manager
from .schema_service import (
    SchemaServiceError,
    load_schema,
    load_session_state,
    save_schema,
    save_session_state,
    validate_features,
)


class ApiError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class CsvLoadRequest(BaseModel):
    path: Optional[str] = None
    file_name: Optional[str] = None
    file_base64: Optional[str] = None
    preview_rows: int = Field(default=8, ge=1, le=100)
    id_column: str = ""
    report_column: str = ""


class SchemaLoadRequest(BaseModel):
    path: str


class SchemaSaveRequest(BaseModel):
    path: str
    features: List[Dict[str, Any]]
    schema_name: str = "data_extraction_calibrated"
    missing_default: str = "NA"


class SessionSaveRequest(BaseModel):
    state: Dict[str, Any]


class JudgeConfigRequest(BaseModel):
    enabled: bool = False
    model: str = ""
    instructions: str = ""
    acceptance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class FeatureTestRequest(BaseModel):
    feature: Dict[str, Any]
    report_text: str
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class TestAllRequest(BaseModel):
    features: List[Dict[str, Any]]
    report_text: str
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class TestBatchReportRequest(BaseModel):
    row_number: Optional[int] = Field(default=None, ge=1)
    report_text: str


class ExperimentRunRequest(BaseModel):
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)


class TestBatchRequest(BaseModel):
    features: List[Dict[str, Any]]
    reports: List[TestBatchReportRequest] = Field(default_factory=list)
    llama_url: str = "http://127.0.0.1:8080"
    temperature: float = 0.0
    max_retries: int = Field(default=5, ge=1, le=20)
    model: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    system_instructions: str = ""
    extraction_instructions: str = ""
    reasoning_mode: str = "direct"
    reasoning_instructions: str = ""
    output_instructions: str = ""
    judge: JudgeConfigRequest = Field(default_factory=JudgeConfigRequest)
    experiments: List[ExperimentRunRequest] = Field(default_factory=list)


class ModelListRequest(BaseModel):
    llama_url: str = "http://127.0.0.1:8080"


class LlamaServerStatusRequest(BaseModel):
    llama_url: str = "http://127.0.0.1:8080"


class LlamaServerStartRequest(BaseModel):
    model_path: str
    port: int = Field(default=8080, ge=1, le=65535)
    ctx_size: int = Field(default=8192, ge=256, le=131072)


class LlamaServerEnsureRequest(BaseModel):
    model_path: str
    port: int = Field(default=8080, ge=1, le=65535)
    ctx_size: int = Field(default=8192, ge=256, le=131072)


class HuggingFaceSearchRequest(BaseModel):
    query: str = ""
    limit: int = Field(default=20, ge=1, le=50)


class HuggingFaceFilesRequest(BaseModel):
    repo_id: str


class HuggingFaceDownloadRequest(BaseModel):
    repo_id: str
    file_name: str
    destination_dir: str = ""
    hf_token: str = ""


def _read_source_value(source: Any, key: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def build_calibration_options(request: Any, source_override: Any = None) -> CalibrationOptions:
    source = source_override if source_override is not None else request

    def read(key: str, default: Any = "") -> Any:
        raw_value = _read_source_value(source, key, None)
        if raw_value is None and source is not request:
            raw_value = _read_source_value(request, key, default)
        if raw_value is None:
            return default
        return raw_value

    judge = read("judge", None)
    return CalibrationOptions(
        llama_url=str(read("llama_url", "http://127.0.0.1:8080")).strip(),
        temperature=float(read("temperature", 0.0)),
        max_retries=int(read("max_retries", 5)),
        model=str(read("model", "")).strip(),
        experiment_id=str(read("experiment_id", "")).strip(),
        experiment_name=str(read("experiment_name", "")).strip(),
        system_instructions=str(read("system_instructions", "")).strip(),
        extraction_instructions=str(read("extraction_instructions", "")).strip(),
        reasoning_mode=str(read("reasoning_mode", "direct") or "direct").strip() or "direct",
        reasoning_instructions=str(read("reasoning_instructions", "")).strip(),
        output_instructions=str(read("output_instructions", "")).strip(),
        judge_enabled=bool(_read_source_value(judge, "enabled", False)),
        judge_model=str(_read_source_value(judge, "model", "") or "").strip(),
        judge_instructions=str(_read_source_value(judge, "instructions", "") or "").strip(),
        judge_acceptance_threshold=float(_read_source_value(judge, "acceptance_threshold", 0.6)),
    )


job_store = CalibrationJobStore()


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    try:
        yield
    finally:
        try:
            llama_server_manager.stop()
        except Exception:
            pass


app = FastAPI(title="Data Calibrator API", version="0.1.0", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "null"],
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def api_error_handler(_: Any, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Unexpected server error.",
                "details": {"reason": str(exc)},
            }
        },
    )


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "calibrator_api",
        "timestamp_unix": int(time.time()),
    }


@app.post("/api/csv/load")
async def load_csv_endpoint(request: CsvLoadRequest) -> Dict[str, Any]:
    if not request.path and not request.file_base64:
        raise ApiError(
            status_code=400,
            code="csv_missing_source",
            message="Provide either 'path' or 'file_base64'.",
        )

    try:
        if request.path:
            result = parse_csv_file(
                Path(request.path),
                preview_rows=request.preview_rows,
                requested_id_column=request.id_column,
                requested_report_column=request.report_column,
            )
        else:
            result = parse_csv_base64(
                request.file_base64 or "",
                file_name=request.file_name or "uploaded.csv",
                preview_rows=request.preview_rows,
                requested_id_column=request.id_column,
                requested_report_column=request.report_column,
            )
        return result
    except CsvServiceError as exc:
        raise ApiError(status_code=400, code="csv_load_failed", message=str(exc)) from exc


@app.post("/api/schema/load")
async def load_schema_endpoint(request: SchemaLoadRequest) -> Dict[str, Any]:
    try:
        return load_schema(Path(request.path))
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="schema_load_failed", message=str(exc)) from exc


@app.post("/api/schema/save")
async def save_schema_endpoint(request: SchemaSaveRequest) -> Dict[str, Any]:
    try:
        return save_schema(
            Path(request.path),
            features=request.features,
            schema_name=request.schema_name,
            missing_default=request.missing_default,
        )
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="schema_save_failed", message=str(exc)) from exc


@app.get("/api/session/load")
async def load_session_endpoint() -> Dict[str, Any]:
    try:
        return {"state": load_session_state()}
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="session_load_failed", message=str(exc)) from exc


@app.post("/api/session/save")
async def save_session_endpoint(request: SessionSaveRequest) -> Dict[str, Any]:
    try:
        saved = save_session_state(request.state)
        return {"state": saved}
    except (SchemaServiceError, ValueError, TypeError) as exc:
        raise ApiError(status_code=400, code="session_save_failed", message=str(exc)) from exc


@app.post("/api/models/list")
async def list_models_endpoint(request: ModelListRequest) -> Dict[str, Any]:
    try:
        models = fetch_server_models(request.llama_url)
        return {"llama_url": request.llama_url.strip().rstrip("/"), "models": models}
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="model_list_failed", message=str(exc)) from exc


@app.post("/api/hf/gguf/search")
async def hf_search_gguf_endpoint(request: HuggingFaceSearchRequest) -> Dict[str, Any]:
    try:
        models = search_gguf_models(query=request.query, limit=request.limit)
        return {"models": models}
    except HuggingFaceServiceError as exc:
        raise ApiError(status_code=400, code="hf_search_failed", message=str(exc)) from exc


@app.post("/api/hf/gguf/files")
async def hf_list_gguf_files_endpoint(request: HuggingFaceFilesRequest) -> Dict[str, Any]:
    try:
        files = list_gguf_files(repo_id=request.repo_id)
        return {"repo_id": request.repo_id.strip(), "files": files}
    except HuggingFaceServiceError as exc:
        raise ApiError(status_code=400, code="hf_files_failed", message=str(exc)) from exc


@app.post("/api/hf/gguf/download")
async def hf_download_gguf_endpoint(request: HuggingFaceDownloadRequest) -> Dict[str, Any]:
    try:
        return download_gguf_model(
            repo_id=request.repo_id,
            file_name=request.file_name,
            destination_dir=request.destination_dir,
            hf_token=request.hf_token,
        )
    except HuggingFaceServiceError as exc:
        raise ApiError(status_code=400, code="hf_download_failed", message=str(exc)) from exc


@app.get("/api/llama/local-models")
async def list_local_models_endpoint() -> Dict[str, Any]:
    models, timed_out = llama_server_manager.list_local_models(include_slow_dirs=False)
    return {
        "models": models,
        "timed_out": timed_out,
        "binary_path": llama_server_manager.get_binary_path(),
    }


@app.post("/api/llama/server/status")
async def llama_server_status_endpoint(request: LlamaServerStatusRequest) -> Dict[str, Any]:
    try:
        return llama_server_manager.status(llama_url=request.llama_url)
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_status_failed", message=str(exc)) from exc


@app.post("/api/llama/server/start")
async def llama_server_start_endpoint(request: LlamaServerStartRequest) -> Dict[str, Any]:
    try:
        started = llama_server_manager.start(
            model_path=request.model_path,
            port=request.port,
            ctx_size=request.ctx_size,
        )
        status = llama_server_manager.status(llama_url=f"http://127.0.0.1:{request.port}")
        status.update(started)
        return status
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_start_failed", message=str(exc)) from exc


@app.post("/api/llama/server/ensure")
async def llama_server_ensure_endpoint(request: LlamaServerEnsureRequest) -> Dict[str, Any]:
    try:
        ensured = llama_server_manager.ensure_running(
            model_path=request.model_path,
            port=request.port,
            ctx_size=request.ctx_size,
        )
        status = llama_server_manager.status(llama_url=f"http://127.0.0.1:{request.port}")
        status.update(ensured)
        return status
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_ensure_failed", message=str(exc)) from exc


@app.post("/api/llama/server/stop")
async def llama_server_stop_endpoint(request: LlamaServerStatusRequest) -> Dict[str, Any]:
    llama_server_manager.stop()
    try:
        return llama_server_manager.status(llama_url=request.llama_url)
    except LlamaServiceError as exc:
        raise ApiError(status_code=400, code="llama_stop_failed", message=str(exc)) from exc


@app.post("/api/llama/server/stop-now")
async def llama_server_stop_now_endpoint() -> Dict[str, Any]:
    llama_server_manager.stop()
    return {"stopped": True}


@app.post("/api/test/feature")
async def test_feature_endpoint(request: FeatureTestRequest) -> Dict[str, Any]:
    try:
        validated_feature = validate_features([request.feature])[0]
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_feature", message=str(exc)) from exc

    options = build_calibration_options(request)

    result = run_feature_test(
        feature=validated_feature,
        report_text=request.report_text,
        options=options,
    )
    return result


@app.post("/api/test/all")
async def test_all_endpoint(request: TestAllRequest) -> Dict[str, Any]:
    try:
        validated_features = validate_features(request.features)
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_features", message=str(exc)) from exc

    if not request.report_text.strip():
        raise ApiError(
            status_code=400,
            code="empty_report_text",
            message="report_text must be non-empty.",
        )

    options = build_calibration_options(request)

    job_id = job_store.create_test_all_job(
        features=validated_features,
        report_text=request.report_text,
        options=options,
    )
    return {"job_id": job_id}


@app.post("/api/test/batch")
async def test_batch_endpoint(request: TestBatchRequest) -> Dict[str, Any]:
    try:
        validated_features = validate_features(request.features)
    except SchemaServiceError as exc:
        raise ApiError(status_code=400, code="invalid_features", message=str(exc)) from exc

    if len(request.reports) == 0:
        raise ApiError(
            status_code=400,
            code="empty_reports",
            message="At least one report is required.",
        )
    if len(request.reports) > 20:
        raise ApiError(
            status_code=400,
            code="too_many_reports",
            message="You can test at most 20 reports at a time.",
        )

    report_items: List[Dict[str, Any]] = []
    for report in request.reports:
        report_text = str(report.report_text or "").strip()
        if not report_text:
            continue
        report_items.append(
            {
                "row_number": report.row_number,
                "report_text": report_text,
            }
        )

    if not report_items:
        raise ApiError(
            status_code=400,
            code="empty_reports",
            message="At least one report_text must be non-empty.",
        )

    if request.experiments:
        options_list = [build_calibration_options(request, source_override=experiment) for experiment in request.experiments]
    else:
        options_list = [build_calibration_options(request)]

    try:
        job_id = job_store.create_test_batch_job(
            features=validated_features,
            reports=report_items,
            options_list=options_list,
        )
    except ValueError as exc:
        raise ApiError(status_code=400, code="invalid_reports", message=str(exc)) from exc

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return job_store.get_job(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="job_not_found",
            message=f"Unknown job_id: {job_id}",
        ) from exc


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str) -> Dict[str, Any]:
    try:
        return job_store.cancel_job(job_id)
    except JobNotFoundError as exc:
        raise ApiError(
            status_code=404,
            code="job_not_found",
            message=f"Unknown job_id: {job_id}",
        ) from exc
