import type {
  CsvLoadResponse,
  JudgeConfig,
  Feature,
  FeatureTestResult,
  HfGgufDownloadResponse,
  HfGgufFilesResponse,
  HfGgufSearchResponse,
  JobResponse,
  LlamaServerStatusResponse,
  LocalModelListResponse,
  ReasoningMode,
  SessionState,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 15_000;
const DOWNLOAD_TIMEOUT_MS = 6 * 60 * 60 * 1000;
const ID_COLUMN_CANDIDATES = ["StudyID", "study_id", "id", "patient_id", "record_id"];
const REPORT_COLUMN_CANDIDATES = ["Report", "report_text", "report", "report_body", "narrative"];

type ApiErrorShape = {
  error?: {
    code?: string;
    message?: string;
    details?: unknown;
  };
};

async function readApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as ApiErrorShape;
    return payload.error?.message ?? `Request failed: ${response.status}`;
  } catch {
    return `Request failed: ${response.status}`;
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function fetchWithTimeout(path: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
  } catch (error) {
    if (isAbortError(error)) {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    }
    throw new Error("Unable to reach API service.");
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

async function postJson<TResponse>(path: string, body: unknown, timeoutMs = REQUEST_TIMEOUT_MS): Promise<TResponse> {
  const response = await fetchWithTimeout(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    timeoutMs,
  );

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as TResponse;
}

async function getJson<TResponse>(path: string, timeoutMs = REQUEST_TIMEOUT_MS): Promise<TResponse> {
  const response = await fetchWithTimeout(path, undefined, timeoutMs);
  if (!response.ok) {
    throw new Error(await readApiError(response));
  }
  return (await response.json()) as TResponse;
}

function normalizeColumnName(value: string): string {
  return value.replace("\ufeff", "").trim();
}

function detectDelimiter(sample: string): string {
  const candidates = [",", ";", "\t", "|"];
  const firstLine = sample
    .split(/\r?\n/)
    .find((line) => line.trim().length > 0)
    ?.slice(0, 4096);

  if (!firstLine) {
    return ",";
  }

  let bestDelimiter = ",";
  let bestCount = -1;

  for (const delimiter of candidates) {
    let count = 0;
    let inQuotes = false;

    for (let index = 0; index < firstLine.length; index += 1) {
      const char = firstLine[index];
      const next = firstLine[index + 1];

      if (char === '"') {
        if (inQuotes && next === '"') {
          index += 1;
          continue;
        }
        inQuotes = !inQuotes;
        continue;
      }

      if (!inQuotes && char === delimiter) {
        count += 1;
      }
    }

    if (count > bestCount) {
      bestDelimiter = delimiter;
      bestCount = count;
    }
  }

  return bestDelimiter;
}

function parseCsvRecords(csvText: string, delimiter: string): string[][] {
  const records: string[][] = [];
  let currentField = "";
  let currentRecord: string[] = [];
  let inQuotes = false;

  for (let index = 0; index < csvText.length; index += 1) {
    const char = csvText[index];
    const next = csvText[index + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        currentField += '"';
        index += 1;
        continue;
      }
      inQuotes = !inQuotes;
      continue;
    }

    if (!inQuotes && char === delimiter) {
      currentRecord.push(currentField);
      currentField = "";
      continue;
    }

    if (!inQuotes && (char === "\n" || char === "\r")) {
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      currentRecord.push(currentField);
      records.push(currentRecord);
      currentRecord = [];
      currentField = "";
      continue;
    }

    currentField += char;
  }

  if (currentField.length > 0 || currentRecord.length > 0) {
    currentRecord.push(currentField);
    records.push(currentRecord);
  }

  return records;
}

function pickExistingColumn(headers: string[], candidates: string[]): string {
  const loweredToActual = new Map(headers.map((column) => [column.toLowerCase(), column]));
  for (const candidate of candidates) {
    const match = loweredToActual.get(candidate.toLowerCase());
    if (match) {
      return match;
    }
  }
  return "";
}

function resolveRequestedColumn(headers: string[], requested: string): string {
  const requestedClean = normalizeColumnName(requested);
  if (!requestedClean) {
    return "";
  }
  if (headers.includes(requestedClean)) {
    return requestedClean;
  }

  const lowered = requestedClean.toLowerCase();
  for (const header of headers) {
    if (header.toLowerCase() === lowered) {
      return header;
    }
  }
  return "";
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let index = 0; index < bytes.length; index += chunkSize) {
    const chunk = bytes.subarray(index, index + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  return btoa(binary);
}

export async function healthCheck(): Promise<{ status: string }> {
  return getJson<{ status: string }>("/api/health");
}

export async function searchGgufModels(payload: { query: string; limit: number }): Promise<HfGgufSearchResponse> {
  return postJson<HfGgufSearchResponse>("/api/hf/gguf/search", {
    query: payload.query,
    limit: payload.limit,
  });
}

export async function listGgufFiles(payload: { repoId: string }): Promise<HfGgufFilesResponse> {
  return postJson<HfGgufFilesResponse>("/api/hf/gguf/files", {
    repo_id: payload.repoId,
  });
}

export async function downloadGgufModel(payload: {
  repoId: string;
  fileName: string;
  destinationDir: string;
  hfToken?: string;
}): Promise<HfGgufDownloadResponse> {
  return postJson<HfGgufDownloadResponse>(
    "/api/hf/gguf/download",
    {
      repo_id: payload.repoId,
      file_name: payload.fileName,
      destination_dir: payload.destinationDir,
      hf_token: payload.hfToken ?? "",
    },
    DOWNLOAD_TIMEOUT_MS,
  );
}

export async function listLocalModels(): Promise<LocalModelListResponse> {
  return getJson<LocalModelListResponse>("/api/llama/local-models");
}

export async function ensureLlamaServer(payload: {
  modelPath: string;
  port: number;
  ctxSize: number;
}): Promise<LlamaServerStatusResponse> {
  return postJson<LlamaServerStatusResponse>("/api/llama/server/ensure", {
    model_path: payload.modelPath,
    port: payload.port,
    ctx_size: payload.ctxSize,
  });
}

export async function stopLlamaServer(payload: { llamaUrl: string }): Promise<LlamaServerStatusResponse> {
  return postJson<LlamaServerStatusResponse>("/api/llama/server/stop", {
    llama_url: payload.llamaUrl,
  });
}

export function stopLlamaServerOnPageUnload(): void {
  const url = `${API_BASE}/api/llama/server/stop-now`;
  if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
    if (navigator.sendBeacon(url)) {
      return;
    }
  }

  void fetch(url, { method: "POST", keepalive: true, mode: "cors" }).catch(() => {
    // Ignore unload errors because the page may already be navigating away.
  });
}

export async function loadCsvFromPath(payload: {
  path: string;
  previewRows: number;
  idColumn: string;
  reportColumn: string;
}): Promise<CsvLoadResponse> {
  return postJson<CsvLoadResponse>("/api/csv/load", {
    path: payload.path,
    preview_rows: payload.previewRows,
    id_column: payload.idColumn,
    report_column: payload.reportColumn,
  });
}

export async function loadCsvFromFile(payload: {
  file: File;
  previewRows: number;
  idColumn: string;
  reportColumn: string;
}): Promise<CsvLoadResponse> {
  const bytes = await payload.file.arrayBuffer();
  const fileBase64 = arrayBufferToBase64(bytes);

  return postJson<CsvLoadResponse>("/api/csv/load", {
    file_name: payload.file.name,
    file_base64: fileBase64,
    preview_rows: payload.previewRows,
    id_column: payload.idColumn,
    report_column: payload.reportColumn,
  });
}

export async function loadCsvFromFileLocal(payload: {
  file: File;
  previewRows: number;
  idColumn: string;
  reportColumn: string;
}): Promise<CsvLoadResponse> {
  const csvText = await payload.file.text();
  if (!csvText.trim()) {
    throw new Error("CSV payload is empty.");
  }

  const delimiter = detectDelimiter(csvText);
  const records = parseCsvRecords(csvText, delimiter).filter((record) =>
    record.some((cell) => cell.trim().length > 0),
  );

  const rawHeaders = records[0] ?? [];
  const headers = rawHeaders.map(normalizeColumnName).filter((header) => header.length > 0);
  if (headers.length === 0) {
    throw new Error("No header columns detected.");
  }

  if (new Set(headers).size !== headers.length) {
    throw new Error("Duplicate header names after normalization are not supported.");
  }

  const previewLimit = Math.max(1, Math.min(Math.floor(payload.previewRows || 8), 100));
  const preview: CsvLoadResponse["preview"] = [];
  let rowCount = 0;

  for (const record of records.slice(1)) {
    const values: Record<string, string> = {};
    let hasValue = false;

    for (let index = 0; index < headers.length; index += 1) {
      const value = record[index] ?? "";
      values[headers[index]] = value;
      if (value.trim().length > 0) {
        hasValue = true;
      }
    }

    if (!hasValue) {
      continue;
    }

    rowCount += 1;
    if (preview.length < previewLimit) {
      preview.push({ row_number: rowCount, values });
    }
  }

  const resolvedId = resolveRequestedColumn(headers, payload.idColumn);
  const resolvedReport = resolveRequestedColumn(headers, payload.reportColumn);

  return {
    source: payload.file.name,
    columns: headers,
    row_count: rowCount,
    preview,
    encoding: "client-local",
    delimiter,
    inferred_id_column: resolvedId || pickExistingColumn(headers, ID_COLUMN_CANDIDATES),
    inferred_report_column: resolvedReport || pickExistingColumn(headers, REPORT_COLUMN_CANDIDATES),
  };
}

export async function loadSchema(path: string): Promise<{ features: Feature[]; schema_path: string }> {
  return postJson<{ features: Feature[]; schema_path: string }>("/api/schema/load", { path });
}

export async function saveSchema(payload: {
  path: string;
  features: Feature[];
}): Promise<{ schema_path: string }> {
  return postJson<{ schema_path: string }>("/api/schema/save", {
    path: payload.path,
    features: payload.features,
  });
}

export async function loadSession(): Promise<SessionState> {
  const response = await getJson<{ state: SessionState }>("/api/session/load");
  return response.state;
}

export async function saveSession(state: SessionState): Promise<void> {
  await postJson<{ state: SessionState }>("/api/session/save", { state });
}

export async function testFeature(payload: {
  feature: Feature;
  reportText: string;
  llamaUrl: string;
  temperature: number;
  maxRetries: number;
  model: string;
  experimentId: string;
  experimentName: string;
  systemInstructions: string;
  extractionInstructions: string;
  reasoningMode: ReasoningMode;
  reasoningInstructions: string;
  outputInstructions: string;
  judge: JudgeConfig;
}): Promise<FeatureTestResult> {
  return postJson<FeatureTestResult>("/api/test/feature", {
    feature: payload.feature,
    report_text: payload.reportText,
    llama_url: payload.llamaUrl,
    temperature: payload.temperature,
    max_retries: payload.maxRetries,
    model: payload.model,
    experiment_id: payload.experimentId,
    experiment_name: payload.experimentName,
    system_instructions: payload.systemInstructions,
    extraction_instructions: payload.extractionInstructions,
    reasoning_mode: payload.reasoningMode,
    reasoning_instructions: payload.reasoningInstructions,
    output_instructions: payload.outputInstructions,
    judge: {
      enabled: payload.judge.enabled,
      model: payload.judge.model,
      instructions: payload.judge.instructions,
      acceptance_threshold: payload.judge.acceptance_threshold,
    },
  });
}

export async function testAll(payload: {
  features: Feature[];
  reportText: string;
  llamaUrl: string;
  temperature: number;
  maxRetries: number;
  model: string;
  experimentId: string;
  experimentName: string;
  systemInstructions: string;
  extractionInstructions: string;
  reasoningMode: ReasoningMode;
  reasoningInstructions: string;
  outputInstructions: string;
  judge: JudgeConfig;
}): Promise<{ job_id: string }> {
  return postJson<{ job_id: string }>("/api/test/all", {
    features: payload.features,
    report_text: payload.reportText,
    llama_url: payload.llamaUrl,
    temperature: payload.temperature,
    max_retries: payload.maxRetries,
    model: payload.model,
    experiment_id: payload.experimentId,
    experiment_name: payload.experimentName,
    system_instructions: payload.systemInstructions,
    extraction_instructions: payload.extractionInstructions,
    reasoning_mode: payload.reasoningMode,
    reasoning_instructions: payload.reasoningInstructions,
    output_instructions: payload.outputInstructions,
    judge: {
      enabled: payload.judge.enabled,
      model: payload.judge.model,
      instructions: payload.judge.instructions,
      acceptance_threshold: payload.judge.acceptance_threshold,
    },
  });
}

export async function testBatch(payload: {
  features: Feature[];
  reports: Array<{ rowNumber: number | null; reportText: string }>;
  llamaUrl: string;
  temperature: number;
  maxRetries: number;
  model: string;
  experimentId: string;
  experimentName: string;
  systemInstructions: string;
  extractionInstructions: string;
  reasoningMode: ReasoningMode;
  reasoningInstructions: string;
  outputInstructions: string;
  judge: JudgeConfig;
  experiments?: Array<{
    id: string;
    name: string;
    systemInstructions: string;
    extractionInstructions: string;
    reasoningMode: ReasoningMode;
    reasoningInstructions: string;
    outputInstructions: string;
    judge: JudgeConfig;
  }>;
}): Promise<{ job_id: string }> {
  return postJson<{ job_id: string }>("/api/test/batch", {
    features: payload.features,
    reports: payload.reports.map((report) => ({
      row_number: report.rowNumber,
      report_text: report.reportText,
    })),
    llama_url: payload.llamaUrl,
    temperature: payload.temperature,
    max_retries: payload.maxRetries,
    model: payload.model,
    experiment_id: payload.experimentId,
    experiment_name: payload.experimentName,
    system_instructions: payload.systemInstructions,
    extraction_instructions: payload.extractionInstructions,
    reasoning_mode: payload.reasoningMode,
    reasoning_instructions: payload.reasoningInstructions,
    output_instructions: payload.outputInstructions,
    judge: {
      enabled: payload.judge.enabled,
      model: payload.judge.model,
      instructions: payload.judge.instructions,
      acceptance_threshold: payload.judge.acceptance_threshold,
    },
    experiments: (payload.experiments ?? []).map((experiment) => ({
      experiment_id: experiment.id,
      experiment_name: experiment.name,
      system_instructions: experiment.systemInstructions,
      extraction_instructions: experiment.extractionInstructions,
      reasoning_mode: experiment.reasoningMode,
      reasoning_instructions: experiment.reasoningInstructions,
      output_instructions: experiment.outputInstructions,
      judge: {
        enabled: experiment.judge.enabled,
        model: experiment.judge.model,
        instructions: experiment.judge.instructions,
        acceptance_threshold: experiment.judge.acceptance_threshold,
      },
    })),
  });
}

export async function getJob(jobId: string): Promise<JobResponse> {
  return getJson<JobResponse>(`/api/jobs/${jobId}`);
}

export async function cancelJob(jobId: string): Promise<void> {
  await postJson(`/api/jobs/${jobId}/cancel`, {});
}
