import { type ChangeEvent, type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import {
  cancelJob,
  downloadGgufModel,
  ensureLlamaServer,
  getJob,
  healthCheck,
  listGgufFiles,
  listLocalModels,
  loadCsvFromFile,
  loadCsvFromFileLocal,
  loadCsvFromPath,
  loadSession,
  searchGgufModels,
  saveSession,
  stopLlamaServer,
  stopLlamaServerOnPageUnload,
  testBatch,
  testFeature,
} from "./api";
import type {
  CsvLoadResponse,
  ExperimentProfile,
  Feature,
  FeatureTestResult,
  HuggingFaceGgufFile,
  HuggingFaceGgufModel,
  JudgeConfig,
  JobResponse,
  LlamaServerStatusResponse,
  ReasoningMode,
  SessionState,
} from "./types";

const DEFAULT_FEATURE: Feature = {
  name: "feature_1",
  description: "",
  missing_value_rule: "NA",
  prompt: "",
  type_hint: "string",
};

const DEFAULT_EXPERIMENT_ID = "baseline_experiment";
const JUDGE_MODEL_USE_EXTRACTION = "__use_extraction_model__";

const DEFAULT_EXPERIMENT: ExperimentProfile = {
  id: DEFAULT_EXPERIMENT_ID,
  name: "Baseline",
  system_instructions: "",
  extraction_instructions: "",
  reasoning_mode: "direct",
  reasoning_instructions: "",
  output_instructions: "",
  judge: {
    enabled: false,
    model: "",
    instructions: "",
    acceptance_threshold: 0.6,
  },
};

const DEFAULT_SESSION: SessionState = {
  version: 1,
  csv_path: "",
  csv_cache: null,
  schema_path: "",
  llama_url: "http://127.0.0.1:8080",
  temperature: 0,
  max_retries: 5,
  id_column: "",
  report_column: "",
  sample_index: 1,
  server_model: "",
  features: [cloneFeature(DEFAULT_FEATURE)],
  experiment_profiles: [cloneExperiment(DEFAULT_EXPERIMENT)],
  active_experiment_id: DEFAULT_EXPERIMENT.id,
  active_experiment_ids: [DEFAULT_EXPERIMENT.id],
};

type TabKey = "setup" | "schema" | "experiments" | "test";
type ToastKind = "success" | "error" | "info";

type Toast = {
  id: number;
  message: string;
  kind: ToastKind;
};

type BatchReportItem = {
  rowNumber: number | null;
  reportText: string;
};

type RunHistoryItem = {
  jobId: string;
  experimentNames: string;
  status: string;
  startedAtUnix: number;
  finishedAtUnix: number | null;
  reportsReviewed: number;
  checksTotal: number;
  checksCompleted: number;
  okCount: number;
  llmErrorCount: number;
  parseErrorCount: number;
  judgeAccepted: number;
  judgeRejected: number;
  judgeUncertain: number;
  judgeErrors: number;
};

const TYPE_HINT_OPTIONS = [
  { value: "string", label: "Short text" },
  { value: "text", label: "Long text" },
  { value: "numeric", label: "Number" },
  { value: "numeric_or_NA", label: "Number or NA" },
  { value: "integer", label: "Integer" },
  { value: "integer_or_NA", label: "Integer or NA" },
  { value: "boolean", label: "True / False" },
  { value: "date", label: "Date" },
];

const REASONING_MODE_OPTIONS: Array<{ value: ReasoningMode; label: string }> = [
  { value: "direct", label: "Direct extraction" },
  { value: "plan_then_extract", label: "Plan then extract" },
  { value: "react_style", label: "ReAct-style internal loop" },
  { value: "custom", label: "Custom strategy" },
];

type SchemaFileHandle = {
  name: string;
  getFile: () => Promise<File>;
  createWritable: () => Promise<{
    write: (content: string) => Promise<void>;
    close: () => Promise<void>;
  }>;
};

type FilePickerWindow = Window & {
  showOpenFilePicker?: (options?: {
    multiple?: boolean;
    excludeAcceptAllOption?: boolean;
    types?: Array<{ description?: string; accept: Record<string, string[]> }>;
  }) => Promise<SchemaFileHandle[]>;
  showSaveFilePicker?: (options?: {
    suggestedName?: string;
    excludeAcceptAllOption?: boolean;
    types?: Array<{ description?: string; accept: Record<string, string[]> }>;
  }) => Promise<SchemaFileHandle>;
};

const SCHEMA_PICKER_TYPES = [
  {
    description: "JSON feature set",
    accept: {
      "application/json": [".json"],
    },
  },
];

function cloneFeature(feature: Feature): Feature {
  return {
    name: feature.name,
    description: feature.description,
    missing_value_rule: feature.missing_value_rule,
    prompt: feature.prompt,
    allowed_values: feature.allowed_values ? [...feature.allowed_values] : undefined,
    type_hint: feature.type_hint,
  };
}

function clampJudgeThreshold(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.6;
  }
  return Math.max(0, Math.min(1, value));
}

function cloneExperiment(experiment: ExperimentProfile): ExperimentProfile {
  return {
    id: experiment.id,
    name: experiment.name,
    system_instructions: experiment.system_instructions,
    extraction_instructions: experiment.extraction_instructions,
    reasoning_mode: experiment.reasoning_mode,
    reasoning_instructions: experiment.reasoning_instructions,
    output_instructions: experiment.output_instructions,
    judge: {
      enabled: Boolean(experiment.judge?.enabled),
      model: experiment.judge?.model ?? "",
      instructions: experiment.judge?.instructions ?? "",
      acceptance_threshold: clampJudgeThreshold(experiment.judge?.acceptance_threshold ?? 0.6),
    },
  };
}

function normalizeExperimentList(
  experiments: ExperimentProfile[] | undefined | null,
): ExperimentProfile[] {
  if (!Array.isArray(experiments) || experiments.length === 0) {
    return [cloneExperiment(DEFAULT_EXPERIMENT)];
  }

  const deduped = new Map<string, ExperimentProfile>();
  for (const [index, raw] of experiments.entries()) {
    if (!raw || typeof raw !== "object") {
      continue;
    }
    const id = String(raw.id ?? "").trim() || `experiment_${index + 1}`;
    const reasoningMode = String(raw.reasoning_mode ?? "direct").trim() as ReasoningMode;
    const normalizedReasoningMode: ReasoningMode = REASONING_MODE_OPTIONS.some(
      (option) => option.value === reasoningMode,
    )
      ? reasoningMode
      : "direct";

    const safeExperiment: ExperimentProfile = cloneExperiment({
      id,
      name: String(raw.name ?? "").trim() || `Experiment ${index + 1}`,
      system_instructions: String(raw.system_instructions ?? "").trim(),
      extraction_instructions: String(raw.extraction_instructions ?? "").trim(),
      reasoning_mode: normalizedReasoningMode,
      reasoning_instructions: String(raw.reasoning_instructions ?? "").trim(),
      output_instructions: String(raw.output_instructions ?? "").trim(),
      judge: {
        enabled: Boolean(raw.judge?.enabled),
        model: String(raw.judge?.model ?? "").trim(),
        instructions: String(raw.judge?.instructions ?? "").trim(),
        acceptance_threshold: clampJudgeThreshold(Number(raw.judge?.acceptance_threshold ?? 0.6)),
      },
    });

    if (!deduped.has(safeExperiment.id)) {
      deduped.set(safeExperiment.id, safeExperiment);
    }
  }

  const normalized = [...deduped.values()];
  if (normalized.length === 0) {
    return [cloneExperiment(DEFAULT_EXPERIMENT)];
  }
  return normalized;
}

function normalizeActiveExperimentIds(
  rawIds: unknown,
  experiments: ExperimentProfile[],
  fallbackId: string,
): string[] {
  const knownIds = new Set(experiments.map((profile) => profile.id));
  const deduped = new Set<string>();

  if (Array.isArray(rawIds)) {
    for (const rawId of rawIds) {
      const experimentId = String(rawId ?? "").trim();
      if (!experimentId || !knownIds.has(experimentId)) {
        continue;
      }
      deduped.add(experimentId);
    }
  }

  if (deduped.size > 0) {
    return [...deduped];
  }
  if (fallbackId && knownIds.has(fallbackId)) {
    return [fallbackId];
  }
  return experiments[0]?.id ? [experiments[0].id] : [];
}

function normalizeFeatureList(features: Feature[] | undefined | null): Feature[] {
  if (!Array.isArray(features) || features.length === 0) {
    return [cloneFeature(DEFAULT_FEATURE)];
  }
  return features
    .filter((f): f is Feature => f != null && typeof f === "object" && typeof (f as Feature).name === "string")
    .map((feature) => cloneFeature(feature as Feature));
}

function parsePortFromLlamaUrl(value: string): number {
  try {
    const parsed = new URL(value);
    if (parsed.port) {
      const port = Number(parsed.port);
      if (Number.isFinite(port) && port >= 1 && port <= 65535) {
        return port;
      }
    }
  } catch {
    // Ignore invalid URL and return default below.
  }
  return 8080;
}

function formatModelSize(sizeGb: number | null): string {
  if (typeof sizeGb !== "number" || !Number.isFinite(sizeGb)) {
    return "size unknown";
  }
  return `${sizeGb.toFixed(2)} GB`;
}

function resolveReportTextFromValues(values: Record<string, string>, reportColumn: string): string {
  if (values[reportColumn] !== undefined) {
    return values[reportColumn] ?? "";
  }

  const matchedColumn = Object.keys(values).find(
    (column) => column.toLowerCase() === reportColumn.toLowerCase(),
  );
  if (!matchedColumn) {
    return "";
  }
  return values[matchedColumn] ?? "";
}

function truncateText(value: string, limit: number): string {
  const clean = value.trim();
  if (clean.length <= limit) {
    return clean;
  }
  return `${clean.slice(0, limit).trimEnd()}...`;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function normalizeSchemaFeature(rawFeature: unknown, index: number): Feature {
  if (!rawFeature || typeof rawFeature !== "object") {
    throw new Error(`Feature #${index + 1} is not a valid object.`);
  }

  const feature = rawFeature as Record<string, unknown>;
  const name = String(feature.name ?? "").trim();
  if (!name) {
    throw new Error(`Feature #${index + 1} is missing a name.`);
  }

  const description = String(feature.description ?? "").trim();
  const missingValueRule = String(feature.missing_value_rule ?? "NA").trim() || "NA";
  const prompt = String(feature.prompt ?? "").trim();

  const allowedValues = Array.isArray(feature.allowed_values)
    ? feature.allowed_values.map((value) => String(value).trim()).filter(Boolean)
    : undefined;
  const typeHint = String(feature.type_hint ?? "").trim();

  return {
    name,
    description,
    missing_value_rule: missingValueRule,
    prompt,
    allowed_values: allowedValues && allowedValues.length > 0 ? allowedValues : undefined,
    type_hint: typeHint || (allowedValues && allowedValues.length > 0 ? undefined : "string"),
  };
}

function parseSchemaFeaturesFromJson(content: string): Feature[] {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(content);
  } catch (error) {
    throw new Error(`Invalid JSON: ${getErrorMessage(error)}`);
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("Feature set must be a JSON object.");
  }

  const payload = parsed as { features?: unknown };
  if (!Array.isArray(payload.features) || payload.features.length === 0) {
    throw new Error("Feature set must include a non-empty 'features' array.");
  }

  const namesSeen = new Set<string>();
  const normalized = payload.features.map((feature, index) => {
    const nextFeature = normalizeSchemaFeature(feature, index);
    if (namesSeen.has(nextFeature.name)) {
      throw new Error(`Duplicate feature name: ${nextFeature.name}`);
    }
    namesSeen.add(nextFeature.name);
    return nextFeature;
  });

  return normalizeFeatureList(normalized);
}

function buildSchemaDocument(features: Feature[]): string {
  const normalizedFeatures = features.map((feature) => {
    const name = feature.name.trim();
    const description = feature.description.trim();
    const missingValueRule = feature.missing_value_rule.trim() || "NA";
    const prompt = feature.prompt.trim();
    const allowedValues = Array.isArray(feature.allowed_values)
      ? feature.allowed_values.map((value) => value.trim()).filter(Boolean)
      : [];

    if (allowedValues.length > 0) {
      return {
        name,
        description,
        missing_value_rule: missingValueRule,
        prompt,
        allowed_values: allowedValues,
      };
    }

    return {
      name,
      description,
      missing_value_rule: missingValueRule,
      prompt,
      type_hint: (feature.type_hint ?? "string").trim() || "string",
    };
  });

  return JSON.stringify(
    {
      schema_name: "data_extraction_calibrated",
      missing_default: "NA",
      features: normalizedFeatures,
    },
    null,
    2,
  );
}

function normalizeCsvCache(raw: unknown): CsvLoadResponse | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const payload = raw as Partial<CsvLoadResponse>;
  if (!Array.isArray(payload.columns) || !Array.isArray(payload.preview)) {
    return null;
  }

  return {
    source: String(payload.source ?? ""),
    columns: payload.columns.map((column) => String(column)),
    row_count: Number(payload.row_count ?? 0),
    preview: payload.preview
      .filter((row) => row && typeof row === "object")
      .map((row) => ({
        row_number: Number((row as { row_number?: unknown }).row_number ?? 0),
        values:
          typeof (row as { values?: unknown }).values === "object" &&
          (row as { values?: unknown }).values !== null
            ? Object.fromEntries(
                Object.entries((row as { values: Record<string, unknown> }).values).map(([key, value]) => [
                  key,
                  String(value ?? ""),
                ]),
              )
            : {},
      }))
      .filter((row) => row.row_number >= 1),
    encoding: String(payload.encoding ?? ""),
    delimiter: String(payload.delimiter ?? ","),
    inferred_id_column: String(payload.inferred_id_column ?? ""),
    inferred_report_column: String(payload.inferred_report_column ?? ""),
  };
}

function createExperimentId(name: string, position: number): string {
  const cleaned = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned ? `${cleaned}_${Date.now()}` : `experiment_${position}_${Date.now()}`;
}

function countJudgeOutcome(results: FeatureTestResult[]): {
  accepted: number;
  rejected: number;
  uncertain: number;
  judgeError: number;
} {
  const summary = { accepted: 0, rejected: 0, uncertain: 0, judgeError: 0 };
  for (const result of results) {
    const judgeStatus = result.judge_result?.status;
    if (judgeStatus === "accepted") {
      summary.accepted += 1;
    } else if (judgeStatus === "rejected") {
      summary.rejected += 1;
    } else if (judgeStatus === "uncertain") {
      summary.uncertain += 1;
    } else if (judgeStatus === "judge_error") {
      summary.judgeError += 1;
    }
  }
  return summary;
}

function resolveJudgeConfigForRun(judge: JudgeConfig): JudgeConfig {
  const selectedJudgeModel = String(judge.model ?? "").trim();
  if (!judge.enabled) {
    return { ...judge, enabled: false, model: "" };
  }
  if (!selectedJudgeModel) {
    return { ...judge, enabled: false, model: "" };
  }
  if (selectedJudgeModel === JUDGE_MODEL_USE_EXTRACTION) {
    return { ...judge, enabled: true, model: "" };
  }
  return { ...judge, enabled: true, model: selectedJudgeModel };
}

function csvEscape(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function buildCsvContent(headers: string[], rows: Array<Record<string, unknown>>): string {
  const lines = [headers.map(csvEscape).join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  return lines.join("\r\n");
}

function triggerCsvDownload(fileName: string, headers: string[], rows: Array<Record<string, unknown>>): void {
  const csvContent = buildCsvContent(headers, rows);
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("setup");
  const [session, setSession] = useState<SessionState>(DEFAULT_SESSION);
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const [csvData, setCsvData] = useState<CsvLoadResponse | null>(null);
  const [previewRows, setPreviewRows] = useState(8);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [selectedFeatureIndex, setSelectedFeatureIndex] = useState(0);
  const [selectedExperimentIndex, setSelectedExperimentIndex] = useState(0);
  const [localModelPaths, setLocalModelPaths] = useState<string[]>([]);
  const [selectedLocalModelPath, setSelectedLocalModelPath] = useState("");
  const [llamaPort, setLlamaPort] = useState(8080);
  const [llamaCtxSize, setLlamaCtxSize] = useState(8192);
  const [llamaBinaryPath, setLlamaBinaryPath] = useState("");
  const [llamaStatus, setLlamaStatus] = useState<LlamaServerStatusResponse | null>(null);
  const [hfSearchQuery, setHfSearchQuery] = useState("phi");
  const [hfSearchLimit, setHfSearchLimit] = useState(12);
  const [hfSearchResults, setHfSearchResults] = useState<HuggingFaceGgufModel[]>([]);
  const [selectedHfRepoId, setSelectedHfRepoId] = useState("");
  const [hfRepoFiles, setHfRepoFiles] = useState<HuggingFaceGgufFile[]>([]);
  const [selectedHfFile, setSelectedHfFile] = useState("");
  const [hfDestinationDir, setHfDestinationDir] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [reportText, setReportText] = useState("");
  const [singleResult, setSingleResult] = useState<FeatureTestResult | null>(null);
  const [jobId, setJobId] = useState<string>("");
  const [lastJobStatus, setLastJobStatus] = useState<string>("");
  const [reportsToTest, setReportsToTest] = useState(5);
  const [requestedBatchSize, setRequestedBatchSize] = useState(1);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [runHistory, setRunHistory] = useState<RunHistoryItem[]>([]);
  const [pendingRunExperimentNames, setPendingRunExperimentNames] = useState<string[]>([]);
  const [schemaFileHandle, setSchemaFileHandle] = useState<SchemaFileHandle | null>(null);
  const [isLocalSchemaIOPending, setIsLocalSchemaIOPending] = useState(false);
  const schemaFileInputRef = useRef<HTMLInputElement>(null);
  const recordedJobIdsRef = useRef<Set<string>>(new Set());
  const llamaStatusRef = useRef<LlamaServerStatusResponse | null>(null);

  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: healthCheck,
    refetchInterval: 15_000,
  });

  const sessionQuery = useQuery({
    queryKey: ["session"],
    queryFn: loadSession,
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const data = query.state.data as JobResponse | undefined;
      if (!data) {
        return 1000;
      }
      return data.status === "pending" || data.status === "running" ? 1000 : false;
    },
  });

  const loadCsvByPathMutation = useMutation({
    mutationFn: loadCsvFromPath,
    onSuccess: (data) => {
      setCsvData(data);
      setSession((prev) => ({
        ...prev,
        csv_cache: data,
        id_column: data.inferred_id_column || prev.id_column,
        report_column: data.inferred_report_column || prev.report_column,
      }));
      addToast("CSV loaded from path.", "success");
    },
    onError: (error: Error) => {
      setPendingRunExperimentNames([]);
      addToast(error.message, "error");
    },
  });

  const loadCsvByFileMutation = useMutation({
    mutationFn: async (payload: {
      file: File;
      previewRows: number;
      idColumn: string;
      reportColumn: string;
    }) => {
      try {
        return await loadCsvFromFile(payload);
      } catch {
        return loadCsvFromFileLocal(payload);
      }
    },
    onSuccess: (data) => {
      setCsvData(data);
      setSession((prev) => ({
        ...prev,
        csv_path: data.source,
        csv_cache: data,
        id_column: data.inferred_id_column || prev.id_column,
        report_column: data.inferred_report_column || prev.report_column,
      }));
      if (data.encoding === "client-local") {
        addToast("CSV preview loaded locally.", "info");
      } else {
        addToast("CSV loaded from file.", "success");
      }
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  function applyLlamaStatus(data: LlamaServerStatusResponse): void {
    setLlamaStatus(data);
    setLlamaBinaryPath(data.binary_path || "");

    if (data.managed_port) {
      setLlamaPort(data.managed_port);
    }
    if (data.managed_ctx_size) {
      setLlamaCtxSize(data.managed_ctx_size);
    }

    if (data.managed_model_path) {
      setSelectedLocalModelPath(data.managed_model_path);
      setLocalModelPaths((prev) => {
        if (prev.includes(data.managed_model_path)) {
          return prev;
        }
        return [data.managed_model_path, ...prev];
      });
    }
  }

  const listLocalModelsMutation = useMutation({
    mutationFn: listLocalModels,
    onSuccess: (data) => {
      setLocalModelPaths(data.models);
      setLlamaBinaryPath(data.binary_path || "");
      if (data.models.length > 0 && !selectedLocalModelPath) {
        setSelectedLocalModelPath(data.models[0]);
      }

      if (data.models.length === 0) {
        addToast("No local GGUF models found. Set a manual path or download one.", "info");
      } else if (data.timed_out) {
        addToast(`Found ${data.models.length} local model(s). Scan timed out early.`, "info");
      } else {
        addToast(`Found ${data.models.length} local model(s).`, "success");
      }
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
      setLocalModelPaths([]);
    },
  });

  const listHfFilesMutation = useMutation({
    mutationFn: listGgufFiles,
    onSuccess: (data) => {
      setHfRepoFiles(data.files);
      if (data.files.length === 0) {
        setSelectedHfFile("");
        addToast("No GGUF files found in the selected repository.", "info");
        return;
      }

      setSelectedHfFile((prev) =>
        prev && data.files.some((file) => file.file_name === prev) ? prev : data.files[0].file_name,
      );
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
      setHfRepoFiles([]);
      setSelectedHfFile("");
    },
  });

  const searchHfModelsMutation = useMutation({
    mutationFn: searchGgufModels,
    onSuccess: (data) => {
      setHfSearchResults(data.models);
      if (data.models.length === 0) {
        setSelectedHfRepoId("");
        setHfRepoFiles([]);
        setSelectedHfFile("");
        addToast("No GGUF repositories matched this search.", "info");
        return;
      }

      const firstRepo = data.models[0];
      setSelectedHfRepoId(firstRepo.repo_id);
      setHfRepoFiles(firstRepo.gguf_files);
      setSelectedHfFile(firstRepo.gguf_files[0]?.file_name ?? "");
      addToast(`Found ${data.models.length} GGUF repositories.`, "success");
      listHfFilesMutation.mutate({ repoId: firstRepo.repo_id });
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  const downloadHfModelMutation = useMutation({
    mutationFn: downloadGgufModel,
    onSuccess: (data) => {
      setSelectedLocalModelPath(data.downloaded_path);
      setLocalModelPaths((prev) => {
        if (prev.includes(data.downloaded_path)) {
          return prev;
        }
        return [data.downloaded_path, ...prev];
      });
      listLocalModelsMutation.mutate();
      addToast(`Installed ${data.file_name} to ${data.destination_dir}.`, "success");
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  const ensureLlamaMutation = useMutation({
    mutationFn: ensureLlamaServer,
    onSuccess: (data) => {
      applyLlamaStatus(data);
      if (data.managed_port) {
        setSession((prev) => ({
          ...prev,
          llama_url: `http://127.0.0.1:${data.managed_port}`,
          server_model: "",
        }));
      }
      if (data.changed) {
        addToast("Local model server updated.", "info");
      }
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
      setLlamaStatus(null);
    },
  });

  const stopLlamaMutation = useMutation({
    mutationFn: stopLlamaServer,
    onSuccess: (data) => {
      applyLlamaStatus(data);
      addToast("Local model server stopped.", "info");
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  const testFeatureMutation = useMutation({
    mutationFn: testFeature,
    onSuccess: (data) => {
      setSingleResult(data);
      if (data.status === "ok") {
        addToast(`Feature tested: ${data.feature_name}`, "success");
      } else {
        addToast(`${data.feature_name}: ${data.error}`, "error");
      }
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  const testBatchMutation = useMutation({
    mutationFn: testBatch,
    onSuccess: (data, variables) => {
      const experimentCount = Array.isArray(variables.experiments) && variables.experiments.length > 0
        ? variables.experiments.length
        : 1;
      setJobId(data.job_id);
      setLastJobStatus("");
      setRequestedBatchSize(variables.reports.length * experimentCount);
      addToast(
        `Started feature run for ${variables.reports.length} report(s) x ${experimentCount} experiment(s).`,
        "info",
      );
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  const cancelJobMutation = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => {
      addToast("Cancellation requested.", "info");
      void jobQuery.refetch();
    },
    onError: (error: Error) => {
      addToast(error.message, "error");
    },
  });

  const selectedExperiment = session.experiment_profiles[selectedExperimentIndex] ?? null;
  const activeExperiment = useMemo(() => {
    const profileById = session.experiment_profiles.find(
      (profile) => profile.id === session.active_experiment_id,
    );
    if (profileById) {
      return profileById;
    }
    return session.experiment_profiles[0] ?? null;
  }, [session.experiment_profiles, session.active_experiment_id]);
  const activeRunExperimentIds = useMemo(
    () =>
      normalizeActiveExperimentIds(
        session.active_experiment_ids,
        session.experiment_profiles,
        session.active_experiment_id,
      ),
    [session.active_experiment_ids, session.experiment_profiles, session.active_experiment_id],
  );
  const activeRunExperiments = useMemo(
    () => session.experiment_profiles.filter((profile) => activeRunExperimentIds.includes(profile.id)),
    [session.experiment_profiles, activeRunExperimentIds],
  );

  const selectedFeature = session.features[selectedFeatureIndex] ?? null;
  const selectedTypeHintValue = (selectedFeature?.type_hint ?? "string").trim() || "string";
  const typeHintOptions = useMemo(() => {
    if (TYPE_HINT_OPTIONS.some((option) => option.value === selectedTypeHintValue)) {
      return TYPE_HINT_OPTIONS;
    }
    return [
      ...TYPE_HINT_OPTIONS,
      { value: selectedTypeHintValue, label: `Custom (${selectedTypeHintValue})` },
    ];
  }, [selectedTypeHintValue]);

  const selectedHfRepo = useMemo(
    () => hfSearchResults.find((model) => model.repo_id === selectedHfRepoId) ?? null,
    [hfSearchResults, selectedHfRepoId],
  );
  const selectedHfFileInfo = useMemo(
    () => hfRepoFiles.find((file) => file.file_name === selectedHfFile) ?? null,
    [hfRepoFiles, selectedHfFile],
  );
  const availableJudgeModels = useMemo(() => {
    const deduped = new Set<string>();
    for (const model of llamaStatus?.server_models ?? []) {
      const cleaned = String(model ?? "").trim();
      if (cleaned) {
        deduped.add(cleaned);
      }
    }
    for (const modelPath of localModelPaths) {
      const cleaned = String(modelPath ?? "").trim();
      if (cleaned) {
        deduped.add(cleaned);
      }
    }
    const selectedModel = String(selectedLocalModelPath ?? "").trim();
    if (selectedModel) {
      deduped.add(selectedModel);
    }
    return [...deduped];
  }, [llamaStatus, localModelPaths, selectedLocalModelPath]);
  const selectedJudgeModelValue = selectedExperiment
    ? String(selectedExperiment.judge.model ?? "").trim()
    : "";
  const judgeModelOptions = useMemo(() => {
    if (!selectedJudgeModelValue || selectedJudgeModelValue === JUDGE_MODEL_USE_EXTRACTION) {
      return availableJudgeModels;
    }
    if (availableJudgeModels.includes(selectedJudgeModelValue)) {
      return availableJudgeModels;
    }
    return [selectedJudgeModelValue, ...availableJudgeModels];
  }, [availableJudgeModels, selectedJudgeModelValue]);

  const sampleRows = csvData?.preview ?? [];
  const availableColumns = csvData?.columns ?? [];
  const idColumnSelectValue = availableColumns.includes(session.id_column) ? session.id_column : "";
  const reportColumnSelectValue = availableColumns.includes(session.report_column) ? session.report_column : "";
  const safeSampleIndex = Math.max(1, Math.min(session.sample_index || 1, Math.max(sampleRows.length, 1)));

  const selectedSampleRow = useMemo(() => {
    const index = safeSampleIndex - 1;
    if (index < 0 || index >= sampleRows.length) {
      return null;
    }
    return sampleRows[index];
  }, [safeSampleIndex, sampleRows]);

  const safeReportsToTest = Math.max(1, Math.min(20, Math.floor(reportsToTest || 1)));
  const selectedBatchReports = useMemo<BatchReportItem[]>(() => {
    if (sampleRows.length > 0) {
      const startIndex = Math.max(0, safeSampleIndex - 1);
      const fromPreview = sampleRows
        .slice(startIndex, startIndex + safeReportsToTest)
        .map((row) => ({
          rowNumber: row.row_number,
          reportText: resolveReportTextFromValues(row.values, session.report_column).trim(),
        }))
        .filter((row) => row.reportText.length > 0);

      if (fromPreview.length > 0) {
        return fromPreview;
      }
    }

    if (!reportText.trim()) {
      return [];
    }

    return [{ rowNumber: null, reportText: reportText.trim() }];
  }, [sampleRows, safeSampleIndex, safeReportsToTest, session.report_column, reportText]);

  useEffect(() => {
    if (!sessionQuery.data || sessionLoaded) {
      return;
    }
    const loaded = sessionQuery.data as Partial<SessionState>;
    const restoredFeatures = normalizeFeatureList(loaded.features);
    const restoredCsvCache = normalizeCsvCache(loaded.csv_cache);
    const restoredExperiments = normalizeExperimentList(loaded.experiment_profiles);
    const restoredActiveExperimentId = restoredExperiments.some(
      (profile) => profile.id === loaded.active_experiment_id,
    )
      ? String(loaded.active_experiment_id)
      : restoredExperiments[0].id;
    const restoredActiveExperimentIds = normalizeActiveExperimentIds(
      loaded.active_experiment_ids,
      restoredExperiments,
      restoredActiveExperimentId,
    );
    setCsvData(restoredCsvCache);
    setSession({
      ...DEFAULT_SESSION,
      csv_path: String(loaded.csv_path ?? ""),
      csv_cache: restoredCsvCache,
      schema_path: "",
      llama_url: String(loaded.llama_url ?? DEFAULT_SESSION.llama_url),
      temperature: Number(loaded.temperature ?? DEFAULT_SESSION.temperature),
      max_retries: Number(loaded.max_retries ?? DEFAULT_SESSION.max_retries),
      id_column: String(loaded.id_column ?? DEFAULT_SESSION.id_column),
      report_column: String(loaded.report_column ?? DEFAULT_SESSION.report_column),
      sample_index: Number(loaded.sample_index ?? DEFAULT_SESSION.sample_index),
      server_model: String(loaded.server_model ?? ""),
      features: restoredFeatures,
      experiment_profiles: restoredExperiments,
      active_experiment_id: restoredActiveExperimentId,
      active_experiment_ids: restoredActiveExperimentIds,
    });
    setLlamaPort(parsePortFromLlamaUrl(String(loaded.llama_url ?? DEFAULT_SESSION.llama_url)));
    setSessionLoaded(true);
  }, [sessionQuery.data, sessionLoaded]);

  useEffect(() => {
    if (!sessionLoaded || localModelPaths.length > 0 || listLocalModelsMutation.isPending) {
      return;
    }
    listLocalModelsMutation.mutate();
  }, [sessionLoaded]);

  useEffect(() => {
    if (!sessionLoaded) {
      return;
    }
    const nextLlamaUrl = `http://127.0.0.1:${llamaPort}`;
    setSession((prev) => {
      if (prev.llama_url === nextLlamaUrl) {
        return prev;
      }
      return { ...prev, llama_url: nextLlamaUrl };
    });
  }, [llamaPort, sessionLoaded]);

  useEffect(() => {
    if (!selectedSampleRow) {
      return;
    }

    setReportText(resolveReportTextFromValues(selectedSampleRow.values, session.report_column));
  }, [selectedSampleRow, session.report_column]);

  useEffect(() => {
    const status = jobQuery.data?.status;
    if (!status || status === lastJobStatus) {
      return;
    }
    setLastJobStatus(status);
    if (status === "completed") {
      addToast("Feature run completed.", "success");
    }
    if (status === "failed") {
      addToast(jobQuery.data?.error || "Feature run failed.", "error");
    }
    if (status === "cancelled") {
      addToast("Feature run cancelled.", "info");
    }
  }, [jobQuery.data?.status]);

  useEffect(() => {
    const currentJob = jobQuery.data;
    if (!jobId || !currentJob) {
      return;
    }

    if (!["completed", "failed", "cancelled"].includes(currentJob.status)) {
      return;
    }
    if (recordedJobIdsRef.current.has(jobId)) {
      return;
    }
    recordedJobIdsRef.current.add(jobId);

    const allResults = currentJob.results ?? [];
    const okCount = allResults.filter((result) => result.status === "ok").length;
    const llmErrorCount = allResults.filter((result) => result.status === "llm_error").length;
    const parseErrorCount = allResults.filter((result) => result.status === "parse_error").length;
    const judgeSummary = countJudgeOutcome(allResults);
    const resultExperimentNames = [...new Set(
      allResults
        .map((result) => String(result.experiment_name ?? "").trim())
        .filter((name) => name.length > 0),
    )];
    const experimentNames =
      resultExperimentNames.length > 0
        ? resultExperimentNames.join(", ")
        : pendingRunExperimentNames.length > 0
          ? pendingRunExperimentNames.join(", ")
          : activeExperiment?.name || "Unknown";

    const nextRunRecord: RunHistoryItem = {
      jobId,
      experimentNames,
      status: currentJob.status,
      startedAtUnix: currentJob.started_at ?? Math.floor(Date.now() / 1000),
      finishedAtUnix: currentJob.finished_at ?? null,
      reportsReviewed: currentJob.reports_completed ?? 0,
      checksTotal: currentJob.total ?? 0,
      checksCompleted: currentJob.completed ?? 0,
      okCount,
      llmErrorCount,
      parseErrorCount,
      judgeAccepted: judgeSummary.accepted,
      judgeRejected: judgeSummary.rejected,
      judgeUncertain: judgeSummary.uncertain,
      judgeErrors: judgeSummary.judgeError,
    };

    setRunHistory((prev) => [nextRunRecord, ...prev].slice(0, 30));
    setPendingRunExperimentNames([]);
  }, [jobId, jobQuery.data, activeExperiment?.name, pendingRunExperimentNames]);

  useEffect(() => {
    llamaStatusRef.current = llamaStatus;
  }, [llamaStatus]);

  useEffect(() => {
    let stopSent = false;
    const handlePageClose = () => {
      if (stopSent) {
        return;
      }
      if (!llamaStatusRef.current?.process_running) {
        return;
      }
      stopSent = true;
      stopLlamaServerOnPageUnload();
    };

    window.addEventListener("pagehide", handlePageClose);
    window.addEventListener("beforeunload", handlePageClose);
    return () => {
      window.removeEventListener("pagehide", handlePageClose);
      window.removeEventListener("beforeunload", handlePageClose);
    };
  }, []);

  useEffect(() => {
    if (!sessionLoaded) {
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        await saveSession(session);
      } catch {
        // Ignore autosave failures. The user still has explicit feature-set save.
      }
    }, 600);

    return () => {
      window.clearTimeout(timer);
    };
  }, [session, sessionLoaded]);

  useEffect(() => {
    setSelectedFeatureIndex((prev) => {
      const maxIndex = Math.max(0, session.features.length - 1);
      return Math.max(0, Math.min(prev, maxIndex));
    });
  }, [session.features.length]);

  useEffect(() => {
    setSelectedExperimentIndex((prev) => {
      const maxIndex = Math.max(0, session.experiment_profiles.length - 1);
      return Math.max(0, Math.min(prev, maxIndex));
    });
  }, [session.experiment_profiles.length]);

  useEffect(() => {
    if (session.experiment_profiles.length === 0) {
      return;
    }
    const hasActiveExperiment = session.experiment_profiles.some(
      (profile) => profile.id === session.active_experiment_id,
    );
    if (hasActiveExperiment) {
      return;
    }
    setSession((prev) => ({
      ...prev,
      active_experiment_id: prev.experiment_profiles[0]?.id ?? DEFAULT_EXPERIMENT_ID,
    }));
  }, [session.experiment_profiles, session.active_experiment_id]);

  useEffect(() => {
    if (session.experiment_profiles.length === 0) {
      return;
    }

    const normalizedActiveIds = normalizeActiveExperimentIds(
      session.active_experiment_ids,
      session.experiment_profiles,
      session.active_experiment_id,
    );
    const currentActiveIds = Array.isArray(session.active_experiment_ids)
      ? session.active_experiment_ids
      : [];
    const unchanged =
      normalizedActiveIds.length === currentActiveIds.length &&
      normalizedActiveIds.every((experimentId, index) => experimentId === currentActiveIds[index]);

    if (unchanged) {
      return;
    }

    setSession((prev) => ({
      ...prev,
      active_experiment_ids: normalizedActiveIds,
    }));
  }, [
    session.active_experiment_ids,
    session.active_experiment_id,
    session.experiment_profiles,
  ]);

  useEffect(() => {
    const nextIndex = session.experiment_profiles.findIndex(
      (profile) => profile.id === session.active_experiment_id,
    );
    if (nextIndex >= 0) {
      setSelectedExperimentIndex(nextIndex);
    }
  }, [session.experiment_profiles, session.active_experiment_id]);

  function addToast(message: string, kind: ToastKind): void {
    const id = Date.now() + Math.floor(Math.random() * 1000);
    setToasts((prev) => [...prev, { id, message, kind }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 3500);
  }

  function updateFeature(index: number, patch: Partial<Feature>): void {
    setSession((prev) => {
      const nextFeatures = [...prev.features];
      const current = nextFeatures[index];
      if (!current) {
        return prev;
      }
      nextFeatures[index] = { ...current, ...patch };
      return { ...prev, features: nextFeatures };
    });
  }

  function handleAddFeature(): void {
    setSession((prev) => {
      const nextIndex = prev.features.length + 1;
      const nextFeature: Feature = cloneFeature({
        ...DEFAULT_FEATURE,
        name: `feature_${nextIndex}`,
      });
      return { ...prev, features: [...prev.features, nextFeature] };
    });
    setSelectedFeatureIndex(session.features.length);
  }

  function handleDeleteFeature(): void {
    if (!selectedFeature) {
      return;
    }
    setSession((prev) => {
      const nextFeatures = prev.features.filter((_, index) => index !== selectedFeatureIndex);
      return { ...prev, features: normalizeFeatureList(nextFeatures) };
    });
    setSelectedFeatureIndex((prev) => Math.max(0, prev - 1));
  }

  function updateExperiment(index: number, patch: Partial<ExperimentProfile>): void {
    setSession((prev) => {
      const nextExperiments = [...prev.experiment_profiles];
      const current = nextExperiments[index];
      if (!current) {
        return prev;
      }
      nextExperiments[index] = cloneExperiment({
        ...current,
        ...patch,
        judge: current.judge,
      });
      return { ...prev, experiment_profiles: nextExperiments };
    });
  }

  function updateExperimentJudge(index: number, patch: Partial<ExperimentProfile["judge"]>): void {
    setSession((prev) => {
      const nextExperiments = [...prev.experiment_profiles];
      const current = nextExperiments[index];
      if (!current) {
        return prev;
      }
      nextExperiments[index] = cloneExperiment({
        ...current,
        judge: {
          ...current.judge,
          ...patch,
          acceptance_threshold: clampJudgeThreshold(
            Number(patch.acceptance_threshold ?? current.judge.acceptance_threshold),
          ),
        },
      });
      return { ...prev, experiment_profiles: nextExperiments };
    });
  }

  function handleAddExperiment(): void {
    setSession((prev) => {
      const nextPosition = prev.experiment_profiles.length + 1;
      const name = `Experiment ${nextPosition}`;
      const nextExperiment = cloneExperiment({
        ...DEFAULT_EXPERIMENT,
        id: createExperimentId(name, nextPosition),
        name,
      });
      return {
        ...prev,
        experiment_profiles: [...prev.experiment_profiles, nextExperiment],
        active_experiment_id: nextExperiment.id,
        active_experiment_ids: [...new Set([...prev.active_experiment_ids, nextExperiment.id])],
      };
    });
    setSelectedExperimentIndex(session.experiment_profiles.length);
  }

  function handleDeleteExperiment(): void {
    if (!selectedExperiment) {
      return;
    }
    setSession((prev) => {
      const remaining = prev.experiment_profiles.filter((_, index) => index !== selectedExperimentIndex);
      const normalized = normalizeExperimentList(remaining);
      const hasActive = normalized.some((profile) => profile.id === prev.active_experiment_id);
      const nextPrimaryExperimentId = hasActive ? prev.active_experiment_id : normalized[0].id;
      const nextActiveExperimentIds = normalizeActiveExperimentIds(
        prev.active_experiment_ids,
        normalized,
        nextPrimaryExperimentId,
      );
      return {
        ...prev,
        experiment_profiles: normalized,
        active_experiment_id: nextPrimaryExperimentId,
        active_experiment_ids: nextActiveExperimentIds,
      };
    });
    setSelectedExperimentIndex((prev) => Math.max(0, prev - 1));
  }

  function setActiveExperiment(experimentId: string): void {
    setSession((prev) => {
      if (!prev.experiment_profiles.some((profile) => profile.id === experimentId)) {
        return prev;
      }
      return {
        ...prev,
        active_experiment_id: experimentId,
        active_experiment_ids: prev.active_experiment_ids.includes(experimentId)
          ? prev.active_experiment_ids
          : [...prev.active_experiment_ids, experimentId],
      };
    });
  }

  function setRunExperimentEnabled(experimentId: string, enabled: boolean): void {
    setSession((prev) => {
      if (!prev.experiment_profiles.some((profile) => profile.id === experimentId)) {
        return prev;
      }

      const nextActiveIds = enabled
        ? [...new Set([...prev.active_experiment_ids, experimentId])]
        : prev.active_experiment_ids.filter((id) => id !== experimentId);

      if (nextActiveIds.length === prev.active_experiment_ids.length) {
        const unchanged = nextActiveIds.every((id, index) => id === prev.active_experiment_ids[index]);
        if (unchanged) {
          return prev;
        }
      }

      return {
        ...prev,
        active_experiment_ids: nextActiveIds,
      };
    });
  }

  function handleLoadCsvByPath(pathOverride?: string): void {
    const csvPath = (pathOverride ?? session.csv_path).trim();
    if (!csvPath) {
      addToast("Set a CSV path first.", "error");
      return;
    }
    loadCsvByPathMutation.mutate({
      path: csvPath,
      previewRows,
      idColumn: session.id_column,
      reportColumn: session.report_column,
    });
  }

  function handleCsvPathInputKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
    if (event.key !== "Enter") {
      return;
    }

    event.preventDefault();
    handleLoadCsvByPath(event.currentTarget.value);
  }

  function handleCsvFileChange(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      setSelectedFileName("");
      return;
    }

    setSelectedFileName(file.name);
    loadCsvByFileMutation.mutate({
      file,
      previewRows,
      idColumn: session.id_column,
      reportColumn: session.report_column,
    });
  }

  async function applySchemaFromFile(file: File, handle: SchemaFileHandle | null): Promise<void> {
    const content = await file.text();
    const features = parseSchemaFeaturesFromJson(content);
    setSchemaFileHandle(handle);
    setSession((prev) => ({
      ...prev,
      schema_path: "",
      features,
    }));
    setSelectedFeatureIndex(0);
    addToast(`Feature set loaded from ${file.name}.`, "success");
  }

  async function handleLoadSchema(): Promise<void> {
    if (isLocalSchemaIOPending) {
      return;
    }

    const pickerWindow = window as FilePickerWindow;
    if (!pickerWindow.showOpenFilePicker) {
      if (schemaFileInputRef.current) {
        schemaFileInputRef.current.value = "";
        schemaFileInputRef.current.click();
      }
      return;
    }

    setIsLocalSchemaIOPending(true);
    try {
      const [handle] = await pickerWindow.showOpenFilePicker({
        multiple: false,
        excludeAcceptAllOption: false,
        types: SCHEMA_PICKER_TYPES,
      });
      if (!handle) {
        return;
      }
      const file = await handle.getFile();
      await applySchemaFromFile(file, handle);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      addToast(`Failed to load feature set: ${getErrorMessage(error)}`, "error");
    } finally {
      setIsLocalSchemaIOPending(false);
    }
  }

  function handleSchemaFileInputChange(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0] ?? null;
    event.target.value = "";
    if (!file) {
      return;
    }

    setIsLocalSchemaIOPending(true);
    void applySchemaFromFile(file, null)
      .catch((error) => {
        addToast(`Failed to load feature set: ${getErrorMessage(error)}`, "error");
      })
      .finally(() => {
        setIsLocalSchemaIOPending(false);
      });
  }

  async function saveSchemaToFileHandle(handle: SchemaFileHandle): Promise<void> {
    const writable = await handle.createWritable();
    await writable.write(buildSchemaDocument(session.features));
    await writable.close();
    setSchemaFileHandle(handle);
    setSession((prev) => ({
      ...prev,
      schema_path: "",
    }));
    addToast(`Feature set saved to ${handle.name}.`, "success");
  }

  function triggerSchemaDownloadFallback(): void {
    const blob = new Blob([buildSchemaDocument(session.features)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "data_extraction_calibrated.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    addToast("Feature set download started.", "info");
  }

  async function handleSaveSchema(): Promise<void> {
    if (isLocalSchemaIOPending) {
      return;
    }

    if (schemaFileHandle) {
      setIsLocalSchemaIOPending(true);
      try {
        await saveSchemaToFileHandle(schemaFileHandle);
      } catch (error) {
        addToast(`Failed to save feature set: ${getErrorMessage(error)}`, "error");
      } finally {
        setIsLocalSchemaIOPending(false);
      }
      return;
    }

    const pickerWindow = window as FilePickerWindow;
    if (!pickerWindow.showSaveFilePicker) {
      triggerSchemaDownloadFallback();
      return;
    }

    setIsLocalSchemaIOPending(true);
    try {
      const handle = await pickerWindow.showSaveFilePicker({
        suggestedName: "data_extraction_calibrated.json",
        excludeAcceptAllOption: false,
        types: SCHEMA_PICKER_TYPES,
      });
      await saveSchemaToFileHandle(handle);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      addToast(`Failed to save feature set: ${getErrorMessage(error)}`, "error");
    } finally {
      setIsLocalSchemaIOPending(false);
    }
  }

  async function ensureLlamaReady(): Promise<boolean> {
    if (!selectedLocalModelPath.trim()) {
      addToast("Select a local GGUF model first.", "error");
      return false;
    }

    try {
      await ensureLlamaMutation.mutateAsync({
        modelPath: selectedLocalModelPath,
        port: llamaPort,
        ctxSize: llamaCtxSize,
      });
      return true;
    } catch {
      return false;
    }
  }

  async function handleRunSingleTest(): Promise<void> {
    if (!selectedFeature) {
      addToast("Select a feature first.", "error");
      return;
    }
    if (!activeExperiment) {
      addToast("Create an experiment profile first.", "error");
      return;
    }
    if (!reportText.trim()) {
      addToast("Report text is empty.", "error");
      return;
    }

    const ready = await ensureLlamaReady();
    if (!ready) {
      return;
    }

    try {
      const resolvedJudge = resolveJudgeConfigForRun(activeExperiment.judge);
      if (activeExperiment.judge.enabled && !resolvedJudge.enabled) {
        addToast("Judge pass enabled but no judge model selected. Judge is skipped.", "info");
      }
      setPendingRunExperimentNames([activeExperiment.name]);
      await testFeatureMutation.mutateAsync({
        feature: selectedFeature,
        reportText,
        llamaUrl: `http://127.0.0.1:${llamaPort}`,
        temperature: session.temperature,
        maxRetries: session.max_retries,
        model: "",
        experimentId: activeExperiment.id,
        experimentName: activeExperiment.name,
        systemInstructions: activeExperiment.system_instructions,
        extractionInstructions: activeExperiment.extraction_instructions,
        reasoningMode: activeExperiment.reasoning_mode,
        reasoningInstructions: activeExperiment.reasoning_instructions,
        outputInstructions: activeExperiment.output_instructions,
        judge: resolvedJudge,
      });
    } catch {
      // Error toast is handled by mutation onError.
    }
  }

  function handleRefreshLocalModels(): void {
    listLocalModelsMutation.mutate();
  }

  function handleStopLlamaServer(): void {
    const currentLlamaUrl = llamaStatus?.llama_url || `http://127.0.0.1:${llamaPort}`;
    stopLlamaMutation.mutate({ llamaUrl: currentLlamaUrl });
  }

  function handleSearchHfModels(): void {
    const query = hfSearchQuery.trim();
    if (!query) {
      addToast("Enter a Hugging Face search query first.", "error");
      return;
    }

    searchHfModelsMutation.mutate({
      query,
      limit: Math.max(1, Math.min(50, Math.floor(hfSearchLimit || 12))),
    });
  }

  function handleSelectHfRepo(repoId: string): void {
    const repoValue = repoId.trim();
    setSelectedHfRepoId(repoValue);

    if (!repoValue) {
      setHfRepoFiles([]);
      setSelectedHfFile("");
      return;
    }

    const repository = hfSearchResults.find((model) => model.repo_id === repoValue);
    const fallbackFiles = repository?.gguf_files ?? [];
    setHfRepoFiles(fallbackFiles);
    setSelectedHfFile(fallbackFiles[0]?.file_name ?? "");
    listHfFilesMutation.mutate({ repoId: repoValue });
  }

  async function handleInstallHfModel(): Promise<void> {
    if (!selectedHfRepoId) {
      addToast("Select a Hugging Face repository first.", "error");
      return;
    }
    if (!selectedHfFile) {
      addToast("Select a GGUF file first.", "error");
      return;
    }

    try {
      await downloadHfModelMutation.mutateAsync({
        repoId: selectedHfRepoId,
        fileName: selectedHfFile,
        destinationDir: hfDestinationDir.trim(),
        hfToken: hfToken.trim(),
      });
    } catch {
      // Error toast is handled by mutation onError.
    }
  }

  async function handleRunAllTests(): Promise<void> {
    if (session.features.length === 0) {
      addToast("Define at least one feature.", "error");
      return;
    }
    if (activeRunExperiments.length === 0) {
      addToast("Select at least one experiment profile for runs.", "error");
      return;
    }
    if (selectedBatchReports.length === 0) {
      addToast("No valid reports found in the selected range.", "error");
      return;
    }

    const ready = await ensureLlamaReady();
    if (!ready) {
      return;
    }

    try {
      const primaryExperiment = activeRunExperiments[0];
      if (!primaryExperiment) {
        addToast("Select at least one experiment profile for runs.", "error");
        return;
      }
      const primaryJudge = resolveJudgeConfigForRun(primaryExperiment.judge);
      const judgeSkippedProfiles = activeRunExperiments
        .filter((experiment) => experiment.judge.enabled && !resolveJudgeConfigForRun(experiment.judge).enabled)
        .map((experiment) => experiment.name);
      if (judgeSkippedProfiles.length > 0) {
        addToast(`Judge skipped (no model selected): ${judgeSkippedProfiles.join(", ")}.`, "info");
      }
      setPendingRunExperimentNames(activeRunExperiments.map((experiment) => experiment.name));
      await testBatchMutation.mutateAsync({
        features: session.features,
        reports: selectedBatchReports,
        llamaUrl: `http://127.0.0.1:${llamaPort}`,
        temperature: session.temperature,
        maxRetries: session.max_retries,
        model: "",
        experimentId: primaryExperiment.id,
        experimentName: primaryExperiment.name,
        systemInstructions: primaryExperiment.system_instructions,
        extractionInstructions: primaryExperiment.extraction_instructions,
        reasoningMode: primaryExperiment.reasoning_mode,
        reasoningInstructions: primaryExperiment.reasoning_instructions,
        outputInstructions: primaryExperiment.output_instructions,
        judge: primaryJudge,
        experiments: activeRunExperiments.map((experiment) => ({
          judge: resolveJudgeConfigForRun(experiment.judge),
          id: experiment.id,
          name: experiment.name,
          systemInstructions: experiment.system_instructions,
          extractionInstructions: experiment.extraction_instructions,
          reasoningMode: experiment.reasoning_mode,
          reasoningInstructions: experiment.reasoning_instructions,
          outputInstructions: experiment.output_instructions,
        })),
      });
    } catch {
      // Error toast is handled by mutation onError.
    }
  }

  function handleExportRunCsv(): void {
    const currentJob = jobQuery.data;
    if (!currentJob) {
      addToast("Run at least one job before exporting CSV.", "error");
      return;
    }

    const rows: Array<Record<string, unknown>> = [];
    const reportItems = currentJob.report_results ?? [];

    if (reportItems.length > 0) {
      for (const reportItem of reportItems) {
        for (const result of reportItem.results ?? []) {
          rows.push({
            job_id: currentJob.job_id,
            run_index: result.run_index ?? reportItem.run_index ?? "",
            report_index: result.report_index ?? reportItem.report_index ?? "",
            row_number: result.row_number ?? reportItem.row_number ?? "",
            experiment_id:
              String(result.experiment_id ?? "").trim() || String(reportItem.experiment_id ?? "").trim(),
            experiment_name:
              String(result.experiment_name ?? "").trim() || String(reportItem.experiment_name ?? "").trim(),
            feature_name: result.feature_name,
            status: result.status,
            value: result.value,
            duration_ms: result.duration_ms,
            model: result.model,
            error: result.error,
            judge_status: result.judge_result?.status ?? "",
            judge_verdict: result.judge_result?.verdict ?? "",
            judge_score: result.judge_result?.score ?? "",
            judge_rationale: result.judge_result?.rationale ?? "",
            judge_model: result.judge_result?.model ?? "",
            judge_error: result.judge_result?.error ?? "",
            report_text: reportItem.report_text ?? "",
          });
        }
      }
    } else {
      for (const result of currentJob.results ?? []) {
        rows.push({
          job_id: currentJob.job_id,
          run_index: result.run_index ?? "",
          report_index: result.report_index ?? "",
          row_number: result.row_number ?? "",
          experiment_id: result.experiment_id ?? "",
          experiment_name: result.experiment_name ?? "",
          feature_name: result.feature_name,
          status: result.status,
          value: result.value,
          duration_ms: result.duration_ms,
          model: result.model,
          error: result.error,
          judge_status: result.judge_result?.status ?? "",
          judge_verdict: result.judge_result?.verdict ?? "",
          judge_score: result.judge_result?.score ?? "",
          judge_rationale: result.judge_result?.rationale ?? "",
          judge_model: result.judge_result?.model ?? "",
          judge_error: result.judge_result?.error ?? "",
          report_text: "",
        });
      }
    }

    if (rows.length === 0) {
      addToast("No run results available to export.", "error");
      return;
    }

    const headers = [
      "job_id",
      "run_index",
      "report_index",
      "row_number",
      "experiment_id",
      "experiment_name",
      "feature_name",
      "status",
      "value",
      "duration_ms",
      "model",
      "error",
      "judge_status",
      "judge_verdict",
      "judge_score",
      "judge_rationale",
      "judge_model",
      "judge_error",
      "report_text",
    ];
    const suffix = currentJob.job_id || `run_${Date.now()}`;
    triggerCsvDownload(`calibrator_run_${suffix}.csv`, headers, rows);
    addToast("Run results CSV download started.", "info");
  }

  function handleExportRunHistoryCsv(): void {
    if (runHistory.length === 0) {
      addToast("No run comparison rows available to export.", "error");
      return;
    }

    const headers = [
      "job_id",
      "experiments",
      "status",
      "started_at",
      "finished_at",
      "reports_reviewed",
      "checks_completed",
      "checks_total",
      "ok_count",
      "llm_error_count",
      "parse_error_count",
      "judge_accepted",
      "judge_rejected",
      "judge_uncertain",
      "judge_errors",
    ];
    const rows = runHistory.map((item) => ({
      job_id: item.jobId,
      experiments: item.experimentNames,
      status: item.status,
      started_at: new Date(item.startedAtUnix * 1000).toISOString(),
      finished_at: item.finishedAtUnix ? new Date(item.finishedAtUnix * 1000).toISOString() : "",
      reports_reviewed: item.reportsReviewed,
      checks_completed: item.checksCompleted,
      checks_total: item.checksTotal,
      ok_count: item.okCount,
      llm_error_count: item.llmErrorCount,
      parse_error_count: item.parseErrorCount,
      judge_accepted: item.judgeAccepted,
      judge_rejected: item.judgeRejected,
      judge_uncertain: item.judgeUncertain,
      judge_errors: item.judgeErrors,
    }));
    triggerCsvDownload(`calibrator_run_history_${Date.now()}.csv`, headers, rows);
    addToast("Run comparison CSV download started.", "info");
  }

  const flatResults = jobQuery.data?.results ?? [];
  const reportResults = jobQuery.data?.report_results ?? [];
  const displayReportResults = useMemo(() => {
    if (reportResults.length > 0) {
      return reportResults;
    }
    if (flatResults.length === 0) {
      return [];
    }
    return [
      {
        run_index: 1,
        report_index: 1,
        row_number: selectedSampleRow?.row_number ?? null,
        report_text: reportText,
        experiment_id: activeExperiment?.id ?? "",
        experiment_name: activeExperiment?.name ?? "",
        results: flatResults,
      },
    ];
  }, [reportResults, flatResults, selectedSampleRow, reportText, activeExperiment?.id, activeExperiment?.name]);

  const reportsTotal = jobQuery.data?.reports_total ?? (jobId ? requestedBatchSize : 0);
  const reportsCompleted = jobQuery.data?.reports_completed ?? displayReportResults.length;
  const currentJobStatus = jobQuery.data?.status ?? "idle";
  const activeReportIndex = jobQuery.data?.active_report_index ?? null;
  const runSummaryText = useMemo(() => {
    if (currentJobStatus === "idle") {
      return "No feature run started yet.";
    }
    if (currentJobStatus === "pending") {
      return "Preparing run inputs for extraction.";
    }
    if (currentJobStatus === "running") {
      const inProgress = Math.max(1, Math.min(activeReportIndex ?? reportsCompleted + 1, Math.max(reportsTotal, 1)));
      return `Working on run item ${inProgress} of ${Math.max(reportsTotal, 1)}.`;
    }
    if (currentJobStatus === "completed") {
      return `Done. Processed ${Math.max(reportsCompleted, reportsTotal)} run item(s).`;
    }
    if (currentJobStatus === "cancelled") {
      return `Stopped after ${reportsCompleted} of ${Math.max(reportsTotal, reportsCompleted, 1)} run item(s).`;
    }
    return "Run stopped because of an error.";
  }, [currentJobStatus, reportsCompleted, reportsTotal, activeReportIndex]);

  const runDetailText = useMemo(() => {
    if (jobQuery.data?.cancel_requested && (currentJobStatus === "running" || currentJobStatus === "pending")) {
      return "Cancellation requested. Waiting for the current model call to stop.";
    }
    if (currentJobStatus === "failed") {
      return jobQuery.data?.error || "Please review your model settings and try again.";
    }
    if (currentJobStatus === "completed") {
      return "You can review extracted values for each report below.";
    }
    if (currentJobStatus === "cancelled") {
      return "No new reports will be processed until you start another run.";
    }
    if (currentJobStatus === "running" || currentJobStatus === "pending") {
      return "This may take a few minutes depending on model size and report length.";
    }
    return "Choose a report count and run the full feature set.";
  }, [currentJobStatus, jobQuery.data?.cancel_requested, jobQuery.data?.error]);

  const isJobRunning = jobQuery.data?.status === "pending" || jobQuery.data?.status === "running";
  const isCancelRequested = Boolean(jobQuery.data?.cancel_requested);
  const apiOnline = healthQuery.data?.status === "ok";
  const csvSummary = csvData ? `${csvData.row_count.toLocaleString()} rows loaded` : "No CSV loaded";
  const activeFeature = selectedFeature?.name || "No feature selected";
  const activeExperimentName = activeExperiment?.name || "No experiment selected";
  const activeExperimentJudgeConfigured = activeExperiment
    ? resolveJudgeConfigForRun(activeExperiment.judge).enabled
    : false;
  const activeRunExperimentNames =
    activeRunExperiments.length > 0
      ? activeRunExperiments.map((experiment) => experiment.name).join(", ")
      : "None selected";
  const runJudgeSummary = useMemo(() => countJudgeOutcome(flatResults), [flatResults]);
  const latestJobStatus = jobQuery.data?.status ?? "idle";
  const hasRunResultsForExport = reportResults.length > 0 || flatResults.length > 0;
  const hasRunHistoryForExport = runHistory.length > 0;

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-copy">
          <h1>automated extraction workbench</h1>
        </div>
        <div className="hero-side">
          <div className={`status-chip ${apiOnline ? "online" : "offline"}`}>
            <span className={`dot ${apiOnline ? "ok" : "bad"}`} />
            API {apiOnline ? "online" : "offline"}
          </div>
          <div className="hero-metrics">
            <div className="metric-card">
              <span>Dataset</span>
              <strong>{csvSummary}</strong>
            </div>
            <div className="metric-card">
              <span>Features</span>
              <strong>{session.features.length}</strong>
            </div>
            <div className="metric-card">
              <span>Job status</span>
              <strong>{latestJobStatus}</strong>
            </div>
          </div>
        </div>
      </header>

      <nav className="tabs" role="tablist" aria-label="Calibrator views">
        <button
          className={`tab-btn ${activeTab === "setup" ? "active" : ""}`}
          onClick={() => setActiveTab("setup")}
          type="button"
          role="tab"
          aria-selected={activeTab === "setup"}
        >
          Setup
        </button>
        <button
          className={`tab-btn ${activeTab === "schema" ? "active" : ""}`}
          onClick={() => setActiveTab("schema")}
          type="button"
          role="tab"
          aria-selected={activeTab === "schema"}
        >
          Feature set
        </button>
        <button
          className={`tab-btn ${activeTab === "experiments" ? "active" : ""}`}
          onClick={() => setActiveTab("experiments")}
          type="button"
          role="tab"
          aria-selected={activeTab === "experiments"}
        >
          Experiments
        </button>
        <button
          className={`tab-btn ${activeTab === "test" ? "active" : ""}`}
          onClick={() => setActiveTab("test")}
          type="button"
          role="tab"
          aria-selected={activeTab === "test"}
        >
          Test
        </button>
      </nav>

      {activeTab === "setup" && (
        <section className="view panel-grid">
          <article className="panel card">
            <div className="section-title">
              <h2>Data Source</h2>
              <p>Pick a CSV and it loads immediately.</p>
            </div>
            <label>
              CSV file
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleCsvFileChange}
                disabled={loadCsvByFileMutation.isPending || loadCsvByPathMutation.isPending}
              />
            </label>
            <p className="muted">
              {loadCsvByFileMutation.isPending
                ? `Loading ${selectedFileName || "file"}...`
                : "No extra click needed after choosing a file."}
            </p>
            <p className="muted">
              Active dataset: <b>{csvData?.source || "none loaded"}</b>
            </p>

            <details className="path-advanced">
              <summary>Advanced: load by absolute path</summary>
              <label>
                CSV path
                <input
                  value={session.csv_path}
                  onChange={(event) => setSession((prev) => ({ ...prev, csv_path: event.target.value }))}
                  onKeyDown={handleCsvPathInputKeyDown}
                  placeholder="/absolute/path/to/reports.csv"
                />
              </label>
              <div className="action-row">
                <button
                  className="btn btn-secondary"
                  onClick={() => handleLoadCsvByPath()}
                  disabled={loadCsvByPathMutation.isPending || loadCsvByFileMutation.isPending}
                  type="button"
                >
                  {loadCsvByPathMutation.isPending ? "Loading path..." : "Load path"}
                </button>
              </div>
            </details>

            <label>
              Preview rows
              <input
                type="number"
                min={1}
                max={100}
                value={previewRows}
                onChange={(event) => setPreviewRows(Number(event.target.value || 8))}
              />
            </label>

            {csvData && (
              <div className="meta-grid">
                <div>Rows: {csvData.row_count}</div>
                <div>Columns: {csvData.columns.length}</div>
                <div>Encoding: {csvData.encoding}</div>
                <div>Delimiter: {csvData.delimiter}</div>
              </div>
            )}
          </article>

          <article className="panel card">
            <div className="section-title">
              <h2>Mapping and Inference</h2>
              <p>Set source columns and local model runtime options.</p>
            </div>
            <label>
              ID column
              <select
                value={idColumnSelectValue}
                onChange={(event) => setSession((prev) => ({ ...prev, id_column: event.target.value }))}
              >
                <option value="">Select ID column</option>
                {csvData?.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Report column
              <select
                value={reportColumnSelectValue}
                onChange={(event) => setSession((prev) => ({ ...prev, report_column: event.target.value }))}
              >
                <option value="">Select Report column</option>
                {csvData?.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>

            <div className="llama-setup">
              <div className="section-title">
                <h3>Local model</h3>
                <p>Choose a GGUF model. The local server starts automatically when you run tests.</p>
              </div>

              <label>
                Model file
                <select
                  value={selectedLocalModelPath}
                  onChange={(event) => setSelectedLocalModelPath(event.target.value)}
                >
                  <option value="">Select a local model</option>
                  {localModelPaths.map((modelPath) => (
                    <option key={modelPath} value={modelPath}>
                      {modelPath}
                    </option>
                  ))}
                </select>
              </label>

              <div className="action-row">
                <button
                  className="btn btn-secondary"
                  onClick={handleRefreshLocalModels}
                  disabled={listLocalModelsMutation.isPending}
                  type="button"
                >
                  {listLocalModelsMutation.isPending ? "Scanning models..." : "Refresh models"}
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={handleStopLlamaServer}
                  disabled={stopLlamaMutation.isPending}
                  type="button"
                >
                  {stopLlamaMutation.isPending ? "Stopping server..." : "Stop local server"}
                </button>
              </div>

              <div className="hf-install">
                <div className="section-title">
                  <h3>Install from Hugging Face</h3>
                  <p>Search GGUF repositories and install directly without leaving this page.</p>
                </div>

                <div className="field-grid two-col">
                  <label>
                    Search query
                    <input
                      value={hfSearchQuery}
                      onChange={(event) => setHfSearchQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key !== "Enter") {
                          return;
                        }
                        event.preventDefault();
                        handleSearchHfModels();
                      }}
                      placeholder="phi-3 gguf, llama 3.2 gguf, mistral gguf..."
                    />
                  </label>
                  <label>
                    Result limit
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={hfSearchLimit}
                      onChange={(event) => setHfSearchLimit(Number(event.target.value || 12))}
                    />
                  </label>
                </div>

                <div className="action-row">
                  <button
                    className="btn btn-secondary"
                    onClick={handleSearchHfModels}
                    disabled={searchHfModelsMutation.isPending}
                    type="button"
                  >
                    {searchHfModelsMutation.isPending ? "Searching..." : "Search Hugging Face"}
                  </button>
                </div>

                <label>
                  Repository
                  <select
                    value={selectedHfRepoId}
                    onChange={(event) => handleSelectHfRepo(event.target.value)}
                    disabled={hfSearchResults.length === 0}
                  >
                    {hfSearchResults.length === 0 && <option value="">Search to list repositories</option>}
                    {hfSearchResults.map((model) => (
                      <option key={model.repo_id} value={model.repo_id}>
                        {model.repo_id} ({model.gguf_files.length} files)
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  GGUF file
                  <select
                    value={selectedHfFile}
                    onChange={(event) => setSelectedHfFile(event.target.value)}
                    disabled={!selectedHfRepoId || listHfFilesMutation.isPending}
                  >
                    {hfRepoFiles.length === 0 && (
                      <option value="">
                        {selectedHfRepoId ? "No GGUF files found" : "Select a repository first"}
                      </option>
                    )}
                    {hfRepoFiles.map((file) => (
                      <option key={file.file_name} value={file.file_name}>
                        {file.file_name} ({formatModelSize(file.size_gb)})
                      </option>
                    ))}
                  </select>
                </label>

                <div className="action-row">
                  <button
                    className="btn btn-primary"
                    onClick={handleInstallHfModel}
                    disabled={
                      !selectedHfRepoId ||
                      !selectedHfFile ||
                      downloadHfModelMutation.isPending ||
                      listHfFilesMutation.isPending
                    }
                    type="button"
                  >
                    {downloadHfModelMutation.isPending ? "Installing model..." : "Install selected model"}
                  </button>
                </div>

                <details className="path-advanced">
                  <summary>Advanced install settings</summary>
                  <label>
                    Destination directory
                    <input
                      value={hfDestinationDir}
                      onChange={(event) => setHfDestinationDir(event.target.value)}
                      placeholder="Default: ~/models"
                    />
                  </label>
                  <label>
                    Hugging Face token (optional)
                    <input
                      type="password"
                      value={hfToken}
                      onChange={(event) => setHfToken(event.target.value)}
                      placeholder="Needed for gated/private models"
                    />
                  </label>
                </details>

                <p className="muted">
                  {selectedHfRepo
                    ? `${selectedHfRepo.repo_id} • ${selectedHfRepo.downloads.toLocaleString()} downloads • ${selectedHfRepo.likes.toLocaleString()} likes`
                    : "Search Hugging Face to install another local GGUF model."}
                </p>
                {selectedHfFileInfo && (
                  <p className="muted">
                    Selected file size: <b>{formatModelSize(selectedHfFileInfo.size_gb)}</b>
                  </p>
                )}
                {listHfFilesMutation.isPending && <p className="muted">Loading repository file list...</p>}
              </div>

              <details className="path-advanced">
                <summary>Advanced runtime settings</summary>
                <label>
                  Model file path
                  <input
                    value={selectedLocalModelPath}
                    onChange={(event) => setSelectedLocalModelPath(event.target.value)}
                    placeholder="/absolute/path/to/model.gguf"
                  />
                </label>

                <div className="field-grid two-col">
                  <label>
                    Server port
                    <input
                      type="number"
                      min={1}
                      max={65535}
                      value={llamaPort}
                      onChange={(event) => setLlamaPort(Number(event.target.value || 8080))}
                    />
                  </label>
                  <label>
                    Context size
                    <input
                      type="number"
                      min={256}
                      max={131072}
                      value={llamaCtxSize}
                      onChange={(event) => setLlamaCtxSize(Number(event.target.value || 8192))}
                    />
                  </label>
                </div>
              </details>

              <p className="muted">
                {llamaStatus
                  ? `Local server ${llamaStatus.process_running ? "running" : "idle"} at ${llamaStatus.llama_url}.`
                  : `Local server will run at http://127.0.0.1:${llamaPort}.`}
              </p>
              <p className="muted">
                llama-server binary: {llamaBinaryPath || "not found in PATH"}
              </p>
              {ensureLlamaMutation.isPending && <p className="muted">Preparing local model server...</p>}
              {stopLlamaMutation.isPending && <p className="muted">Stopping local model server...</p>}
              {llamaStatus?.connect_error && !ensureLlamaMutation.isPending && <p className="error">{llamaStatus.connect_error}</p>}
            </div>

            <div className="field-grid two-col">
              <label>
                Temperature
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.1}
                  value={session.temperature}
                  onChange={(event) =>
                    setSession((prev) => ({ ...prev, temperature: Number(event.target.value || 0) }))
                  }
                />
              </label>
              <label>
                Max retries
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={session.max_retries}
                  onChange={(event) =>
                    setSession((prev) => ({ ...prev, max_retries: Number(event.target.value || 5) }))
                  }
                />
              </label>
            </div>
          </article>

          <article className="panel card wide">
            <div className="section-title">
              <h2>CSV Preview</h2>
              <p>Quick sanity check for detected columns and row content.</p>
            </div>
            {!csvData && <p className="muted">Load a CSV to view columns and sample rows.</p>}
            {csvData && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Row</th>
                      {csvData.columns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csvData.preview.map((row) => (
                      <tr key={row.row_number}>
                        <td>{row.row_number}</td>
                        {csvData.columns.map((column) => (
                          <td key={`${row.row_number}-${column}`}>{row.values[column] ?? ""}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </section>
      )}

      {activeTab === "schema" && (
        <section className="view schema-layout">
          <aside className="panel card feature-list">
            <div className="section-title">
              <h2>Feature Library</h2>
              <p>{session.features.length} features in current feature set.</p>
            </div>
            <div className="action-row">
              <button className="btn btn-primary" onClick={handleAddFeature} type="button">
                Add feature
              </button>
              <button className="btn btn-ghost" onClick={handleDeleteFeature} type="button">
                Delete
              </button>
            </div>

            <ul>
              {session.features.map((feature, index) => (
                <li key={`${feature.name}-${index}`}>
                  <button
                    type="button"
                    className={`feature-item ${index === selectedFeatureIndex ? "selected" : ""}`}
                    onClick={() => setSelectedFeatureIndex(index)}
                  >
                    {feature.name || `feature_${index + 1}`}
                  </button>
                </li>
              ))}
            </ul>
          </aside>

          <article className="panel card feature-editor">
            <div className="section-title">
              <h2>Feature Editor</h2>
              <p>Selected feature: <b>{activeFeature}</b></p>
            </div>

            <input
              ref={schemaFileInputRef}
              type="file"
              accept=".json,application/json"
              onChange={handleSchemaFileInputChange}
              style={{ display: "none" }}
            />

            <p className="muted">
              Feature set target: <b>{schemaFileHandle?.name || "Choose when saving"}</b>
            </p>
            <p className="muted">
              Load opens a file picker. Save opens a save dialog unless overwriting the current picked feature-set
              file.
            </p>

            <div className="action-row">
              <button
                className="btn btn-secondary"
                onClick={() => void handleLoadSchema()}
                disabled={isLocalSchemaIOPending}
                type="button"
              >
                {isLocalSchemaIOPending ? "Working..." : "Load feature set"}
              </button>
              <button
                className="btn btn-primary"
                onClick={() => void handleSaveSchema()}
                disabled={isLocalSchemaIOPending}
                type="button"
              >
                {isLocalSchemaIOPending ? "Saving..." : "Save feature set"}
              </button>
            </div>

            {!selectedFeature && <p className="muted">Select a feature.</p>}
            {selectedFeature && (
              <>
                <div className="meta-grid">
                  <div>
                    Model sees: <b>Name</b>, <b>Description</b>, <b>Missing value</b>,{" "}
                    <b>Allowed values or Type</b>, and <b>Prompt guidance</b>.
                  </div>
                  <div>
                    Use <b>Description</b> for what the feature means. Use <b>Prompt guidance</b> for extraction rules.
                  </div>
                </div>

                <label>
                  Name
                  <input
                    value={selectedFeature.name}
                    onChange={(event) => updateFeature(selectedFeatureIndex, { name: event.target.value })}
                  />
                </label>

                <label>
                  Description (model sees this)
                  <textarea
                    rows={3}
                    value={selectedFeature.description}
                    onChange={(event) =>
                      updateFeature(selectedFeatureIndex, { description: event.target.value })
                    }
                  />
                </label>

                <div className="field-grid two-col">
                  <label>
                    Missing value
                    <input
                      value={selectedFeature.missing_value_rule}
                      onChange={(event) =>
                        updateFeature(selectedFeatureIndex, {
                          missing_value_rule: event.target.value,
                        })
                      }
                    />
                  </label>

                  <label>
                    Type
                    <select
                      value={selectedTypeHintValue}
                      onChange={(event) =>
                        updateFeature(selectedFeatureIndex, {
                          type_hint: event.target.value,
                        })
                      }
                    >
                      {typeHintOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <label>
                  Allowed values (optional, comma separated)
                  <input
                    value={(selectedFeature.allowed_values ?? []).join(", ")}
                    onChange={(event) => {
                      const allowedValues = event.target.value
                        .split(",")
                        .map((part) => part.trim())
                        .filter(Boolean);
                      updateFeature(selectedFeatureIndex, {
                        allowed_values: allowedValues.length > 0 ? allowedValues : undefined,
                      });
                    }}
                    placeholder="normal, reduced, NA"
                  />
                </label>
                <p className="muted">
                  If you provide allowed values, those labels are prioritized during extraction.
                </p>

                <label>
                  Prompt guidance (model sees this)
                  <textarea
                    rows={6}
                    value={selectedFeature.prompt}
                    onChange={(event) => updateFeature(selectedFeatureIndex, { prompt: event.target.value })}
                    placeholder="Instructions for how to extract this feature from report text."
                  />
                </label>
              </>
            )}
          </article>
        </section>
      )}

      {activeTab === "experiments" && (
        <section className="view schema-layout">
          <aside className="panel card feature-list">
            <div className="section-title">
              <h2>Experiment Profiles</h2>
              <p>{session.experiment_profiles.length} prompt variants in this session.</p>
            </div>
            <div className="action-row">
              <button className="btn btn-primary" onClick={handleAddExperiment} type="button">
                Add profile
              </button>
              <button className="btn btn-ghost" onClick={handleDeleteExperiment} type="button">
                Delete
              </button>
            </div>
            <ul>
              {session.experiment_profiles.map((experiment, index) => (
                <li key={experiment.id}>
                  <button
                    type="button"
                    className={`feature-item ${index === selectedExperimentIndex ? "selected" : ""}`}
                    onClick={() => setSelectedExperimentIndex(index)}
                    aria-current={session.active_experiment_id === experiment.id ? "true" : undefined}
                  >
                    {experiment.name}
                    {session.active_experiment_id === experiment.id ? " • primary" : ""}
                    {activeRunExperimentIds.includes(experiment.id) ? " • run" : ""}
                  </button>
                </li>
              ))}
            </ul>
          </aside>

          <article className="panel card feature-editor">
            <div className="section-title">
              <h2>Prompt Strategy Editor</h2>
              <p>Active profile for runs: <b>{activeExperimentName}</b></p>
            </div>

            {!selectedExperiment && <p className="muted">Select an experiment profile.</p>}
            {selectedExperiment && (
              <>
                <div className="action-row">
                  <button
                    className={`btn ${session.active_experiment_id === selectedExperiment.id ? "btn-primary" : "btn-secondary"}`}
                    onClick={() => setActiveExperiment(selectedExperiment.id)}
                    type="button"
                  >
                    {session.active_experiment_id === selectedExperiment.id
                      ? "Primary for single tests"
                      : "Set as primary for single tests"}
                  </button>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={activeRunExperimentIds.includes(selectedExperiment.id)}
                      onChange={(event) =>
                        setRunExperimentEnabled(selectedExperiment.id, event.target.checked)
                      }
                    />
                    Include in full run batches
                  </label>
                </div>

                <div className="meta-grid">
                  <div>
                    Keep one profile as your baseline and clone variants for prompt wording tests.
                  </div>
                  <div>
                    Reasoning and judge settings are applied to both single-feature tests and full runs.
                  </div>
                </div>

                <label>
                  Profile name
                  <input
                    value={selectedExperiment.name}
                    onChange={(event) => {
                      updateExperiment(selectedExperimentIndex, { name: event.target.value });
                    }}
                  />
                </label>

                <label>
                  Reasoning mode
                  <select
                    value={selectedExperiment.reasoning_mode}
                    onChange={(event) =>
                      updateExperiment(selectedExperimentIndex, {
                        reasoning_mode: event.target.value as ReasoningMode,
                      })
                    }
                  >
                    {REASONING_MODE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  System instructions (optional)
                  <textarea
                    rows={3}
                    value={selectedExperiment.system_instructions}
                    onChange={(event) =>
                      updateExperiment(selectedExperimentIndex, {
                        system_instructions: event.target.value,
                      })
                    }
                    placeholder="Global role and safety framing for the extraction model."
                  />
                </label>

                <label>
                  Extraction instructions
                  <textarea
                    rows={4}
                    value={selectedExperiment.extraction_instructions}
                    onChange={(event) =>
                      updateExperiment(selectedExperimentIndex, {
                        extraction_instructions: event.target.value,
                      })
                    }
                    placeholder="Rules for evidence use, missing values, and ambiguity handling."
                  />
                </label>

                <label>
                  Reasoning instructions (optional override)
                  <textarea
                    rows={4}
                    value={selectedExperiment.reasoning_instructions}
                    onChange={(event) =>
                      updateExperiment(selectedExperimentIndex, {
                        reasoning_instructions: event.target.value,
                      })
                    }
                    placeholder="Leave empty to use mode defaults. Add custom ReAct-style guidance when needed."
                  />
                </label>

                <label>
                  Output instructions
                  <textarea
                    rows={3}
                    value={selectedExperiment.output_instructions}
                    onChange={(event) =>
                      updateExperiment(selectedExperimentIndex, {
                        output_instructions: event.target.value,
                      })
                    }
                    placeholder='Example: Return only {"value": "..."} with no extra keys.'
                  />
                </label>

                <div className="llama-setup">
                  <div className="section-title">
                    <h3>Judge model</h3>
                    <p>Second-pass QA to score whether extracted values are evidence-supported.</p>
                  </div>

                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={selectedExperiment.judge.enabled}
                      onChange={(event) =>
                        updateExperimentJudge(selectedExperimentIndex, { enabled: event.target.checked })
                      }
                    />
                    Enable judge pass
                  </label>

                  {selectedExperiment.judge.enabled && (
                    <>
                      <label>
                        Judge model
                        <select
                          value={selectedJudgeModelValue}
                          onChange={(event) =>
                            updateExperimentJudge(selectedExperimentIndex, { model: event.target.value })
                          }
                        >
                          <option value="">Select a judge model</option>
                          <option value={JUDGE_MODEL_USE_EXTRACTION}>Use extraction model</option>
                          {judgeModelOptions.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      </label>
                      <p className="muted">
                        Judge runs only when a model is selected.
                      </p>

                      <label>
                        Accept threshold (0-1)
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={selectedExperiment.judge.acceptance_threshold}
                          onChange={(event) =>
                            updateExperimentJudge(selectedExperimentIndex, {
                              acceptance_threshold: Number(event.target.value || 0.6),
                            })
                          }
                        />
                      </label>

                      <label>
                        Judge instructions
                        <textarea
                          rows={4}
                          value={selectedExperiment.judge.instructions}
                          onChange={(event) =>
                            updateExperimentJudge(selectedExperimentIndex, {
                              instructions: event.target.value,
                            })
                          }
                          placeholder="Instructions for adjudicating support from report evidence."
                        />
                      </label>
                    </>
                  )}
                </div>
              </>
            )}
          </article>
        </section>
      )}

      {activeTab === "test" && (
        <section className="view panel-grid">
          <article className="panel card">
            <div className="section-title">
              <h2>Sample Input</h2>
              <p>Choose report(s) and run one feature or your full feature set.</p>
            </div>

            <label>
              Primary experiment profile (single-feature test)
              <select
                value={session.active_experiment_id}
                onChange={(event) => {
                  const nextId = event.target.value;
                  setActiveExperiment(nextId);
                  const nextIndex = session.experiment_profiles.findIndex((profile) => profile.id === nextId);
                  if (nextIndex >= 0) {
                    setSelectedExperimentIndex(nextIndex);
                  }
                }}
              >
                {session.experiment_profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="llama-setup">
              <div className="section-title">
                <h3>Experiments for full runs</h3>
                <p>Select one or more setups to execute in the same batch.</p>
              </div>
              {session.experiment_profiles.map((profile) => (
                <label key={`run-select-${profile.id}`} className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={activeRunExperimentIds.includes(profile.id)}
                    onChange={(event) => setRunExperimentEnabled(profile.id, event.target.checked)}
                  />
                  {profile.name}
                </label>
              ))}
            </div>

            <div className="field-grid two-col">
              <label>
                Starting row number
                <input
                  type="number"
                  min={1}
                  max={Math.max(sampleRows.length, 1)}
                  value={safeSampleIndex}
                  onChange={(event) =>
                    setSession((prev) => ({
                      ...prev,
                      sample_index: Number(event.target.value || 1),
                    }))
                  }
                />
              </label>

              <label>
                Reports to test (max 20)
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={safeReportsToTest}
                  onChange={(event) => setReportsToTest(Number(event.target.value || 1))}
                />
              </label>
            </div>

            <p className="muted">
              Loaded preview rows: {sampleRows.length}. Ready to test <b>{selectedBatchReports.length}</b> report(s)
              x <b>{activeRunExperiments.length}</b> experiment(s) from column <b>{session.report_column}</b>.
            </p>
            <div className="meta-grid">
              <div>Primary profile: <b>{activeExperimentName}</b></div>
              <div>Batch profiles: <b>{activeRunExperimentNames}</b></div>
              <div>Reasoning: {activeExperiment?.reasoning_mode ?? "direct"}</div>
              <div>Judge pass: {activeExperimentJudgeConfigured ? "enabled" : "disabled"}</div>
            </div>

            <label>
              Report text
              <textarea rows={14} value={reportText} onChange={(event) => setReportText(event.target.value)} />
            </label>

            <div className="action-row">
              <button
                className="btn btn-primary"
                onClick={handleRunAllTests}
                disabled={
                  testBatchMutation.isPending ||
                  isJobRunning ||
                  ensureLlamaMutation.isPending ||
                  activeRunExperiments.length === 0
                }
                type="button"
              >
                {ensureLlamaMutation.isPending
                  ? "Preparing model..."
                  : testBatchMutation.isPending || isJobRunning
                    ? "Running report set..."
                    : "Run full feature set"}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleRunSingleTest}
                disabled={testFeatureMutation.isPending || ensureLlamaMutation.isPending}
                type="button"
              >
                {ensureLlamaMutation.isPending
                  ? "Preparing model..."
                  : testFeatureMutation.isPending
                    ? "Testing feature..."
                    : "Test selected feature"}
              </button>
              {(isJobRunning || cancelJobMutation.isPending || isCancelRequested) && (
                <button
                  className="btn btn-ghost"
                  onClick={() => cancelJobMutation.mutate(jobId)}
                  disabled={!isJobRunning || cancelJobMutation.isPending || isCancelRequested}
                  type="button"
                >
                  {cancelJobMutation.isPending
                    ? "Cancelling..."
                    : isCancelRequested
                      ? "Cancel requested"
                      : "Cancel all"}
                </button>
              )}
            </div>
          </article>

          <article className="panel card">
            <div className="section-title">
              <h2>Run Summary</h2>
              <p>Simple status updates while your report set is processed.</p>
            </div>

            <div className="run-summary-card">
              <p className="run-summary-text">{runSummaryText}</p>
              <p className="muted">{runDetailText}</p>
            </div>

            <div className="action-row">
              <button
                className="btn btn-secondary"
                onClick={handleExportRunCsv}
                disabled={!hasRunResultsForExport}
                type="button"
              >
                Export current run CSV
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleExportRunHistoryCsv}
                disabled={!hasRunHistoryForExport}
                type="button"
              >
                Export run comparison CSV
              </button>
            </div>

            {reportsTotal > 0 && (
              <div className="meta-grid">
                <div>Primary profile: {activeExperimentName}</div>
                <div>Batch profiles: {activeRunExperimentNames}</div>
                <div>
                  Run items done: {reportsCompleted}/{Math.max(reportsTotal, reportsCompleted, 1)}
                </div>
                <div>Features per report: {session.features.length}</div>
                <div>
                  Total checks: {jobQuery.data?.completed ?? 0}/{jobQuery.data?.total ?? 0}
                </div>
              </div>
            )}

            {singleResult && (
              <div className="result-box">
                <h3>Selected Feature</h3>
                <p>
                  <b>{singleResult.feature_name}</b>: <code>{singleResult.value}</code>
                </p>
                {singleResult.experiment_name && (
                  <p className="muted">Experiment: {singleResult.experiment_name}</p>
                )}
                {singleResult.error && <p className="error">{singleResult.error}</p>}
                {singleResult.judge_result && (
                  <div className="meta-grid">
                    <div>Judge: {singleResult.judge_result.status}</div>
                    <div>
                      Score:{" "}
                      {typeof singleResult.judge_result.score === "number"
                        ? singleResult.judge_result.score.toFixed(2)
                        : "NA"}
                    </div>
                    <div>
                      Model: {singleResult.judge_result.model || "same as extractor"}
                    </div>
                    {singleResult.judge_result.rationale && (
                      <div>Rationale: {singleResult.judge_result.rationale}</div>
                    )}
                    {singleResult.judge_result.error && (
                      <div className="error">{singleResult.judge_result.error}</div>
                    )}
                  </div>
                )}
                <details>
                  <summary>Raw response</summary>
                  <pre>{singleResult.raw_response}</pre>
                </details>
                {singleResult.judge_result?.raw_response && (
                  <details>
                    <summary>Judge raw response</summary>
                    <pre>{singleResult.judge_result.raw_response}</pre>
                  </details>
                )}
              </div>
            )}

            {jobId && (
              <details>
                <summary>Technical details</summary>
                <p className="muted">Run id: {jobId}</p>
                <p className="muted">Status code: {currentJobStatus}</p>
                <p className="muted">Batch profiles: {activeRunExperimentNames}</p>
              </details>
            )}

            {(runJudgeSummary.accepted > 0 ||
              runJudgeSummary.rejected > 0 ||
              runJudgeSummary.uncertain > 0 ||
              runJudgeSummary.judgeError > 0) && (
              <div className="meta-grid">
                <div>Judge accepted: {runJudgeSummary.accepted}</div>
                <div>Judge rejected: {runJudgeSummary.rejected}</div>
                <div>Judge uncertain: {runJudgeSummary.uncertain}</div>
                <div>Judge errors: {runJudgeSummary.judgeError}</div>
              </div>
            )}
          </article>

          <article className="panel card wide">
            <div className="section-title">
              <h2>Report-by-Report Results</h2>
              <p>Each report with extracted feature values, so review is straightforward.</p>
            </div>
            {displayReportResults.length === 0 && (
              <p className="muted">Run the full feature set to see results across multiple reports.</p>
            )}
            {displayReportResults.length > 0 && (
              <div className="report-results-stack">
                {displayReportResults.map((reportResult) => (
                  <details
                    key={`${reportResult.run_index ?? reportResult.report_index}-${reportResult.experiment_id ?? "default"}-${reportResult.row_number ?? "manual"}`}
                    className="report-result-card"
                  >
                    <summary>
                      <span>
                        Report {reportResult.report_index}
                        {reportResult.row_number ? ` (row ${reportResult.row_number})` : ""}
                      </span>
                      {reportResult.experiment_name && (
                        <span className="muted">Experiment: {reportResult.experiment_name}</span>
                      )}
                      <span className="muted">
                        {truncateText(reportResult.report_text, 110)}
                      </span>
                    </summary>

                    <div className="report-result-body">
                      <label>
                        Report text
                        <textarea rows={6} value={reportResult.report_text} readOnly />
                      </label>

                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Feature</th>
                              <th>Status</th>
                              <th>Extracted value</th>
                              <th>Duration (ms)</th>
                              <th>Judge</th>
                              <th>Raw</th>
                            </tr>
                          </thead>
                          <tbody>
                            {reportResult.results.map((result, index) => (
                              <tr
                                key={`${reportResult.report_index}-${reportResult.experiment_id ?? "default"}-${result.feature_name}-${index}`}
                              >
                                <td>{result.feature_name}</td>
                                <td>
                                  <span className={`status-pill ${result.status}`}>{result.status}</span>
                                  {result.error && <div className="error">{result.error}</div>}
                                </td>
                                <td>{result.value}</td>
                                <td>{result.duration_ms}</td>
                                <td>
                                  {!result.judge_result && <span className="muted">n/a</span>}
                                  {result.judge_result && (
                                    <div className="judge-cell">
                                      <span className={`status-pill ${result.judge_result.status}`}>
                                        {result.judge_result.status}
                                      </span>
                                      {typeof result.judge_result.score === "number" && (
                                        <div>{result.judge_result.score.toFixed(2)}</div>
                                      )}
                                      {result.judge_result.error && (
                                        <div className="error">{result.judge_result.error}</div>
                                      )}
                                      {result.judge_result.rationale && (
                                        <div className="muted">{truncateText(result.judge_result.rationale, 90)}</div>
                                      )}
                                      {result.judge_result.raw_response && (
                                        <details>
                                          <summary>Judge raw</summary>
                                          <pre>{result.judge_result.raw_response}</pre>
                                        </details>
                                      )}
                                    </div>
                                  )}
                                </td>
                                <td>
                                  <details>
                                    <summary>View</summary>
                                    <pre>{result.raw_response}</pre>
                                  </details>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </details>
                ))}
              </div>
            )}
          </article>

          <article className="panel card wide">
            <div className="section-title">
              <h2>Run Comparison</h2>
              <p>Recent runs by experiment profile so prompt variants are easy to compare.</p>
            </div>
            {runHistory.length === 0 && (
              <p className="muted">No completed runs yet. Run at least one batch to populate comparison metrics.</p>
            )}
            {runHistory.length > 0 && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Experiments</th>
                      <th>Status</th>
                      <th>Started</th>
                      <th>Run items</th>
                      <th>Checks</th>
                      <th>OK</th>
                      <th>LLM err</th>
                      <th>Parse err</th>
                      <th>Judge A/R/U</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runHistory.map((item) => (
                      <tr key={item.jobId}>
                        <td>{item.experimentNames}</td>
                        <td>
                          <span className={`status-pill ${item.status}`}>{item.status}</span>
                        </td>
                        <td>{new Date(item.startedAtUnix * 1000).toLocaleString()}</td>
                        <td>{item.reportsReviewed}</td>
                        <td>
                          {item.checksCompleted}/{item.checksTotal}
                        </td>
                        <td>{item.okCount}</td>
                        <td>{item.llmErrorCount}</td>
                        <td>{item.parseErrorCount}</td>
                        <td>
                          {item.judgeAccepted}/{item.judgeRejected}/{item.judgeUncertain}
                          {item.judgeErrors > 0 ? ` (+${item.judgeErrors} err)` : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        </section>
      )}

      <section className="diagnostics panel card">
        <h2>Last Action</h2>
        <p>
          CSV load: {loadCsvByPathMutation.status}/{loadCsvByFileMutation.status}. Feature-set IO:{" "}
          {isLocalSchemaIOPending ? "pending" : "idle"}. Llama setup: {listLocalModelsMutation.status}/
          {ensureLlamaMutation.status}. HF install:{" "}
          {searchHfModelsMutation.status}/{listHfFilesMutation.status}/{downloadHfModelMutation.status}. Tests:{" "}
          {testFeatureMutation.status}/{testBatchMutation.status}. Job fetch: {jobQuery.status}.
        </p>
      </section>

      <div className="toast-stack">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.kind}`}>
            {toast.message}
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
