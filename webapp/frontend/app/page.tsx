"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// Icons are imported by direct path rather than from the package barrel: the barrel pulls in
// every icon and noticeably slows the dev server.
import ArrowForwardRounded from "@mui/icons-material/ArrowForwardRounded";
import CheckRounded from "@mui/icons-material/CheckRounded";
import CloudUploadOutlined from "@mui/icons-material/CloudUploadOutlined";
import DownloadRounded from "@mui/icons-material/DownloadRounded";
import ErrorOutlineRounded from "@mui/icons-material/ErrorOutlineRounded";
import GraphicEqRounded from "@mui/icons-material/GraphicEqRounded";
import InfoOutlined from "@mui/icons-material/InfoOutlined";
import InsertDriveFileOutlined from "@mui/icons-material/InsertDriveFileOutlined";
import MusicNoteRounded from "@mui/icons-material/MusicNoteRounded";
import PlayArrowRounded from "@mui/icons-material/PlayArrowRounded";
import ScheduleRounded from "@mui/icons-material/ScheduleRounded";
import SmartDisplayOutlined from "@mui/icons-material/SmartDisplayOutlined";
import SubtitlesOutlined from "@mui/icons-material/SubtitlesOutlined";

type Direction = "es-en" | "en-es";

type Segment = {
  index: number;
  start: number;
  end: number;
  source_text: string;
  translated_text: string;
  original_duration: number;
  raw_tts_duration: number;
  final_duration: number;
  scale_applied: number;
  clamped: boolean;
  error: string;
};

type DubResult = {
  job_id: string;
  direction: string;
  segments: Segment[];
  speaker_similarity: number | null;
  duration_match_ratio: number | null;
  raw_duration_ratio: number | null;
  clamped_segments: number;
  failed_segments: number;
  background_mode?: string;
  elapsed_s: number;
};

type Job = {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  /** Seconds remaining, measured from how long this machine actually took per segment.
   *  Null until two segments have been timed — the first carries one-off warm-up and
   *  extrapolating from it alone reported 135s on a job that finished in 68s. */
  eta_s?: number | null;
  result?: DubResult;
};

type Health = {
  ok: boolean;
  ffmpeg: boolean;
  xtts_license_accepted: boolean;
  mt_mode: string;
  models_loaded: boolean;
  max_upload_mb: number;
};

const DIRECTIONS: { id: Direction; from: string; to: string; codeFrom: string; codeTo: string }[] = [
  { id: "es-en", from: "Spanish", to: "English", codeFrom: "ES", codeTo: "EN" },
  { id: "en-es", from: "English", to: "Spanish", codeFrom: "EN", codeTo: "ES" },
];

// Plain descriptions of what happens, not which model does it -- the model choices are an
// implementation detail and naming them dates the page every time one changes.
const STAGES = [
  "Extract audio",
  "Separate voice and background",
  "Transcribe speech",
  "Translate text",
  "Synthesise voice",
  "Mix and merge",
];

/** Map backend progress percentages onto the stage list. */
function stageFromProgress(p: number): number {
  if (p < 0) return -1;
  if (p < 5) return 0;
  if (p < 10) return 1;
  if (p < 20) return 2;
  if (p < 88) return 4; // the segment loop translates and synthesises together
  if (p < 100) return 5;
  return STAGES.length;
}

function fmtEta(seconds: number): string {
  if (seconds < 45) return "under a minute left";
  const minutes = Math.round(seconds / 60);
  return `about ${minutes} min left`;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [direction, setDirection] = useState<Direction>("es-en");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string>("");
  const [over, setOver] = useState(false);
  // -1 means "no upload in flight"; 0-100 is a live upload percentage.
  const [uploadPct, setUploadPct] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() =>
        setHealth({
          ok: false,
          ffmpeg: false,
          xtts_license_accepted: false,
          mt_mode: "unavailable",
          models_loaded: false,
          max_upload_mb: 200,
        }),
      );
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const res = await fetch(`/api/jobs/${id}`);
          if (!res.ok) return;
          const data: Job = await res.json();
          setJob(data);
          if (data.status === "done" || data.status === "error") stopPolling();
        } catch {
          /* transient network error: keep polling */
        }
      }, 1500);
    },
    [stopPolling],
  );

  async function submit() {
    if (!file) return;
    setError("");
    setJob(null);
    const body = new FormData();
    body.append("file", file);
    body.append("direction", direction);
    // XMLHttpRequest rather than fetch: fetch() exposes no upload-progress events, so a large
    // file would upload with no feedback at all. XHR gives us upload.onprogress.
    setUploadPct(0);
    await new Promise<void>((resolve) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/jobs");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setUploadPct(Math.round((e.loaded / e.total) * 100));
      };
      xhr.onload = () => {
        setUploadPct(-1);
        let data: { job_id?: string; detail?: string } = {};
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          setError(`Server returned ${xhr.status} with an unreadable body`);
          resolve();
          return;
        }
        if (xhr.status >= 200 && xhr.status < 300 && data.job_id) {
          setJob({ job_id: data.job_id, status: "queued", progress: 0, message: "Queued" });
          poll(data.job_id);
        } else {
          setError(data.detail ?? `Upload failed (HTTP ${xhr.status})`);
        }
        resolve();
      };
      xhr.onerror = () => {
        setUploadPct(-1);
        setError("Could not reach the backend. Is it running on port 8000?");
        resolve();
      };
      xhr.onabort = () => {
        setUploadPct(-1);
        resolve();
      };
      xhr.send(body);
    });
  }

  function pick(f: File | undefined) {
    if (!f) return;
    setFile(f);
    setJob(null);
    setError("");
  }

  const uploading = uploadPct >= 0;
  const busy = uploading || job?.status === "queued" || job?.status === "running";
  const result = job?.status === "done" ? job.result : undefined;
  const activeStage = stageFromProgress(job?.progress ?? -1);
  const pct = uploading ? uploadPct : (job?.progress ?? 0);

  return (
    <div className="wrap">
      <header className="top">
        <div>
          <h1>Bilingual Voice Dubbing</h1>
          <p>Dub a clip into another language, in the original speaker&rsquo;s voice.</p>
        </div>
        <span className={`pill ${health?.ok ? "ok" : health === null ? "" : "bad"}`}>
          <span className="dot" />
          {health === null ? "Checking" : health.ok ? "Ready" : "Setup needed"}
        </span>
      </header>

      <div className="layout">
        {/* ---------------- main column ---------------- */}
        <main className="main">
          <section className="card">
            <h2>
              <span className="step">1</span> Source
            </h2>
            <div
              className={`drop ${over ? "over" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                pick(e.dataTransfer.files?.[0]);
              }}
            >
              <CloudUploadOutlined className="dropicon" />
              <div className="big">Drop a video or audio file</div>
              <div className="small">
                or click to browse &middot; up to {health?.max_upload_mb ?? 200} MB
              </div>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="video/*,audio/*"
              hidden
              onChange={(e) => pick(e.target.files?.[0])}
            />
            {file && (
              <div className="filechip">
                <InsertDriveFileOutlined className="i16" />
                <span className="name">{file.name}</span>
                <span className="sz">
                  {uploading ? (
                    `${uploadPct}%`
                  ) : job ? (
                    <CheckRounded className="i16 ok" />
                  ) : (
                    fmtBytes(file.size)
                  )}
                </span>
              </div>
            )}
          </section>

          <section className="card">
            <div className="cardhead">
              <h2>Progress</h2>
              <span className="note">
                {result ? `done in ${Math.round(result.elapsed_s)}s` : busy ? `${pct}%` : "idle"}
              </span>
            </div>
            <div className="bar">
              <i style={{ width: `${pct}%` }} />
            </div>
            <div className="msg">
              <span>
                {uploading
                  ? `Uploading — ${uploadPct}%`
                  : (job?.message ?? "Choose a file, then start dubbing.")}
              </span>
              {!uploading && busy && (
                <span className="eta">
                  <ScheduleRounded className="i14" />
                  {typeof job?.eta_s === "number" ? fmtEta(job.eta_s) : "estimating…"}
                </span>
              )}
            </div>
            {(error || job?.status === "error") && (
              <div className="err-box">
                <ErrorOutlineRounded className="i16" />
                <span>{error || job?.message}</span>
              </div>
            )}
          </section>

          <section className="card">
            <h2>
              <span className="step">2</span> Result
            </h2>
            {result ? (
              <div className="resultrow">
                <video controls src={`/api/jobs/${result.job_id}/video`} />
                <div className="resultside">
                  {result.background_mode && result.background_mode !== "none" && (
                    <span className="chip">
                      <MusicNoteRounded className="i16" /> Music and effects kept
                    </span>
                  )}
                  <a href={`/api/jobs/${result.job_id}/video`} download>
                    <button className="btn ghost">
                      <DownloadRounded className="i18" /> Video
                    </button>
                  </a>
                  <a href={`/api/jobs/${result.job_id}/audio`} download>
                    <button className="btn ghost">
                      <GraphicEqRounded className="i18" /> Audio only
                    </button>
                  </a>
                </div>
              </div>
            ) : (
              <div className="empty">
                <SmartDisplayOutlined className="emptyicon" />
                The dubbed video appears here.
              </div>
            )}
          </section>

          <section className="card">
            <div className="cardhead">
              <h2>
                <span className="step">3</span> Segments
              </h2>
              {result && (
                <span className="note">
                  {result.segments.length} found
                  {result.clamped_segments > 0 && ` · ${result.clamped_segments} clamped`}
                  {result.failed_segments > 0 && ` · ${result.failed_segments} failed`}
                </span>
              )}
            </div>
            {result ? (
              <div className="segs">
                {result.segments.map((s) => (
                  <article className="seg" key={s.index}>
                    <div className="seghead">
                      <span className="time">
                        {s.start.toFixed(2)}s – {s.end.toFixed(2)}s
                      </span>
                      <span className="tags">
                        {s.clamped && (
                          <span className="tag clamp" title="Hit the time-compression limit">
                            clamped
                          </span>
                        )}
                        {s.error && <span className="tag err">failed</span>}
                      </span>
                    </div>
                    <div className="segtext">
                      <p className="src">{s.source_text || <em>no speech recognised</em>}</p>
                      <p className="tgt">{s.translated_text || <em>{s.error || "—"}</em>}</p>
                    </div>
                    {/* How much the dub had to be stretched or squeezed to fit the slot it
                        replaces -- the number that explains an unnatural-sounding segment. */}
                    <dl className="segstats">
                      <div>
                        <dt>Slot</dt>
                        <dd>{s.original_duration.toFixed(2)}s</dd>
                      </div>
                      <div>
                        <dt>Synthesised</dt>
                        <dd>{s.raw_tts_duration.toFixed(2)}s</dd>
                      </div>
                      <div>
                        <dt>After retiming</dt>
                        <dd>{s.final_duration.toFixed(2)}s</dd>
                      </div>
                      <div>
                        <dt>Speed</dt>
                        <dd className={s.clamped ? "warn" : undefined}>
                          {s.scale_applied.toFixed(2)}&times;
                        </dd>
                      </div>
                    </dl>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty">
                <SubtitlesOutlined className="emptyicon" />
                Each utterance, its translation and its timing appear here.
              </div>
            )}
          </section>
        </main>

        {/* ---------------- sidebar ---------------- */}
        <aside className="side">
          <section className="card">
            <h2>Direction</h2>
            <div className="dirs">
              {DIRECTIONS.map((d) => (
                <button
                  key={d.id}
                  className={`dirbtn ${direction === d.id ? "sel" : ""}`}
                  onClick={() => setDirection(d.id)}
                  disabled={busy}
                >
                  <span className="code">{d.codeFrom}</span>
                  <ArrowForwardRounded className="i16 arrow" />
                  <span className="code">{d.codeTo}</span>
                  <span className="dirname">
                    {d.from} to {d.to}
                  </span>
                </button>
              ))}
            </div>
            <button className="btn" onClick={submit} disabled={!file || busy || !health?.ok}>
              {busy ? (
                "Working…"
              ) : (
                <>
                  <PlayArrowRounded className="i18" /> Start dubbing
                </>
              )}
            </button>
            {health && !health.xtts_license_accepted && (
              <div className="setup-box">
                <InfoOutlined className="i16" />
                <span>
                  Review the voice-model licence, then restart the backend with
                  <code>COQUI_TOS_AGREED=1</code>.
                </span>
              </div>
            )}
          </section>

          <section className="card">
            <h2>Pipeline</h2>
            <ol className="stages">
              {STAGES.map((s, i) => (
                <li
                  key={s}
                  className={`stage ${i < activeStage ? "done" : i === activeStage ? "active" : ""}`}
                >
                  <span className="num">
                    {i < activeStage ? <CheckRounded className="i14" /> : i + 1}
                  </span>
                  <span>{s}</span>
                </li>
              ))}
            </ol>
          </section>
        </aside>
      </div>
    </div>
  );
}
