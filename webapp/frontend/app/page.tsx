"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";

// Icons are imported by direct path rather than from the package barrel: the barrel pulls in
// every icon and noticeably slows the dev server.
import ArrowForwardRounded from "@mui/icons-material/ArrowForwardRounded";
import CheckRounded from "@mui/icons-material/CheckRounded";
import CloseRounded from "@mui/icons-material/CloseRounded";
import CloudUploadOutlined from "@mui/icons-material/CloudUploadOutlined";
import DownloadRounded from "@mui/icons-material/DownloadRounded";
import ErrorOutlineRounded from "@mui/icons-material/ErrorOutlineRounded";
import GraphicEqRounded from "@mui/icons-material/GraphicEqRounded";
import InfoOutlined from "@mui/icons-material/InfoOutlined";
import InsertDriveFileOutlined from "@mui/icons-material/InsertDriveFileOutlined";
import LinkRounded from "@mui/icons-material/LinkRounded";
import MusicNoteRounded from "@mui/icons-material/MusicNoteRounded";
import PlayArrowRounded from "@mui/icons-material/PlayArrowRounded";
import ScheduleRounded from "@mui/icons-material/ScheduleRounded";
import SmartDisplayOutlined from "@mui/icons-material/SmartDisplayOutlined";
import SubtitlesOutlined from "@mui/icons-material/SubtitlesOutlined";

type Direction = "es-en" | "en-es";
type SourceMode = "file" | "link";

function isUrl(text: string): boolean {
  try {
    const u = new URL(text.trim());
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

type Word = { word: string; start: number; end: number };

/** Fallback for jobs made before words were aligned: spread the words over the span in
 *  proportion to their length, plus a little for the pause after punctuation. */
function estimateWords(text: string, start: number, end: number): Word[] {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  if (!tokens.length || end <= start) return [];
  const weights = tokens.map(
    (t) => t.replace(/[^\p{L}\p{N}]/gu, "").length + 1 + (/[.,!?;:]$/.test(t) ? 2 : 0),
  );
  const total = weights.reduce((a, b) => a + b, 0);
  let t = start;
  return tokens.map((word, i) => {
    const d = ((end - start) * weights[i]) / total;
    const w = { word, start: t, end: t + d };
    t += d;
    return w;
  });
}

/** Words that light up as they are spoken; clicking one seeks the video there. */
function Words({
  words,
  now,
  onSeek,
}: {
  words: Word[];
  now: number | null;
  onSeek: (t: number) => void;
}) {
  return (
    <>
      {words.map((w, k) => {
        const state = now === null ? "" : now >= w.end ? "past" : now >= w.start ? "now" : "";
        return (
          <Fragment key={k}>
            <span className={`w ${state}`} onClick={() => onSeek(w.start)}>
              {w.word}
            </span>{" "}
          </Fragment>
        );
      })}
    </>
  );
}

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
  /** Which detected speaker's voice this segment was cloned from, from 1. Absent on older jobs. */
  speaker?: number;
  /** Aligned timings of the dubbed words, on the output timeline. Absent on older jobs. */
  words?: Word[];
  source_words?: Word[];
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
  /** One entry per detected speaker. Absent on jobs from before per-speaker cloning. */
  speakers?: {
    speaker: number;
    segments: number;
    voiced?: number;
    reference_s: number | null;
    similarity: number | null;
  }[];
  /** Set when speaker detection failed and every segment used one shared voice. */
  speaker_detection_error?: string;
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

// This page hot-reloads but the backend does not, so after a pull the page can POST to a route the
// running backend does not have yet. None of the POST routes return 404 or 405 themselves, so
// either status on a POST means exactly that -- say so instead of showing "Method Not Allowed".
const STALE_BACKEND =
  "The backend is running older code than this page. Restart it with ./webapp/start_local.sh, then try again.";

// The backend refuses state-changing requests without this header, so other sites cannot drive it.
const CLIENT_HEADERS = { "X-BVT-Client": "web" };

function postFailure(status: number, fallback: string): string {
  return status === 404 || status === 405 ? STALE_BACKEND : fallback;
}

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<SourceMode>("file");
  const [link, setLink] = useState("");
  const [direction, setDirection] = useState<Direction>("es-en");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string>("");
  const [over, setOver] = useState(false);
  // -1 means "no upload in flight"; 0-100 is a live upload percentage.
  const [uploadPct, setUploadPct] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const segsRef = useRef<HTMLDivElement>(null);
  // Playback position of the result video; null until it has been played or seeked.
  const [now, setNow] = useState<number | null>(null);

  const [offline, setOffline] = useState(false);
  const [accepting, setAccepting] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setHealth(await res.json());
      setOffline(false);
    } catch {
      setOffline(true);
      setHealth({
        ok: false,
        ffmpeg: false,
        xtts_license_accepted: false,
        mt_mode: "unavailable",
        models_loaded: false,
        max_upload_mb: 200,
      });
    }
  }, []);

  // Keep checking until the backend is reachable and ready, so the page connects by itself
  // the moment the backend comes up -- no manual refresh.
  const healthOk = health?.ok ?? false;
  useEffect(() => {
    checkHealth();
    if (healthOk) return;
    const id = setInterval(checkHealth, 3000);
    return () => clearInterval(id);
  }, [checkHealth, healthOk]);

  async function acceptLicense() {
    setAccepting(true);
    try {
      const res = await fetch("/api/license/accept", { method: "POST", headers: CLIENT_HEADERS });
      if (res.ok) setHealth(await res.json());
      else setError(postFailure(res.status, `Could not enable the voice model (HTTP ${res.status})`));
    } catch {
      setError("Could not reach the backend. Is it running on port 8000?");
    } finally {
      setAccepting(false);
    }
  }

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

  async function submitLink() {
    setError("");
    setJob(null);
    const body = new FormData();
    body.append("url", link.trim());
    body.append("direction", direction);
    try {
      const res = await fetch("/api/jobs/url", { method: "POST", body, headers: CLIENT_HEADERS });
      const data: { job_id?: string; detail?: string } = await res.json().catch(() => ({}));
      if (res.ok && data.job_id) {
        setJob({ job_id: data.job_id, status: "queued", progress: 0, message: "Queued" });
        poll(data.job_id);
      } else {
        setError(postFailure(res.status, data.detail ?? `Request failed (HTTP ${res.status})`));
      }
    } catch {
      setError("Could not reach the backend. Is it running on port 8000?");
    }
  }

  async function submit() {
    if (mode === "link") return submitLink();
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
      xhr.setRequestHeader("X-BVT-Client", CLIENT_HEADERS["X-BVT-Client"]);
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
          setError(postFailure(xhr.status, data.detail ?? `Upload failed (HTTP ${xhr.status})`));
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
  // During the upload there is no job yet, but the first stage is effectively under way.
  const activeStage = uploading ? 0 : stageFromProgress(job?.progress ?? -1);
  const failed = job?.status === "error";
  const liveMessage = uploading ? `Uploading — ${uploadPct}%` : job?.message;
  function stageState(i: number): "done" | "active" | "failed" | "" {
    if (i < activeStage) return "done";
    if (i !== activeStage) return "";
    if (failed) return "failed";
    return busy ? "active" : "";
  }
  const pct = uploading ? uploadPct : (job?.progress ?? 0);
  const ready = mode === "file" ? !!file : isUrl(link);

  // Follow playback frame by frame while playing: `timeupdate` fires only ~4 times a second,
  // too coarse for word-level highlighting. Rounding lets React skip identical updates.
  const resultId = result?.job_id;
  useEffect(() => {
    setNow(null);
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const sync = () => setNow(Math.round(v.currentTime * 50) / 50);
    const tick = () => {
      sync();
      if (!v.paused && !v.ended) raf = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    v.addEventListener("play", onPlay);
    v.addEventListener("seeked", sync);
    v.addEventListener("pause", sync);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("seeked", sync);
      v.removeEventListener("pause", sync);
    };
  }, [resultId]);

  const segEnd = (s: Segment) => s.start + (s.final_duration || s.original_duration);
  const activeSeg =
    result && now !== null
      ? result.segments.findIndex((s) => now >= s.start && now < segEnd(s))
      : -1;

  // Keep the segment being spoken in view inside the scrolling list (not the whole page).
  useEffect(() => {
    const box = segsRef.current;
    const card = box?.children[activeSeg] as HTMLElement | undefined;
    if (!box || !card || videoRef.current?.paused) return;
    const top = card.offsetTop;
    if (top < box.scrollTop || top + card.offsetHeight > box.scrollTop + box.clientHeight) {
      box.scrollTo({ top: top - 8, behavior: "smooth" });
    }
  }, [activeSeg]);

  function seek(t: number) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    setNow(Math.round(t * 50) / 50);
    v.play().catch(() => {});
  }

  return (
    <div className="wrap">
      <header className="top">
        <div>
          <h1>Bilingual Voice Dubbing</h1>
          <p>Dub a clip into another language, in the original speaker&rsquo;s voice.</p>
        </div>
        <span className={`pill ${health?.ok ? "ok" : health === null ? "" : "bad"}`}>
          <span className="dot" />
          {health === null
            ? "Checking"
            : health.ok
              ? "Ready"
              : offline
                ? "Connecting…"
                : "Setup needed"}
        </span>
      </header>

      <div className="layout">
        {/* ---------------- main column ---------------- */}
        <main className="main">
          <section className="card">
            <div className="cardhead">
              <h2>
                <span className="step">1</span> Source
              </h2>
              <div className="tabs" role="tablist">
                {(["file", "link"] as const).map((m) => (
                  <button
                    key={m}
                    role="tab"
                    aria-selected={mode === m}
                    className={mode === m ? "sel" : ""}
                    onClick={() => {
                      setMode(m);
                      setError("");
                    }}
                    disabled={busy}
                  >
                    {m === "file" ? "Upload file" : "Paste link"}
                  </button>
                ))}
              </div>
            </div>
            {mode === "link" ? (
              <div className="linkbox">
                <label className="linkfield">
                  <LinkRounded className="i18" />
                  <input
                    type="url"
                    inputMode="url"
                    placeholder="https://www.youtube.com/watch?v=…"
                    value={link}
                    onChange={(e) => setLink(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && ready && !busy && health?.ok) submit();
                    }}
                    disabled={busy}
                  />
                </label>
                <div className="small">
                  YouTube, TikTok, Instagram, Facebook and most video sites &middot; up to 10
                  min. Private or login-only videos can&rsquo;t be fetched.
                </div>
              </div>
            ) : (
              <>
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
              </>
            )}
          </section>

          <section className="card">
            <div className="cardhead">
              <h2>Progress</h2>
              <span className="note">
                {result ? `done in ${Math.round(result.elapsed_s)}s` : busy ? `${pct}%` : "idle"}
              </span>
            </div>
            <div className={`bar ${busy ? "live" : ""}`}>
              <i style={{ width: `${pct}%` }} />
            </div>
            <div className="msg">
              <span>
                {uploading
                  ? `Uploading — ${uploadPct}%`
                  : (job?.message ??
                    (ready
                      ? "Ready — press Start dubbing."
                      : mode === "file"
                        ? "Choose a file, then start dubbing."
                        : "Paste a video link, then start dubbing."))}
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
                <video ref={videoRef} controls src={`/api/jobs/${result.job_id}/video`} />
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
                  {(result.speakers?.length ?? 0) > 1 && ` · ${result.speakers!.length} voices`}
                  {result.speaker_detection_error && " · speaker detection failed, one voice used"}
                  {result.clamped_segments > 0 && ` · ${result.clamped_segments} clamped`}
                  {result.failed_segments > 0 && ` · ${result.failed_segments} failed`}
                </span>
              )}
            </div>
            {result && (
              <p className="hint">
                Play the video and the words light up as they&rsquo;re spoken. Click a word to
                jump there.
              </p>
            )}
            {result ? (
              <div className="segs" ref={segsRef}>
                {result.segments.map((s, i) => {
                  const active = i === activeSeg;
                  const dubWords = s.words?.length
                    ? s.words
                    : estimateWords(s.translated_text, s.start, segEnd(s));
                  const srcWords = s.source_words?.length
                    ? s.source_words
                    : estimateWords(s.source_text, s.start, s.end);
                  return (
                    <article className={`seg ${active ? "active" : ""}`} key={s.index}>
                      <div className="seghead">
                        <button
                          className="time"
                          onClick={() => seek(s.start)}
                          title="Play from here"
                        >
                          {s.start.toFixed(2)}s – {s.end.toFixed(2)}s
                        </button>
                        <span className="tags">
                          {(result.speakers?.length ?? 0) > 1 && s.speaker && (
                            <span className="tag" title="Voice cloned from this speaker's own speech">
                              Speaker {s.speaker}
                            </span>
                          )}
                          {s.clamped && (
                            <span className="tag clamp" title="Hit the time-compression limit">
                              clamped
                            </span>
                          )}
                          {s.error && <span className="tag err">failed</span>}
                        </span>
                      </div>
                      <div className="segtext">
                        <p className="src">
                          {s.source_text ? (
                            <Words words={srcWords} now={active ? now : null} onSeek={seek} />
                          ) : (
                            <em>no speech recognised</em>
                          )}
                        </p>
                        <p className="tgt">
                          {s.translated_text ? (
                            <Words words={dubWords} now={active ? now : null} onSeek={seek} />
                          ) : (
                            <em>{s.error || "—"}</em>
                          )}
                        </p>
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
                  );
                })}
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
            <button className="btn" onClick={submit} disabled={!ready || busy || !health?.ok}>
              {busy ? (
                "Working…"
              ) : (
                <>
                  <PlayArrowRounded className="i18" /> Start dubbing
                </>
              )}
            </button>
            {offline && (
              <div className="setup-box">
                <InfoOutlined className="i16" />
                <span>
                  The backend isn&rsquo;t running. Start it with
                  <code>./webapp/start_local.sh</code> &mdash; this page connects automatically.
                </span>
              </div>
            )}
            {health && !offline && !health.xtts_license_accepted && (
              <div className="setup-box col">
                <div className="setup-row">
                  <InfoOutlined className="i16" />
                  <span>
                    The voice model (XTTS-v2) is free for non-commercial use under the{" "}
                    <a href="https://coqui.ai/cpml" target="_blank" rel="noreferrer">
                      Coqui Public Model License
                    </a>
                    . Accept it once to enable dubbing.
                  </span>
                </div>
                <button className="btn" onClick={acceptLicense} disabled={accepting}>
                  <CheckRounded className="i18" />
                  {accepting ? "Enabling…" : "Accept licence & enable"}
                </button>
              </div>
            )}
          </section>

          <section className="card">
            <h2>Pipeline</h2>
            <ol className="stages">
              {STAGES.map((s, i) => {
                const state = stageState(i);
                return (
                  <li
                    key={s}
                    className={`stage ${state}`}
                    aria-current={state === "active" ? "step" : undefined}
                  >
                    <span className="num">
                      {state === "done" ? (
                        <CheckRounded className="i14" />
                      ) : state === "failed" ? (
                        <CloseRounded className="i14" />
                      ) : state === "active" ? (
                        <span className="spin" aria-label="in progress" />
                      ) : (
                        i + 1
                      )}
                    </span>
                    <span className="stagebody">
                      <span>{s}</span>
                      {state === "active" && liveMessage && (
                        <span className="sub">{liveMessage}</span>
                      )}
                      {state === "failed" && <span className="sub">Stopped here</span>}
                    </span>
                  </li>
                );
              })}
            </ol>
          </section>
        </aside>
      </div>
    </div>
  );
}
