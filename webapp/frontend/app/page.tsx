"use client";

import { useCallback, useEffect, useRef, useState } from "react";

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
  elapsed_s: number;
};

type Job = {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  result?: DubResult;
};

const DIRECTIONS: { id: Direction; from: string; to: string; flagFrom: string; flagTo: string }[] = [
  { id: "es-en", from: "Spanish", to: "English", flagFrom: "🇪🇸", flagTo: "🇬🇧" },
  { id: "en-es", from: "English", to: "Spanish", flagFrom: "🇬🇧", flagTo: "🇪🇸" },
];

const STAGES = [
  "Extract audio from video",
  "Transcribe speech (Whisper)",
  "Translate text (MarianMT)",
  "Synthesise cloned voice (XTTS-v2)",
  "Retime to preserve prosody",
  "Merge audio back into video",
];

function stageFromProgress(p: number): number {
  if (p < 8) return 0;
  if (p < 20) return 1;
  if (p < 86) return 3;
  if (p < 92) return 4;
  if (p < 100) return 5;
  return 6;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function Page() {
  const [health, setHealth] = useState<{ ok: boolean; ffmpeg: boolean } | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [direction, setDirection] = useState<Direction>("es-en");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string>("");
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ ok: false, ffmpeg: false }));
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
    try {
      const res = await fetch("/api/jobs", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.detail ?? "Upload failed");
        return;
      }
      setJob({ job_id: data.job_id, status: "queued", progress: 0, message: "Queued" });
      poll(data.job_id);
    } catch {
      setError("Could not reach the backend. Is it running on port 8000?");
    }
  }

  function pick(f: File | undefined) {
    if (!f) return;
    setFile(f);
    setJob(null);
    setError("");
  }

  const busy = job?.status === "queued" || job?.status === "running";
  const result = job?.status === "done" ? job.result : undefined;
  const activeStage = stageFromProgress(job?.progress ?? -1);

  return (
    <div className="wrap">
      <header className="top">
        <div className="brand">
          <h1>Bilingual Voice Dubbing</h1>
          <p>
            Transcription and synthesis preserving speaker identity. Upload a video, pick a
            direction, and the pipeline dubs it in the original speaker&rsquo;s voice — retimed so the
            dub keeps the source&rsquo;s rhythm and stays in sync.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="pill">
            <span className={`dot ${health?.ok ? "on" : "off"}`} />
            {health === null ? "checking backend" : health.ok ? "backend ready" : "backend offline"}
          </span>
          <span className="pill">
            <span className={`dot ${health?.ffmpeg ? "on" : "off"}`} />
            ffmpeg
          </span>
        </div>
      </header>

      <div className="bento">
        {/* upload */}
        <section className="tile span-4">
          <h3>1 · Source media</h3>
          <p className="sub">
            Video or audio. MP4, MOV, MKV, WEBM, WAV, MP3, M4A, FLAC — up to 200 MB.
          </p>
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
            <div className="big">Drop a file here, or click to browse</div>
            <div className="small">The audio track is extracted automatically</div>
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
              <span className="name">{file.name}</span>
              <span className="sz">{fmtBytes(file.size)}</span>
            </div>
          )}
        </section>

        {/* direction */}
        <section className="tile span-2">
          <h3>2 · Direction</h3>
          <p className="sub">Which language to dub from and into.</p>
          <div className="dirs">
            {DIRECTIONS.map((d) => (
              <button
                key={d.id}
                className={`dirbtn ${direction === d.id ? "sel" : ""}`}
                onClick={() => setDirection(d.id)}
                disabled={busy}
              >
                <span className="flag">{d.flagFrom}</span>
                <span>{d.from}</span>
                <span className="arrow">→</span>
                <span className="flag">{d.flagTo}</span>
                <span>{d.to}</span>
              </button>
            ))}
          </div>
          <button className="btn" onClick={submit} disabled={!file || busy || !health?.ok}>
            {busy ? "Dubbing…" : "Start dubbing"}
          </button>
        </section>

        {/* pipeline */}
        <section className="tile span-2">
          <h3>Pipeline</h3>
          <p className="sub">Six stages, run on this machine.</p>
          <div className="stages">
            {STAGES.map((s, i) => (
              <div
                key={s}
                className={`stage ${i < activeStage ? "done" : i === activeStage ? "active" : ""}`}
              >
                <span className="num">{i < activeStage ? "✓" : i + 1}</span>
                <span>{s}</span>
              </div>
            ))}
          </div>
        </section>

        {/* progress */}
        <section className="tile span-4">
          <h3>Progress</h3>
          <div className="pcts">
            <b>{job ? `${job.progress}%` : "—"}</b>
            <span className="note">
              {result ? `finished in ${result.elapsed_s}s` : busy ? "working…" : "idle"}
            </span>
          </div>
          <div className="bar">
            <i style={{ width: `${job?.progress ?? 0}%` }} />
          </div>
          <div className="msg">{job?.message ?? "Upload a file and press Start dubbing."}</div>
          {error && <div className="err-box" style={{ marginTop: 12 }}>{error}</div>}
          {job?.status === "error" && (
            <div className="err-box" style={{ marginTop: 12 }}>{job.message}</div>
          )}
          {!error && !job && (
            <p className="note" style={{ marginTop: 12 }}>
              Expect roughly 13&times; the clip length: a 15-second clip takes about 3 minutes.
              Speech synthesis runs on the CPU and cannot be parallelised.
            </p>
          )}
        </section>

        {/* result */}
        <section className="tile span-4">
          <h3>3 · Dubbed result</h3>
          {result ? (
            <>
              <video controls src={`/api/jobs/${result.job_id}/video`} />
              <div className="vrow">
                <a href={`/api/jobs/${result.job_id}/video`} download>
                  <button className="btn ghost">Download video</button>
                </a>
                <a href={`/api/jobs/${result.job_id}/audio`} download>
                  <button className="btn ghost">Download audio only</button>
                </a>
              </div>
            </>
          ) : (
            <div className="empty">
              The dubbed video appears here once processing finishes.
              <br />
              Video is stream-copied, so only the audio track is replaced.
            </div>
          )}
        </section>

        {/* metrics */}
        <section className="tile span-2">
          <h3>Measurements</h3>
          <p className="sub">Computed on the result, not estimated.</p>
          {result ? (
            <div className="mgrid">
              <div className="metric">
                <div className="k">Voice match</div>
                <div className="v good">
                  {result.speaker_similarity !== null
                    ? result.speaker_similarity.toFixed(3)
                    : "n/a"}
                </div>
                <div className="n">ECAPA cosine</div>
              </div>
              <div className="metric">
                <div className="k">Timing match</div>
                <div
                  className={`v ${
                    result.duration_match_ratio !== null &&
                    Math.abs(result.duration_match_ratio - 1) < 0.1
                      ? "good"
                      : "warn"
                  }`}
                >
                  {result.duration_match_ratio !== null
                    ? `${result.duration_match_ratio.toFixed(3)}×`
                    : "n/a"}
                </div>
                <div className="n">after retiming</div>
              </div>
              <div className="metric">
                <div className="k">Before retiming</div>
                <div className="v warn">
                  {result.raw_duration_ratio !== null
                    ? `${result.raw_duration_ratio.toFixed(3)}×`
                    : "n/a"}
                </div>
                <div className="n">raw TTS length</div>
              </div>
              <div className="metric">
                <div className="k">Segments</div>
                <div className="v">{result.segments.length}</div>
                <div className="n">
                  {result.failed_segments} failed · {result.clamped_segments} clamped
                </div>
              </div>
            </div>
          ) : (
            <div className="empty">Voice similarity and timing accuracy appear after dubbing.</div>
          )}
        </section>

        {/* segments */}
        <section className="tile span-6">
          <h3>4 · Segment detail</h3>
          <p className="sub">
            Each detected utterance, its translation, and how much it was stretched or compressed to
            fit the original slot.
          </p>
          {result ? (
            <div className="segs">
              {result.segments.map((s) => (
                <div className="seg" key={s.index}>
                  <div className="hd">
                    <span>
                      {s.start.toFixed(2)}s – {s.end.toFixed(2)}s
                    </span>
                    <span style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      <span>
                        {s.original_duration.toFixed(2)}s slot · TTS{" "}
                        {s.raw_tts_duration.toFixed(2)}s → {s.final_duration.toFixed(2)}s (
                        {s.scale_applied.toFixed(2)}&times;)
                      </span>
                      {s.clamped && <span className="tag clamp">clamped</span>}
                      {s.error && <span className="tag err">failed</span>}
                    </span>
                  </div>
                  <div className="src">{s.source_text || <em>no speech recognised</em>}</div>
                  <div className="tgt">{s.translated_text || <em>{s.error || "—"}</em>}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">
              Segment-by-segment transcription, translation and retiming appear here.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
