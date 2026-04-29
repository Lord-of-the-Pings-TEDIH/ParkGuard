import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  AlertCircle,
  AlertTriangle,
  Car,
  FileText,
  Moon,
  Search,
  Settings,
  Sun,
  Upload,
} from "lucide-react";
import { UploadZone } from "../components/UploadZone";
import { ProcessingView } from "../components/ProcessingView";
import { ResultsView } from "../components/ResultsView";
import { SessionHistory } from "../components/SessionHistory";
import { PlateSearch } from "../components/PlateSearch";
import { SuspiciousOccupancyPanel } from "../components/SuspiciousOccupancyPanel";
import {
  cancelSession,
  createSession,
  createSessionFromHardcodedTest,
  deleteSession,
  getAlerts,
  getDetections,
  getHardcodedTestFiles,
  getSession,
  getSessions,
  startSessionProcessing,
  subscribeToSessionEvents,
} from "../services/api";
import type { Detection, MobileLprPose, Session } from "../types";

type ViewState = "idle" | "processing" | "results";
type RailTab = "sessions" | "plates" | "alerts";

const RAIL_TABS: Array<{ id: RailTab; Icon: React.ElementType; label: string }> = [
  { id: "sessions", Icon: FileText,      label: "Sesiuni"  },
  { id: "plates",   Icon: Search,        label: "Registru" },
  { id: "alerts",   Icon: AlertTriangle, label: "Alerte"   },
];

const PANEL_TITLE: Record<RailTab, string> = {
  sessions: "Sesiuni",
  plates:   "Registru plăci",
  alerts:   "Alerte & Abuzuri",
};

function viewStateForStatus(status: Session["status"]): ViewState {
  return status === "running" ? "processing" : "results";
}

export function Dashboard() {
  // ── Theme ─────────────────────────────────────────────────────────────────
  const [theme, setThemeState] = useState<"light" | "dark">("dark");
  useEffect(() => {
    const saved = localStorage.getItem("theme") as "light" | "dark" | null;
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initial = saved ?? (prefersDark ? "dark" : "light");
    setThemeState(initial);
    document.documentElement.classList.toggle("dark", initial === "dark");
  }, []);
  const toggleTheme = () => {
    const next = theme === "light" ? "dark" : "light";
    setThemeState(next);
    localStorage.setItem("theme", next);
    document.documentElement.classList.toggle("dark", next === "dark");
  };

  // ── Layout ────────────────────────────────────────────────────────────────
  const [railTab, setRailTab] = useState<RailTab>("sessions");
  const [contextOpen, setContextOpen] = useState(true);

  // ── Alert count for status bar + rail badge ────────────────────────────────
  const [alertCount, setAlertCount] = useState(0);
  useEffect(() => {
    const fetch = async () => {
      try {
        const list = await getAlerts(false);
        setAlertCount(list.length);
      } catch {}
    };
    void fetch();
    const t = setInterval(() => void fetch(), 60_000);
    return () => clearInterval(t);
  }, []);

  // ── Session / detection state ─────────────────────────────────────────────
  const [viewState, setViewState] = useState<ViewState>("idle");
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [testFiles, setTestFiles] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null);
  const [pendingTestFile, setPendingTestFile] = useState<string | null>(null);
  const selectionRequestIdRef = useRef(0);
  const sessionCacheRef = useRef<Map<string, Session>>(new Map());
  const detectionsCacheRef = useRef<Map<string, Detection[]>>(new Map());
  const mobileLprPoseRef = useRef<MobileLprPose | null>(null);

  const handleMobileLprPoseChange = useCallback((pose: MobileLprPose | null) => {
    mobileLprPoseRef.current = pose;
  }, []);

  useEffect(() => {
    void fetchSessions();
    void fetchTestFiles();
  }, []);

  useEffect(() => {
    if (!activeSession || viewState !== "processing") return;
    const sessionId = activeSession.id;
    const close = subscribeToSessionEvents(sessionId, {
      onFrameProcessed: ({ frames_processed, total_frames }) => {
        setActiveSession((prev) =>
          prev
            ? { ...prev, frames_processed, frames_total: total_frames ?? prev.frames_total }
            : prev,
        );
        sessionCacheRef.current.set(sessionId, {
          ...(sessionCacheRef.current.get(sessionId) ?? activeSession),
          frames_processed,
          frames_total: total_frames ?? activeSession.frames_total,
        });
      },
      onDetectionFinalized: async () => {
        try {
          const current = await getDetections(sessionId);
          setDetections(current);
          detectionsCacheRef.current.set(sessionId, current);
        } catch (err) {
          console.error("Eroare la reîmprospătarea detectărilor:", err);
        }
      },
      onCompleted: async ({ frames_processed, total_frames }) => {
        try {
          const updated = await getSession(sessionId);
          setActiveSession(updated);
          sessionCacheRef.current.set(sessionId, updated);
          const final = await getDetections(sessionId);
          setDetections(final);
          detectionsCacheRef.current.set(sessionId, final);
        } catch (err) {
          console.error("Eroare la finalizarea datelor sesiunii:", err);
          setActiveSession((prev) =>
            prev
              ? { ...prev, frames_processed, frames_total: total_frames ?? prev.frames_total, status: "done" }
              : prev,
          );
        }
        setViewState("results");
        void fetchSessions();
      },
      onFailed: async () => {
        try {
          const updated = await getSession(sessionId);
          setActiveSession(updated);
          sessionCacheRef.current.set(sessionId, updated);
        } catch (err) {
          console.error("Eroare la reîmprospătarea sesiunii eșuate:", err);
        }
        void fetchSessions();
      },
    });
    return close;
  }, [activeSession?.id, viewState]);

  const fetchSessions = async () => {
    try {
      const all = await getSessions();
      setSessions(all);
      all.forEach((s) => sessionCacheRef.current.set(s.id, s));
    } catch (err) {
      console.error("Eroare la încărcarea sesiunilor:", err);
    }
  };

  const fetchTestFiles = async () => {
    try {
      const files = await getHardcodedTestFiles();
      setTestFiles(files);
    } catch (err) {
      console.error("Eroare la încărcarea fișierelor de test:", err);
    }
  };

  const handleUpload = async (
    file: File,
    fps: number,
    _pose: MobileLprPose | null,
    zoneId: number | null,
  ) => {
    try {
      setError(null);
      const session = await createSession(file, fps, mobileLprPoseRef.current, zoneId);
      setActiveSession(session);
      setDetections([]);
      sessionCacheRef.current.set(session.id, session);
      detectionsCacheRef.current.set(session.id, []);
      setViewState("processing");
      void fetchSessions();
      startSessionProcessing(session.id)
        .then((updated) => {
          setActiveSession((prev) => (prev?.id === updated.id ? updated : prev));
          sessionCacheRef.current.set(updated.id, updated);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Pornirea procesării a eșuat");
        });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Crearea sesiunii a eșuat");
    }
  };

  const handleRunHardcodedTest = async (filename: string) => {
    setPendingTestFile(filename);
    try {
      setError(null);
      const session = await createSessionFromHardcodedTest(filename, mobileLprPoseRef.current);
      setActiveSession(session);
      setDetections([]);
      sessionCacheRef.current.set(session.id, session);
      detectionsCacheRef.current.set(session.id, []);
      setViewState("processing");
      void fetchSessions();
      startSessionProcessing(session.id)
        .then((updated) => {
          setActiveSession((prev) => (prev?.id === updated.id ? updated : prev));
          sessionCacheRef.current.set(updated.id, updated);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Pornirea procesării a eșuat");
        });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Crearea sesiunii de test a eșuat");
    } finally {
      setPendingTestFile(null);
    }
  };

  const handleSelectSession = async (id: string) => {
    const reqId = ++selectionRequestIdRef.current;
    setPendingSessionId(id);
    setError(null);

    const cached =
      sessionCacheRef.current.get(id) || sessions.find((s) => s.id === id) || null;
    if (cached) {
      setActiveSession(cached);
      setViewState(viewStateForStatus(cached.status));
    }
    const cachedDets = detectionsCacheRef.current.get(id);
    if (cachedDets) setDetections(cachedDets);
    else setDetections([]);

    try {
      const [fresh, dets] = await Promise.all([getSession(id), getDetections(id)]);
      if (reqId !== selectionRequestIdRef.current) return;
      sessionCacheRef.current.set(id, fresh);
      detectionsCacheRef.current.set(id, dets);
      setActiveSession(fresh);
      setDetections(dets);
      setViewState(viewStateForStatus(fresh.status));
    } catch (err) {
      if (reqId !== selectionRequestIdRef.current) return;
      console.error("Eroare la încărcarea sesiunii:", err);
      setError(err instanceof Error ? err.message : "Încărcarea sesiunii a eșuat");
    } finally {
      if (reqId === selectionRequestIdRef.current) setPendingSessionId(null);
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id);
      await fetchSessions();
      if (activeSession?.id === id) handleReset();
    } catch (err) {
      console.error("Eroare la ștergerea sesiunii:", err);
      setError(err instanceof Error ? err.message : "Ștergerea sesiunii a eșuat");
    }
  };

  const handleCancel = async () => {
    if (!activeSession) return;
    try {
      setError(null);
      await cancelSession(activeSession.id);
      await fetchSessions();
      handleReset();
    } catch (err) {
      console.error("Eroare la anularea sesiunii:", err);
      setError(err instanceof Error ? err.message : "Anularea sesiunii a eșuat");
    }
  };

  const handleReset = () => {
    selectionRequestIdRef.current += 1;
    setActiveSession(null);
    setDetections([]);
    setViewState("idle");
    setPendingSessionId(null);
    setPendingTestFile(null);
    setError(null);
  };

  const handleComplete = useCallback(() => {
    setViewState("results");
  }, []);

  // ── Meridian layout ───────────────────────────────────────────────────────
  return (
    <div
      className="flex flex-col overflow-hidden"
      style={{
        height: "calc(100vh - 26px)",
        fontFamily: "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
      }}
    >
      {/* Error banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ y: -60 }}
            animate={{ y: 0 }}
            exit={{ y: -60 }}
            className="border-b border-red-400 bg-gradient-to-r from-red-50 to-orange-50 px-6 py-3 dark:border-red-600 dark:from-red-950 dark:to-orange-950"
            style={{ flexShrink: 0 }}
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-red-700 dark:text-red-300">
                <AlertCircle className="h-4 w-4" />
                <span className="text-sm">{error}</span>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-700 hover:text-red-800 dark:text-red-300 dark:hover:text-red-200"
              >
                ✕
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── ICON RAIL (always dark) ──────────────────────────────────────── */}
        <nav
          style={{
            width: 56,
            background: "#0d1117",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            padding: "14px 0",
            gap: 4,
            flexShrink: 0,
            zIndex: 10,
          }}
        >
          {/* Logo */}
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 9,
              background: "#2563eb",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 10,
              flexShrink: 0,
            }}
          >
            <Car size={17} color="#fff" />
          </div>

          {RAIL_TABS.map((tab) => {
            const active = railTab === tab.id;
            const badge = tab.id === "alerts" ? alertCount : 0;
            return (
              <button
                key={tab.id}
                title={tab.label}
                onClick={() => {
                  setRailTab(tab.id);
                  setContextOpen(true);
                }}
                style={{
                  position: "relative",
                  width: 38,
                  height: 38,
                  borderRadius: 9,
                  background: active ? "rgba(255,255,255,0.14)" : "transparent",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer",
                  border: "none",
                  flexShrink: 0,
                  transition: "background 0.12s",
                }}
              >
                <tab.Icon
                  size={17}
                  color={active ? "#fff" : "rgba(255,255,255,0.4)"}
                />
                {badge > 0 && (
                  <div
                    style={{
                      position: "absolute",
                      top: 4,
                      right: 4,
                      width: 14,
                      height: 14,
                      background: "#f85149",
                      borderRadius: "50%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 9,
                      color: "#fff",
                      fontWeight: 700,
                    }}
                  >
                    {badge > 9 ? "9+" : badge}
                  </div>
                )}
              </button>
            );
          })}

          <div style={{ flex: 1 }} />

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            title={theme === "dark" ? "Mod luminos" : "Mod întunecat"}
            style={{
              width: 38,
              height: 38,
              borderRadius: 9,
              border: "none",
              background: "transparent",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            {theme === "dark" ? (
              <Sun size={16} color="rgba(255,255,255,0.4)" />
            ) : (
              <Moon size={16} color="rgba(255,255,255,0.4)" />
            )}
          </button>

          {/* Settings placeholder */}
          <button
            title="Setări"
            style={{
              width: 38,
              height: 38,
              borderRadius: 9,
              border: "none",
              background: "transparent",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <Settings size={16} color="rgba(255,255,255,0.3)" />
          </button>
        </nav>

        {/* ── CONTEXT PANEL + collapse handle ─────────────────────────────── */}
        <div style={{ position: "relative", display: "flex", flexShrink: 0 }}>

          {contextOpen && (
            <aside
              className="bg-card border-r border-border"
              style={{
                width: 284,
                display: "flex",
                flexDirection: "column",
                flexShrink: 0,
                overflow: "hidden",
              }}
            >
              {/* Panel header */}
              <div
                className="border-b border-border flex-shrink-0"
                style={{ padding: "16px 14px 10px" }}
              >
                <div
                  className="font-bold text-foreground"
                  style={{ fontSize: 14, marginBottom: 1 }}
                >
                  {PANEL_TITLE[railTab]}
                </div>
                {railTab === "sessions" && (
                  <div
                    className="text-muted-foreground"
                    style={{ fontSize: 11 }}
                  >
                    {sessions.length} procesări
                  </div>
                )}
              </div>

              {/* Panel content */}
              <div className="flex-1 overflow-hidden min-h-0">
                {railTab === "sessions" && (
                  <SessionHistory
                    sessions={sessions}
                    testFiles={testFiles}
                    activeSessionId={activeSession?.id ?? null}
                    pendingSessionId={pendingSessionId}
                    pendingTestFile={pendingTestFile}
                    onSelectSession={handleSelectSession}
                    onDeleteSession={handleDeleteSession}
                    onRunHardcodedTest={handleRunHardcodedTest}
                    onMobileLprPoseChange={handleMobileLprPoseChange}
                  />
                )}
                {railTab === "plates" && <PlateSearch />}
                {railTab === "alerts" && (
                  <SuspiciousOccupancyPanel onAlertCountChange={setAlertCount} />
                )}
              </div>
            </aside>
          )}

          {/* Collapse / expand handle */}
          <button
            onClick={() => setContextOpen((o) => !o)}
            className="bg-card border-border text-muted-foreground hover:text-foreground"
            style={{
              position: "absolute",
              left: contextOpen ? 284 : 0,
              top: 16,
              zIndex: 20,
              width: 18,
              height: 28,
              border: "1px solid",
              borderLeft: contextOpen ? "none" : undefined,
              borderRight: !contextOpen ? "none" : undefined,
              borderRadius: contextOpen ? "0 5px 5px 0" : "5px 0 0 5px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              fontSize: 10,
            }}
          >
            {contextOpen ? "‹" : "›"}
          </button>
        </div>

        {/* ── MAIN AREA ────────────────────────────────────────────────────── */}
        <main
          className="bg-background"
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            minWidth: 0,
          }}
        >
          {/* Top bar (processing / results only) */}
          {viewState !== "idle" && (
            <div
              className="bg-card border-b border-border"
              style={{
                height: 50,
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "0 24px",
                flexShrink: 0,
              }}
            >
              <button
                onClick={handleReset}
                className="border border-border text-muted-foreground hover:text-foreground"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "5px 10px",
                  borderRadius: 7,
                  fontSize: 12,
                  cursor: "pointer",
                  fontFamily: "inherit",
                  background: "transparent",
                }}
              >
                <Upload size={13} />
                Nouă sesiune
              </button>
              <div
                className="bg-border"
                style={{ width: 1, height: 20, flexShrink: 0 }}
              />
              <span
                className="text-muted-foreground"
                style={{
                  fontSize: 13,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {activeSession?.source_filename}
              </span>
            </div>
          )}

          {/* View content */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            <AnimatePresence mode="wait">
              {viewState === "idle" && (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  style={{ height: "100%" }}
                >
                  <UploadZone onUpload={handleUpload} isProcessing={false} />
                </motion.div>
              )}
              {viewState === "processing" && activeSession && (
                <motion.div
                  key="processing"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  style={{ height: "100%" }}
                >
                  <ProcessingView
                    session={activeSession}
                    detections={detections}
                    onCancel={handleCancel}
                    onComplete={handleComplete}
                  />
                </motion.div>
              )}
              {viewState === "results" && activeSession && (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  style={{ height: "100%" }}
                >
                  <ResultsView
                    session={activeSession}
                    detections={detections}
                    onReset={handleReset}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>
      </div>

      {/* ── STATUS BAR (always dark, fixed bottom) ───────────────────────── */}
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          height: 26,
          background: "#0d1117",
          display: "flex",
          alignItems: "center",
          padding: "0 16px",
          gap: 18,
          zIndex: 50,
        }}
      >
        <StatusDot color="#4ade80" label="API online" />
        {viewState === "processing" && (
          <StatusDot color="#facc15" label="1 sesiune activă" />
        )}
        {alertCount > 0 && (
          <StatusDot color="#f87171" label={`${alertCount} alerte deschise`} />
        )}
        <div style={{ flex: 1 }} />
        <span
          style={{
            fontSize: 11,
            color: "rgba(255,255,255,0.25)",
            fontFamily: "'IBM Plex Mono', monospace",
          }}
        >
          ParkGuard v2.4.1
        </span>
      </div>
    </div>
  );
}

function StatusDot({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <div
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: color,
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 11, color: "rgba(255,255,255,0.45)" }}>{label}</span>
    </div>
  );
}
