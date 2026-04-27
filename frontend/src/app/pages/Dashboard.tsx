import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { AlertCircle, ArrowLeft } from "lucide-react";
import { UploadZone } from "../components/UploadZone";
import { ProcessingView } from "../components/ProcessingView";
import { ResultsView } from "../components/ResultsView";
import { SessionHistory } from "../components/SessionHistory";
import { PlateSearch } from "../components/PlateSearch";
import { SuspiciousOccupancyPanel } from "../components/SuspiciousOccupancyPanel";
import { ThemeToggle } from "../components/ThemeToggle";
import { Button } from "../components/ui/button";
import {
  createSession,
  createSessionFromHardcodedTest,
  getSession,
  getSessions,
  deleteSession,
  cancelSession,
  getDetections,
  getHardcodedTestFiles,
  startSessionProcessing,
  subscribeToSessionEvents,
} from "../services/api";
import type { Session, Detection, MobileLprPose } from "../types";

type ViewState = "idle" | "processing" | "results";

function viewStateForStatus(status: Session["status"]): ViewState {
  // "running" is the only state that benefits from the live poller; for any
  // other status (done / pending / failed) we just show the existing
  // detections, no auto-restart.
  return status === "running" ? "processing" : "results";
}

export function Dashboard() {
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
  // Mobile-LPR pose set in the sidebar.  Applies to whichever session
  // (upload or test file) the user launches next.
  const mobileLprPoseRef = useRef<MobileLprPose | null>(null);
  const handleMobileLprPoseChange = useCallback((pose: MobileLprPose | null) => {
    mobileLprPoseRef.current = pose;
  }, []);

  // Fetch all sessions on mount
  useEffect(() => {
    fetchSessions();
    fetchTestFiles();
  }, []);

  // SSE subscription for live session updates — replaces 1 s polling.
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
          const currentDetections = await getDetections(sessionId);
          setDetections(currentDetections);
          detectionsCacheRef.current.set(sessionId, currentDetections);
        } catch (err) {
          console.error("Failed to refresh detections:", err);
        }
      },

      onCompleted: async ({ frames_processed, total_frames }) => {
        try {
          const updatedSession = await getSession(sessionId);
          setActiveSession(updatedSession);
          sessionCacheRef.current.set(sessionId, updatedSession);
          const finalDetections = await getDetections(sessionId);
          setDetections(finalDetections);
          detectionsCacheRef.current.set(sessionId, finalDetections);
        } catch (err) {
          console.error("Failed to finalize session data:", err);
          // Fall back to what the event told us.
          setActiveSession((prev) =>
            prev ? { ...prev, frames_processed, frames_total: total_frames ?? prev.frames_total, status: "done" } : prev,
          );
        }
        setViewState("results");
        void fetchSessions();
      },

      onFailed: async () => {
        try {
          const updatedSession = await getSession(sessionId);
          setActiveSession(updatedSession);
          sessionCacheRef.current.set(sessionId, updatedSession);
        } catch (err) {
          console.error("Failed to refresh failed session:", err);
        }
        void fetchSessions();
      },
    });

    return close;
  }, [activeSession?.id, viewState]);

  const fetchSessions = async () => {
    try {
      const allSessions = await getSessions();
      setSessions(allSessions);
      allSessions.forEach((session) => {
        sessionCacheRef.current.set(session.id, session);
      });
    } catch (err) {
      console.error("Failed to fetch sessions:", err);
    }
  };

  const fetchTestFiles = async () => {
    try {
      const files = await getHardcodedTestFiles();
      setTestFiles(files);
    } catch (err) {
      console.error("Failed to fetch test files:", err);
    }
  };

  const handleUpload = async (file: File, fps: number) => {
    try {
      setError(null);
      const session = await createSession(file, fps, mobileLprPoseRef.current);
      setActiveSession(session);
      setDetections([]);
      sessionCacheRef.current.set(session.id, session);
      detectionsCacheRef.current.set(session.id, []);
      setViewState("processing");
      void fetchSessions();
      // Adopt the post-/process status (running) so the Cancel button shows
      // immediately.  Without this the activeSession stays at "pending" until
      // an SSE terminal event fires, which never happens for a long-running
      // session.
      startSessionProcessing(session.id)
        .then((updatedSession) => {
          setActiveSession((prev) =>
            prev?.id === updatedSession.id ? updatedSession : prev,
          );
          sessionCacheRef.current.set(updatedSession.id, updatedSession);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to start processing");
        });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
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
        .then((updatedSession) => {
          setActiveSession((prev) =>
            prev?.id === updatedSession.id ? updatedSession : prev,
          );
          sessionCacheRef.current.set(updatedSession.id, updatedSession);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to start processing");
        });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create test session");
    } finally {
      setPendingTestFile(null);
    }
  };

  const handleSelectSession = async (id: string) => {
    const selectionRequestId = ++selectionRequestIdRef.current;
    setPendingSessionId(id);
    setError(null);

    const cachedSession =
      sessionCacheRef.current.get(id) || sessions.find((session) => session.id === id) || null;
    if (cachedSession) {
      setActiveSession(cachedSession);
      setViewState(viewStateForStatus(cachedSession.status));
      sessionCacheRef.current.set(id, cachedSession);
    }

    const cachedDetections = detectionsCacheRef.current.get(id);
    if (cachedDetections) {
      setDetections(cachedDetections);
    } else {
      setDetections([]);
    }

    try {
      const [freshSession, sessionDetections] = await Promise.all([
        getSession(id),
        getDetections(id),
      ]);
      if (selectionRequestId !== selectionRequestIdRef.current) {
        return;
      }

      sessionCacheRef.current.set(id, freshSession);
      detectionsCacheRef.current.set(id, sessionDetections);

      setActiveSession(freshSession);
      setDetections(sessionDetections);
      // Selecting a session never restarts processing — it only displays
      // whatever is on disk.  "running" sessions go to ProcessingView so the
      // poller picks up live updates; everything else (done/pending/failed)
      // shows the existing detections in ResultsView.
      setViewState(viewStateForStatus(freshSession.status));
    } catch (err) {
      if (selectionRequestId !== selectionRequestIdRef.current) {
        return;
      }
      console.error("Failed to load session:", err);
      setError(err instanceof Error ? err.message : "Failed to load session");
    } finally {
      if (selectionRequestId === selectionRequestIdRef.current) {
        setPendingSessionId(null);
      }
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id);
      await fetchSessions();

      if (activeSession?.id === id) {
        handleReset();
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
      setError(err instanceof Error ? err.message : "Failed to delete session");
    }
  };

  const handleCancel = async () => {
    if (!activeSession) {
      return;
    }

    try {
      setError(null);
      await cancelSession(activeSession.id);
      await fetchSessions();
      handleReset();
    } catch (err) {
      console.error("Failed to cancel session:", err);
      setError(err instanceof Error ? err.message : "Failed to cancel session");
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

  return (
    <div className="flex h-screen flex-col bg-background">
      <ThemeToggle />

      {/* Error banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            exit={{ y: -100 }}
            className="border-b border-red-400 bg-gradient-to-r from-red-50 to-orange-50 px-6 py-3 dark:border-red-600 dark:from-red-950 dark:to-orange-950"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-red-700 dark:text-red-300">
                <AlertCircle className="h-4 w-4" />
                <span>{error}</span>
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

      {/* Main layout */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Center content area */}
        <div className="order-1 min-h-0 flex-1 overflow-hidden">
          <div className="flex h-full min-h-0 flex-col">
            {viewState !== "idle" && (
              <div className="border-b border-border bg-card/80 px-3 py-2 backdrop-blur md:px-4">
                <Button variant="outline" size="sm" onClick={handleReset}>
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to upload
                </Button>
              </div>
            )}

            <div className="min-h-0 flex-1 overflow-hidden">
              <AnimatePresence mode="wait">
                {viewState === "idle" && (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="h-full"
                  >
                    <UploadZone
                      onUpload={handleUpload}
                      isProcessing={false}
                    />
                  </motion.div>
                )}

                {viewState === "processing" && activeSession && (
                  <motion.div
                    key="processing"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="h-full"
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
                    className="h-full"
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
          </div>
        </div>

        {/* Right sidebars - Session history and Plate search */}
        <div className="order-2 flex shrink-0 flex-col border-t border-border md:flex-row lg:w-auto lg:border-t-0">
          <div className="h-72 min-h-0 border-b border-border md:h-auto md:w-80 md:border-b-0 md:border-r lg:border-t-0">
            <SessionHistory
              sessions={sessions}
              testFiles={testFiles}
              activeSessionId={activeSession?.id || null}
              pendingSessionId={pendingSessionId}
              pendingTestFile={pendingTestFile}
              onSelectSession={handleSelectSession}
              onDeleteSession={handleDeleteSession}
              onRunHardcodedTest={handleRunHardcodedTest}
              onMobileLprPoseChange={handleMobileLprPoseChange}
            />
          </div>

          <div className="h-72 min-h-0 border-b border-border md:h-auto md:w-80 md:border-b-0 md:border-r">
            <PlateSearch />
          </div>

          <div className="h-72 min-h-0 md:h-auto md:w-80">
            <SuspiciousOccupancyPanel />
          </div>
        </div>
      </div>
    </div>
  );
}
