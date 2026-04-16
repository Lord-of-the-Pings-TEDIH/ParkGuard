import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { AlertCircle, ArrowLeft } from "lucide-react";
import { UploadZone } from "../components/UploadZone";
import { ProcessingView } from "../components/ProcessingView";
import { ResultsView } from "../components/ResultsView";
import { SessionHistory } from "../components/SessionHistory";
import { PlateSearch } from "../components/PlateSearch";
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
} from "../services/api";
import type { Session, Detection } from "../types";

type ViewState = "idle" | "processing" | "results";

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

  // Fetch all sessions on mount
  useEffect(() => {
    fetchSessions();
    fetchTestFiles();
  }, []);

  // Polling for active session updates
  useEffect(() => {
    if (!activeSession || viewState !== "processing") return;
    const activeSessionId = activeSession.id;

    let lastFramesProcessed = activeSession.frames_processed;

    const pollInterval = setInterval(async () => {
      try {
        const updatedSession = await getSession(activeSessionId);

        setActiveSession(updatedSession);
        sessionCacheRef.current.set(updatedSession.id, updatedSession);
        const shouldRefreshDetections =
          updatedSession.frames_processed !== lastFramesProcessed ||
          updatedSession.status === "done" ||
          updatedSession.status === "failed";
        if (shouldRefreshDetections) {
          const currentDetections = await getDetections(activeSessionId);
          setDetections(currentDetections);
          detectionsCacheRef.current.set(activeSessionId, currentDetections);
          lastFramesProcessed = updatedSession.frames_processed;
        }

        // Stop polling when terminal state reached
        if (updatedSession.status === "done" || updatedSession.status === "failed") {
          clearInterval(pollInterval);
          if (updatedSession.status === "done") {
            setViewState("results");
          }
          // Refresh session list to update status
          fetchSessions();
        }
      } catch (err) {
        console.error("Polling error:", err);
        setError(err instanceof Error ? err.message : "Polling failed");
      }
    }, 1000);

    return () => clearInterval(pollInterval);
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
      const session = await createSession(file, fps);
      setActiveSession(session);
      setDetections([]);
      sessionCacheRef.current.set(session.id, session);
      detectionsCacheRef.current.set(session.id, []);
      setViewState("processing");
      void fetchSessions();
      void startSessionProcessing(session.id).catch((err) => {
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
      const session = await createSessionFromHardcodedTest(filename);
      setActiveSession(session);
      setDetections([]);
      sessionCacheRef.current.set(session.id, session);
      detectionsCacheRef.current.set(session.id, []);
      setViewState("processing");
      void fetchSessions();
      void startSessionProcessing(session.id).catch((err) => {
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
      setViewState(cachedSession.status === "done" ? "results" : "processing");
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
      setViewState(freshSession.status === "done" ? "results" : "processing");

      if (freshSession.status === "pending" || freshSession.status === "failed") {
        void startSessionProcessing(freshSession.id)
          .then((startedSession) => {
            if (selectionRequestId !== selectionRequestIdRef.current) {
              return;
            }
            setActiveSession(startedSession);
            sessionCacheRef.current.set(startedSession.id, startedSession);
            setViewState("processing");
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : "Failed to start processing");
          });
      }
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
            />
          </div>

          <div className="h-72 min-h-0 md:h-auto md:w-80">
            <PlateSearch />
          </div>
        </div>
      </div>
    </div>
  );
}
