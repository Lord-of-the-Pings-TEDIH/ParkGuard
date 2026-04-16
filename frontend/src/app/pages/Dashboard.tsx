import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import { AlertCircle } from "lucide-react";
import { UploadZone } from "../components/UploadZone";
import { ProcessingView } from "../components/ProcessingView";
import { ResultsView } from "../components/ResultsView";
import { SessionHistory } from "../components/SessionHistory";
import { PlateSearch } from "../components/PlateSearch";
import { ThemeToggle } from "../components/ThemeToggle";
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

  // Fetch all sessions on mount
  useEffect(() => {
    fetchSessions();
    fetchTestFiles();
  }, []);

  // Polling for active session updates
  useEffect(() => {
    if (!activeSession || viewState !== "processing") return;

    const pollInterval = setInterval(async () => {
      try {
        const [updatedSession, currentDetections] = await Promise.all([
          getSession(activeSession.id),
          getDetections(activeSession.id),
        ]);

        setActiveSession(updatedSession);
        setDetections(currentDetections);

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
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [activeSession, viewState]);

  const fetchSessions = async () => {
    try {
      const allSessions = await getSessions();
      setSessions(allSessions);
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
      setViewState("processing");
      await fetchSessions();
      void startSessionProcessing(session.id).catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to start processing");
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
    }
  };

  const handleRunHardcodedTest = async (filename: string) => {
    try {
      setError(null);
      const session = await createSessionFromHardcodedTest(filename);
      setActiveSession(session);
      setDetections([]);
      setViewState("processing");
      await fetchSessions();
      void startSessionProcessing(session.id).catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to start processing");
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create test session");
    }
  };

  const handleSelectSession = async (id: string) => {
    try {
      const [session, sessionDetections] = await Promise.all([
        getSession(id),
        getDetections(id),
      ]);

      setActiveSession(session);
      setDetections(sessionDetections);

      if (session.status === "running") {
        setViewState("processing");
      } else if (session.status === "done") {
        setViewState("results");
      } else if (session.status === "pending" || session.status === "failed") {
        setViewState("processing");
        void startSessionProcessing(session.id).catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to start processing");
        });
      }
    } catch (err) {
      console.error("Failed to load session:", err);
      setError(err instanceof Error ? err.message : "Failed to load session");
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
    setActiveSession(null);
    setDetections([]);
    setViewState("idle");
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
      <div className="flex flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Center content area */}
        <div className="flex-1 overflow-hidden order-1">
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

        {/* Right sidebars - Session history and Plate search */}
        <div className="order-2 flex flex-col lg:flex-row lg:w-auto">
          <div className="h-64 lg:h-auto lg:w-80 border-t lg:border-t-0">
            <SessionHistory
              sessions={sessions}
              testFiles={testFiles}
              activeSessionId={activeSession?.id || null}
              onSelectSession={handleSelectSession}
              onDeleteSession={handleDeleteSession}
              onRunHardcodedTest={handleRunHardcodedTest}
            />
          </div>

          <div className="h-64 lg:h-auto lg:w-80 border-t lg:border-t-0">
            <PlateSearch />
          </div>
        </div>
      </div>
    </div>
  );
}
