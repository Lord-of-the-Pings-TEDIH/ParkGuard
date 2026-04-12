import type { Detection, Session, Plate } from "../types";
import {
  MOCK_SESSIONS,
  MOCK_DETECTIONS,
  MOCK_PLATES,
  generateMockSession,
  simulateProcessing,
  generateMockDetection,
} from "./mockData";

const API_BASE = "/api";

// Development mode - use mock data if API is not available
let useMockData = false;
let mockSessionsStore: Session[] = [...MOCK_SESSIONS];
let mockDetectionsStore: Detection[] = [...MOCK_DETECTIONS];
let processingIntervals: Map<string, NodeJS.Timeout> = new Map();

async function fetchWithFallback<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  if (useMockData) {
    throw new Error("Using mock data");
  }

  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  } catch (error) {
    // Enable mock data mode on first failure
    useMockData = true;
    console.warn("API unavailable, switching to mock data mode");
    throw error;
  }
}

export async function createSession(file: File, fpsTarget: number): Promise<Session> {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("fps_target", String(fpsTarget));

    return await fetchWithFallback(`${API_BASE}/sessions`, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    // Mock mode
    const session = generateMockSession(file.name, fpsTarget);
    mockSessionsStore.unshift(session);
    startMockProcessing(session.id);
    return session;
  }
}

export async function createTestSession(): Promise<Session> {
  try {
    return await fetchWithFallback(`${API_BASE}/sessions/test`, {
      method: "POST",
    });
  } catch (error) {
    // Mock mode
    const session = generateMockSession("test_video.mp4", 5);
    mockSessionsStore.unshift(session);
    startMockProcessing(session.id);
    return session;
  }
}

export async function getSession(id: string): Promise<Session> {
  try {
    return await fetchWithFallback(`${API_BASE}/sessions/${id}`);
  } catch (error) {
    // Mock mode
    const session = mockSessionsStore.find(s => s.id === id);
    if (!session) {
      throw new Error("Session not found");
    }
    return session;
  }
}

export async function getSessions(): Promise<Session[]> {
  try {
    return await fetchWithFallback(`${API_BASE}/sessions`);
  } catch (error) {
    // Mock mode
    return mockSessionsStore;
  }
}

export async function deleteSession(id: string): Promise<void> {
  try {
    await fetchWithFallback(`${API_BASE}/sessions/${id}`, {
      method: "DELETE",
    });
  } catch (error) {
    // Mock mode
    mockSessionsStore = mockSessionsStore.filter(s => s.id !== id);
    mockDetectionsStore = mockDetectionsStore.filter(d => d.id !== id);
    const interval = processingIntervals.get(id);
    if (interval) {
      clearInterval(interval);
      processingIntervals.delete(id);
    }
  }
}

export async function getDetections(sessionId: string): Promise<Detection[]> {
  try {
    return await fetchWithFallback(`${API_BASE}/sessions/${sessionId}/detections`);
  } catch (error) {
    // Mock mode
    return mockDetectionsStore.filter(d => {
      // For existing mock sessions, return mock detections
      // For new sessions, return generated detections
      return sessionId === "1" || sessionId === "2" ? d.id <= "6" : true;
    });
  }
}

export async function searchPlates(query: string, county?: string): Promise<Plate[]> {
  try {
    const params = new URLSearchParams();
    if (query) params.append("q", query);
    if (county) params.append("county", county);

    return await fetchWithFallback(`${API_BASE}/plates?${params.toString()}`);
  } catch (error) {
    // Mock mode
    let results = MOCK_PLATES;

    if (query) {
      results = results.filter(p =>
        p.normalized_text.toLowerCase().includes(query.toLowerCase())
      );
    }

    if (county) {
      results = results.filter(p =>
        p.county_code.toLowerCase() === county.toLowerCase()
      );
    }

    return results;
  }
}

// Mock processing simulation
function startMockProcessing(sessionId: string) {
  let detectionCount = 0;

  const interval = setInterval(() => {
    const sessionIndex = mockSessionsStore.findIndex(s => s.id === sessionId);
    if (sessionIndex === -1) {
      clearInterval(interval);
      processingIntervals.delete(sessionId);
      return;
    }

    const session = mockSessionsStore[sessionIndex];
    const updatedSession = simulateProcessing(session);
    mockSessionsStore[sessionIndex] = updatedSession;

    // Generate new detections randomly
    if (Math.random() > 0.3 && detectionCount < 15) {
      const newDetection = generateMockDetection(sessionId, detectionCount);
      mockDetectionsStore.unshift(newDetection);
      detectionCount++;
    }

    if (updatedSession.status === "done") {
      clearInterval(interval);
      processingIntervals.delete(sessionId);
    }
  }, 1500);

  processingIntervals.set(sessionId, interval);
}
