import { useState, useCallback, useRef } from "react";
import { Upload } from "lucide-react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";

interface UploadZoneProps {
  onUpload: (file: File, fps: number) => void;
  isProcessing: boolean;
}

export function UploadZone({ onUpload, isProcessing }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fps, setFps] = useState([5]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && isValidVideoFile(file)) {
      setSelectedFile(file);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  }, []);

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile, fps[0]);
    }
  };

  const openFilePicker = () => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="flex h-full flex-col items-center justify-center gap-6 p-4 md:gap-8 md:p-12">
      <div className="text-center">
        <h1 className="mb-3 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-4xl font-bold tracking-tight text-transparent md:text-6xl">
          RO-PLATE
        </h1>
        <p className="text-sm text-muted-foreground md:text-base">
          License Plate Detection & Enforcement
        </p>
      </div>

      <div
        className={`relative w-full max-w-2xl cursor-pointer rounded-lg border-2 border-dashed transition-all focus-within:ring-2 focus-within:ring-blue-500 ${
          isDragging
            ? "border-blue-500 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-blue-950 dark:via-indigo-950 dark:to-purple-950"
            : "border-border bg-card hover:border-blue-400 hover:bg-blue-50/40 dark:hover:border-blue-700 dark:hover:bg-blue-950/30"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={openFilePicker}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openFilePicker();
          }
        }}
        role="button"
        tabIndex={isProcessing ? -1 : 0}
        aria-label="Select video file for processing"
      >
        <div className="p-8 text-center md:p-12">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 md:h-16 md:w-16">
            <Upload className="h-6 w-6 text-white md:h-8 md:w-8" />
          </div>
          <p className="mb-2 text-base font-medium text-foreground md:text-lg">
            {selectedFile ? selectedFile.name : "Drop video here"}
          </p>
          <p className="mb-4 text-xs text-muted-foreground md:text-sm">
            .mp4, .avi, .mkv · max 2GB
          </p>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".mp4,.avi,.mkv,video/mp4,video/x-msvideo,video/x-matroska"
            onChange={handleFileSelect}
            disabled={isProcessing}
          />
          <Button
            variant="outline"
            onClick={(e) => {
              e.stopPropagation();
              openFilePicker();
            }}
            disabled={isProcessing}
          >
            Browse Files
          </Button>
        </div>
      </div>

      {selectedFile && (
        <div className="w-full max-w-2xl space-y-4 rounded-lg border border-border bg-card p-6 shadow-lg">
          <div>
            <div className="mb-3">
              <Label className="text-foreground">Sampling: {fps[0]} FPS</Label>
            </div>
            <Slider
              value={fps}
              onValueChange={setFps}
              min={1}
              max={10}
              step={1}
              className="w-full"
            />
          </div>

          <Button
            onClick={handleSubmit}
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
          >
            Process Video
          </Button>
        </div>
      )}
    </div>
  );
}

function isValidVideoFile(file: File): boolean {
  const validTypes = ["video/mp4", "video/x-msvideo", "video/x-matroska"];
  const validExtensions = [".mp4", ".avi", ".mkv"];
  const maxSize = 2 * 1024 * 1024 * 1024; // 2GB

  const hasValidType = validTypes.includes(file.type);
  const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  const hasValidSize = file.size <= maxSize;

  return (hasValidType || hasValidExtension) && hasValidSize;
}
