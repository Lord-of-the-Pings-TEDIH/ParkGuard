import { useState, useCallback, useEffect, useRef } from "react";
import { Upload } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";
import { Switch } from "./ui/switch";
import type { MobileLprPose, ParkingZone } from "../types";
import { getParkingZones } from "../services/api";

interface UploadZoneProps {
  onUpload: (file: File, fps: number, pose: MobileLprPose | null, zoneId: number | null) => void;
  isProcessing: boolean;
}

export function UploadZone({ onUpload, isProcessing }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fps, setFps] = useState([5]);
  const [mobileLprEnabled, setMobileLprEnabled] = useState(false);
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [headingDeg, setHeadingDeg] = useState("");
  const [poseError, setPoseError] = useState<string | null>(null);
  const [zones, setZones] = useState<ParkingZone[]>([]);
  const [selectedZoneId, setSelectedZoneId] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    getParkingZones()
      .then((z) => {
        setZones(z);
        if (z.length > 0 && selectedZoneId === null) {
          setSelectedZoneId(z[0].id);
        }
      })
      .catch(() => {
        // Zone selector is optional — silently degrade if the API is unavailable.
      });
  }, []);

  const parsePose = (): MobileLprPose | null | "invalid" => {
    if (!mobileLprEnabled) return null;
    const lat = Number.parseFloat(latitude);
    const lon = Number.parseFloat(longitude);
    const hdg = Number.parseFloat(headingDeg);
    if (
      Number.isNaN(lat) || lat < -90 || lat > 90 ||
      Number.isNaN(lon) || lon < -180 || lon > 180 ||
      Number.isNaN(hdg)
    ) {
      return "invalid";
    }
    return { latitude: lat, longitude: lon, headingDeg: ((hdg % 360) + 360) % 360 };
  };

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
    if (!selectedFile) return;
    const pose = parsePose();
    if (pose === "invalid") {
      setPoseError("Enter valid latitude (-90..90), longitude (-180..180), and heading (degrees).");
      return;
    }
    setPoseError(null);
    onUpload(selectedFile, fps[0], pose, selectedZoneId);
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
          Detectare și Monitorizare Numere Auto
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
        aria-label="Selectează clip video pentru procesare"
      >
        <div className="p-8 text-center md:p-12">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 md:h-16 md:w-16">
            <Upload className="h-6 w-6 text-white md:h-8 md:w-8" />
          </div>
          <p className="mb-2 text-base font-medium text-foreground md:text-lg">
            {selectedFile ? selectedFile.name : "Trageți clipul video aici"}
          </p>
          <p className="mb-4 text-xs text-muted-foreground md:text-sm">
            .mp4, .mov, .avi, .mkv
          </p>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
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
            Răsfoiește Fișierele
          </Button>
        </div>
      </div>

      {selectedFile && (
        <div className="w-full max-w-2xl space-y-4 rounded-lg border border-border bg-card p-6 shadow-lg">
          <div>
            <div className="mb-3">
              <Label className="text-foreground">Eșantionare: {fps[0]} FPS</Label>
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

          {zones.length > 0 && (
            <div>
              <Label htmlFor="zone-select" className="text-foreground">
                Zonă parcare
              </Label>
              <select
                id="zone-select"
                className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={selectedZoneId ?? ""}
                onChange={(e) =>
                  setSelectedZoneId(e.target.value ? Number(e.target.value) : null)
                }
              >
                {zones.map((z) => (
                  <option key={z.id} value={z.id}>
                    {z.name} ({z.code})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="space-y-3 rounded-md border border-border bg-muted/30 p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="space-y-0.5">
                <Label className="text-foreground">Mod LPR Mobil</Label>
                <p className="text-xs text-muted-foreground">
                  Proiectează numerele pe GPS &amp; verifică locurile de parcare alocate
                </p>
              </div>
              <Switch
                checked={mobileLprEnabled}
                onCheckedChange={(checked) => {
                  setMobileLprEnabled(checked);
                  setPoseError(null);
                }}
              />
            </div>

            {mobileLprEnabled && (
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Label htmlFor="lpr-lat" className="text-xs">Latitudine</Label>
                  <Input
                    id="lpr-lat"
                    inputMode="decimal"
                    placeholder="46.7701"
                    value={latitude}
                    onChange={(e) => setLatitude(e.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="lpr-lon" className="text-xs">Longitudine</Label>
                  <Input
                    id="lpr-lon"
                    inputMode="decimal"
                    placeholder="23.5895"
                    value={longitude}
                    onChange={(e) => setLongitude(e.target.value)}
                  />
                </div>
                <div>
                  <Label htmlFor="lpr-hdg" className="text-xs">Direcție °</Label>
                  <Input
                    id="lpr-hdg"
                    inputMode="decimal"
                    placeholder="90"
                    value={headingDeg}
                    onChange={(e) => setHeadingDeg(e.target.value)}
                  />
                </div>
              </div>
            )}

            {poseError && (
              <p className="text-xs text-red-600 dark:text-red-400">{poseError}</p>
            )}
          </div>

          <Button
            onClick={handleSubmit}
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
          >
            Procesează Video
          </Button>
        </div>
      )}
    </div>
  );
}

function isValidVideoFile(file: File): boolean {
  const validTypes = [
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
  ];
  const validExtensions = [".mp4", ".mov", ".avi", ".mkv"];

  const hasValidType = validTypes.includes(file.type);
  const hasValidExtension = validExtensions.some((ext) =>
    file.name.toLowerCase().endsWith(ext)
  );

  return hasValidType || hasValidExtension;
}
