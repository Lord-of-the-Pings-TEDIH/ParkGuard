import { useState, useCallback, useEffect, useRef } from "react";
import { Upload } from "lucide-react";
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
      .catch(() => {});
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
    if (file && isValidVideoFile(file)) setSelectedFile(file);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSelectedFile(file);
  }, []);

  const handleSubmit = () => {
    if (!selectedFile) return;
    const pose = parsePose();
    if (pose === "invalid") {
      setPoseError("Introduceți latitudine validă (-90..90), longitudine (-180..180) și direcție (grade).");
      return;
    }
    setPoseError(null);
    onUpload(selectedFile, fps[0], pose, selectedZoneId);
  };

  const openFilePicker = () => {
    if (!isProcessing) fileInputRef.current?.click();
  };

  return (
    <div className="flex h-full flex-col items-center justify-center gap-7 p-10 md:p-16">
      {/* Title */}
      <div className="text-center">
        <div className="mb-2 font-bold text-foreground" style={{ fontSize: 30, letterSpacing: "-0.02em" }}>
          Procesare Video LPR
        </div>
        <p className="text-sm text-muted-foreground">
          Detectare și monitorizare numere de înmatriculare
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`relative w-full max-w-xl cursor-pointer rounded-xl border-2 border-dashed transition-all focus-within:ring-2 focus-within:ring-primary ${
          isDragging
            ? "border-primary bg-primary/5"
            : "border-border bg-card hover:border-primary/50 hover:bg-primary/[0.03]"
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
        <div className="p-12 text-center">
          <div
            className="mx-auto mb-4 flex items-center justify-center rounded-xl bg-primary/10 border border-primary/20"
            style={{ width: 52, height: 52 }}
          >
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <p className="mb-1.5 font-semibold text-foreground" style={{ fontSize: 15 }}>
            {selectedFile ? selectedFile.name : "Trageți fișierul video aici"}
          </p>
          <p className="mb-5 text-xs text-muted-foreground">.mp4 · .mov · .avi · .mkv</p>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); openFilePicker(); }}
            disabled={isProcessing}
            className="border border-border bg-card text-muted-foreground hover:text-foreground hover:border-border/80 rounded-lg px-5 py-2 text-sm cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Răsfoiește
          </button>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
            onChange={handleFileSelect}
            disabled={isProcessing}
          />
        </div>
      </div>

      {/* Config panel */}
      {selectedFile && (
        <div className="w-full max-w-xl rounded-xl border border-border bg-card p-6 shadow-sm space-y-5">

          {/* FPS slider */}
          <div>
            <div className="flex justify-between items-center mb-3">
              <Label className="text-foreground font-medium text-sm">Eșantionare</Label>
              <span className="font-mono text-sm font-bold text-primary">{fps[0]} FPS</span>
            </div>
            <Slider value={fps} onValueChange={setFps} min={1} max={10} step={1} className="w-full" />
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1.5">
              <span>1 FPS (rapid)</span><span>10 FPS (precis)</span>
            </div>
          </div>

          {/* Zone selector */}
          {zones.length > 0 && (
            <div>
              <Label htmlFor="zone-select" className="text-foreground font-medium text-sm block mb-1.5">
                Zonă parcare
              </Label>
              <select
                id="zone-select"
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                value={selectedZoneId ?? ""}
                onChange={(e) => setSelectedZoneId(e.target.value ? Number(e.target.value) : null)}
              >
                {zones.map((z) => (
                  <option key={z.id} value={z.id}>{z.name} ({z.code})</option>
                ))}
              </select>
            </div>
          )}

          {/* Mobile LPR */}
          <div className="rounded-lg border border-border bg-muted/30 p-3 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <Label className="text-foreground text-sm font-medium">Mod LPR Mobil</Label>
                <p className="text-[11px] text-muted-foreground leading-tight mt-0.5">
                  GPS + verificare locuri alocate
                </p>
              </div>
              <Switch
                checked={mobileLprEnabled}
                onCheckedChange={(checked) => { setMobileLprEnabled(checked); setPoseError(null); }}
              />
            </div>
            {mobileLprEnabled && (
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: "lpr-lat", label: "Latitudine", placeholder: "46.7701", value: latitude, onChange: setLatitude },
                  { id: "lpr-lon", label: "Longitudine", placeholder: "23.5895", value: longitude, onChange: setLongitude },
                  { id: "lpr-hdg", label: "Direcție °", placeholder: "90", value: headingDeg, onChange: setHeadingDeg },
                ].map((f) => (
                  <div key={f.id}>
                    <Label htmlFor={f.id} className="text-[10px] text-muted-foreground">{f.label}</Label>
                    <input
                      id={f.id}
                      inputMode="decimal"
                      placeholder={f.placeholder}
                      value={f.value}
                      onChange={(e) => f.onChange(e.target.value)}
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                    />
                  </div>
                ))}
              </div>
            )}
            {poseError && (
              <p className="text-[10px] text-destructive">{poseError}</p>
            )}
          </div>

          {/* Submit */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={isProcessing}
            className="w-full rounded-lg bg-primary text-primary-foreground font-semibold py-2.5 text-sm cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
          >
            Procesează Video
          </button>
        </div>
      )}
    </div>
  );
}

function isValidVideoFile(file: File): boolean {
  const validTypes = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"];
  const validExtensions = [".mp4", ".mov", ".avi", ".mkv"];
  return validTypes.includes(file.type) || validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext));
}
