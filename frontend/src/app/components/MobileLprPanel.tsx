import { useEffect, useState } from "react";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import type { MobileLprPose } from "../types";

interface MobileLprPanelProps {
  /** Notifies parent when the pose becomes (in)valid.  Pose is null when the
   * toggle is off OR when any field is missing/out-of-range. */
  onPoseChange: (pose: MobileLprPose | null) => void;
}

export function MobileLprPanel({ onPoseChange }: MobileLprPanelProps) {
  const [enabled, setEnabled] = useState(false);
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [headingDeg, setHeadingDeg] = useState("");

  useEffect(() => {
    if (!enabled) {
      onPoseChange(null);
      return;
    }
    const lat = Number.parseFloat(latitude);
    const lon = Number.parseFloat(longitude);
    const hdg = Number.parseFloat(headingDeg);
    const valid =
      !Number.isNaN(lat) && lat >= -90 && lat <= 90 &&
      !Number.isNaN(lon) && lon >= -180 && lon <= 180 &&
      !Number.isNaN(hdg);
    onPoseChange(
      valid
        ? { latitude: lat, longitude: lon, headingDeg: ((hdg % 360) + 360) % 360 }
        : null,
    );
  }, [enabled, latitude, longitude, headingDeg, onPoseChange]);

  const isInvalid =
    enabled &&
    (latitude !== "" || longitude !== "" || headingDeg !== "") &&
    !isPoseComplete(latitude, longitude, headingDeg);

  return (
    <div className="space-y-2 rounded-md border border-border bg-muted/30 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="space-y-0.5">
          <Label className="text-foreground">LPR Mobil</Label>
          <p className="text-[11px] leading-tight text-muted-foreground">
            Poziția GPS pentru următoarea sesiune lansată
          </p>
        </div>
        <Switch checked={enabled} onCheckedChange={setEnabled} />
      </div>

      {enabled && (
        <>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <Label htmlFor="lpr-lat" className="text-[10px]">Lat</Label>
              <Input
                id="lpr-lat"
                inputMode="decimal"
                placeholder="46.7700"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                className="h-8 text-xs"
              />
            </div>
            <div>
              <Label htmlFor="lpr-lon" className="text-[10px]">Lon</Label>
              <Input
                id="lpr-lon"
                inputMode="decimal"
                placeholder="23.5895"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                className="h-8 text-xs"
              />
            </div>
            <div>
              <Label htmlFor="lpr-hdg" className="text-[10px]">Direcție °</Label>
              <Input
                id="lpr-hdg"
                inputMode="decimal"
                placeholder="0"
                value={headingDeg}
                onChange={(e) => setHeadingDeg(e.target.value)}
                className="h-8 text-xs"
              />
            </div>
          </div>
          {isInvalid && (
            <p className="text-[10px] text-red-600 dark:text-red-400">
              Lat -90..90, Lon -180..180, direcția orice număr.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function isPoseComplete(lat: string, lon: string, hdg: string): boolean {
  const latN = Number.parseFloat(lat);
  const lonN = Number.parseFloat(lon);
  const hdgN = Number.parseFloat(hdg);
  return (
    !Number.isNaN(latN) && latN >= -90 && latN <= 90 &&
    !Number.isNaN(lonN) && lonN >= -180 && lonN <= 180 &&
    !Number.isNaN(hdgN)
  );
}
