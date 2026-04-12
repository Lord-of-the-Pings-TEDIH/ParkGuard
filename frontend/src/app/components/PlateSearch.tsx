import { useState, useEffect } from "react";
import { Search } from "lucide-react";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { RomanianPlate } from "./RomanianPlate";
import { TicketStatusBadge } from "./TicketStatusBadge";
import { searchPlates } from "../services/api";
import { formatRelativeTime } from "../utils/format";
import type { Plate } from "../types";

export function PlateSearch() {
  const [query, setQuery] = useState("");
  const [county, setCounty] = useState("");
  const [plates, setPlates] = useState<Plate[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (query.length >= 2 || county) {
        fetchPlates();
      } else {
        setPlates([]);
      }
    }, 300);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, county]);

  const fetchPlates = async () => {
    setLoading(true);
    try {
      const results = await searchPlates(query, county);
      setPlates(results);
    } catch (error) {
      console.error("Failed to search plates:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col lg:border-l border-border bg-card">
      <div className="border-b border-border bg-gradient-to-r from-cyan-50 to-blue-50 p-3 dark:from-cyan-950 dark:to-blue-950 md:p-4">
        <h3 className="mb-3 font-medium text-foreground">Registry</h3>

        <div className="space-y-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              placeholder="Search plates..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          <Input
            placeholder="County code (e.g. B, CJ)"
            value={county}
            onChange={(e) => setCounty(e.target.value.toUpperCase())}
            className="text-center"
          />
        </div>

        {plates.length > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            {plates.length}
          </p>
        )}
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-3 p-3">
          {loading && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Searching...
            </div>
          )}

          {!loading && plates.length === 0 && (query || county) && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              No plates found
            </div>
          )}

          {!loading && plates.length === 0 && !query && !county && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Search plates
            </div>
          )}

          {!loading && plates.map((plate) => (
            <div
              key={plate.id}
              className="rounded-lg border border-border bg-muted p-3 shadow-sm"
            >
              <div className="mb-3 flex items-center justify-between">
                <RomanianPlate text={plate.normalized_text} className="scale-90" />
                <TicketStatusBadge status={plate.last_ticket_status} />
              </div>

              <div className="space-y-1 text-xs text-muted-foreground">
                <div className="flex justify-between">
                  <span>County:</span>
                  <span className="font-medium text-foreground">
                    {plate.county_name} ({plate.county_code})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>First seen:</span>
                  <span>{formatRelativeTime(plate.first_seen_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Last seen:</span>
                  <span>{formatRelativeTime(plate.last_seen_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Detections:</span>
                  <span className="font-mono font-medium text-foreground">
                    {plate.seen_count}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
