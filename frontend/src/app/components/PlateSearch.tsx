import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { Input } from "./ui/input";
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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const trimmedQuery = query.trim();
    const normalizedCounty = county.trim().toUpperCase();
    const shouldSearch = trimmedQuery.length >= 2 || normalizedCounty.length > 0;

    if (!shouldSearch) {
      setPlates([]);
      setError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => {
      void fetchPlates(trimmedQuery, normalizedCounty, controller.signal);
    }, 300);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [query, county]);

  const fetchPlates = async (
    searchQuery: string,
    searchCounty: string,
    signal: AbortSignal
  ) => {
    setLoading(true);
    setError(null);
    try {
      const results = await searchPlates(searchQuery, searchCounty, signal);
      setPlates(results);
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      console.error("Failed to search plates:", error);
      setError(error instanceof Error ? error.message : "Search failed");
    } finally {
      if (!signal.aborted) {
        setLoading(false);
      }
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col border-border bg-card lg:border-l">
      <div className="border-b border-border bg-gradient-to-r from-cyan-50 to-blue-50 p-3 dark:from-cyan-950 dark:to-blue-950 md:p-4">
        <h3 className="mb-3 font-medium text-foreground">Registru</h3>

        <div className="space-y-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              placeholder="Caută numere..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          <Input
            placeholder="Cod județ (ex: B, CJ)"
            value={county}
            onChange={(e) =>
              setCounty(e.target.value.toUpperCase().replace(/[^A-Z]/g, "").slice(0, 2))
            }
            className="text-center"
          />
        </div>

        {plates.length > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            {plates.length}
          </p>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="space-y-3 p-3">
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          {loading && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Se caută...
            </div>
          )}

          {!loading && plates.length === 0 && (query || county) && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Nu s-au găsit numere
            </div>
          )}

          {!loading && plates.length === 0 && !query && !county && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              Caută numere
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
                  <span>Județ:</span>
                  <span className="font-medium text-foreground">
                    {plate.county_name} ({plate.county_code})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Prima detectare:</span>
                  <span>{formatRelativeTime(plate.first_seen_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Ultima detectare:</span>
                  <span>{formatRelativeTime(plate.last_seen_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Detectări:</span>
                  <span className="font-mono font-medium text-foreground">
                    {plate.seen_count}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
