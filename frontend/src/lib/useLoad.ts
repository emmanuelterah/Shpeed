import { useEffect, useState } from "react";

/**
 * Run an async loader when the component mounts or its key changes.
 * Returns undefined while loading, then the loader's result (null for nothing published).
 */
export function useLoad<T>(load: () => Promise<T | null>, key = ""): T | null | undefined {
  const [state, setState] = useState<{ key: string; value: T | null } | undefined>(undefined);
  useEffect(() => {
    let live = true;
    load().then((value) => live && setState({ key, value }));
    return () => {
      live = false;
    };
    // The key identifies the request; the loader itself is recreated on every render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state?.key === key ? state.value : undefined;
}
