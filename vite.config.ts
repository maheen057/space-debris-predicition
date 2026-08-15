// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - TanStack devtools (dev-only, first), tanstackStart, viteReact, tailwindcss, tsConfigPaths,
//     nitro (build-only using cloudflare as a default target), VITE_* env injection, @ path alias,
//     React/TanStack dedupe, error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

/**
 * The dev-only source tagger injects `data-tsd-source` props into every JSX
 * element. react-three-fiber throws on unknown dashed props applied to three.js
 * objects, which crashes the 3D globe in dev. Strip the tag inside the R3F
 * scene tree only; the rest of the app keeps its tagging.
 */
const stripSourceTagsInScene = {
  name: "strip-source-tags-in-r3f-scene",
  enforce: "post" as const,
  transform(code: string, id: string) {
    if (!id.includes("/src/ssa/")) return null;
    if (!code.includes("data-tsd-source")) return null;
    return {
      code: code
        .replace(/["']data-tsd-source["']\s*:\s*(["'])(?:(?!\1).)*\1\s*,?/g, "")
        .replace(/["']data-tsd-source["']\s*:\s*[A-Za-z0-9_$.]+\s*,?/g, ""),
      map: null,
    };
  },
};


export default defineConfig({
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    // nitro/vite builds from this
    server: { entry: "server" },
  },
  vite: {
    plugins: [stripSourceTagsInScene],
  },
});
