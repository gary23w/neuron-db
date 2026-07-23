/* neuron_ffi.h — C ABI over the neuron-db memory engine (staticlib).
 *
 * One entry point mirroring the `neuron` CLI: hand it the same argv you would have handed the
 * subprocess (minus argv[0]) and it returns the bytes that subprocess would have written to
 * STDOUT. Existing stdout parsers keep working unchanged.
 *
 *   const char *args[] = { "observe", "chat:42", "the build needs zig 0.16" };
 *   int rc = 0;
 *   char *out = neuron_call_ex("C:/data/hive.db", 0, 3, args, &rc);
 *   ...
 *   neuron_free(out);
 *
 * Only stdout is reproduced. Commands whose human-readable output goes to stderr in the CLI
 * (notably non-`--json` `import`) return an EMPTY string here — identical to reading the
 * subprocess's stdout. Pass `--json` when you need those counts.
 */
#ifndef NEURON_FFI_H
#define NEURON_FFI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Run one command; returns a NUL-terminated stdout copy the caller must release with neuron_free.
 * `db`  : database path. May be NULL/empty if argv itself carries "--db <path>".
 * `max` : max-facts cap (the CLI's --max). 0 means "default" (500).
 * Returns an empty string (never NULL) on a usage error or unknown command. */
char *neuron_call(const char *db, size_t max, int argc, const char *const *argv);

/* As neuron_call, but also reports the exit code the CLI would have used, when exit_code != NULL:
 *   0   ok
 *   1   IO failure (import/export could not read/write a file)
 *   2   usage error, missing scope, or unknown command
 *   3   no match — get / recall / assoc / chain found nothing
 *   101 a panic inside the engine was caught at the boundary
 * Hosts that treat "subprocess exited non-zero" as a miss should branch on this identically. */
char *neuron_call_ex(const char *db, size_t max, int argc, const char *const *argv, int *exit_code);

/* Release a buffer from neuron_call / neuron_call_ex. NULL is a no-op. */
void neuron_free(char *p);

/* Enable (non-zero) / disable (0) the open-handle cache. OFF by default: it is only sound when this
 * process is the SOLE writer of the db files it touches, because a cached handle carries an
 * in-memory scope cache another writer would silently invalidate. Durability is unchanged either
 * way. Disabling drops all cached handles. */
void neuron_cache(int enable);

/* Version of this shim (not of neuron-core). Static storage; do not free. */
const char *neuron_ffi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_FFI_H */
