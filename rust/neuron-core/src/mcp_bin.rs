//! neuron-mcp: stdio MCP server exposing NeuronDB as LLM memory.
//! Configure your MCP client to launch this binary; it speaks JSON-RPC 2.0 on stdio.
//! Env: NEURON_MCP_DB (db file path), NEURON_MAX_FACTS (per-scope cap).
//!
//! Run `neuron-mcp --config` to print a ready-to-paste client config for this machine.
fn main() {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("--config") | Some("config") => print_config(),
        Some("--help") | Some("-h") => print_help(),
        _ => {
            if let Err(e) = neuron_core::mcp::serve_stdio() {
                eprintln!("neuron-mcp error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Print a copy-paste MCP client config that points at this exact binary, so adding persistent
/// memory to Claude Desktop / Cursor / Claude Code is one paste with no path-editing.
fn print_config() {
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "neuron-mcp".into());
    let exe_json = exe.replace('\\', "\\\\"); // JSON-escape Windows backslashes
    let db = std::env::var("NEURON_MCP_DB").unwrap_or_else(|_| "neuron.db".into());
    println!(
        "# Add this to your MCP client config, then restart the client.\n\
         # Claude Desktop: claude_desktop_config.json   Cursor: ~/.cursor/mcp.json\n\
         {{\n  \"mcpServers\": {{\n    \"neuron\": {{\n      \"command\": \"{exe}\",\n      \"env\": {{ \"NEURON_MCP_DB\": \"{db}\" }}\n    }}\n  }}\n}}\n\n\
         # Claude Code (one command):\n\
         claude mcp add neuron --env NEURON_MCP_DB={db} -- \"{exe}\"",
        exe = exe_json, db = db
    );
}

fn print_help() {
    println!(
        "neuron-mcp — persistent associative memory for any MCP client (stdio, JSON-RPC 2.0)\n\n\
         USAGE:\n  neuron-mcp            run the server (clients launch this)\n  \
         neuron-mcp --config   print a ready-to-paste client config for this machine\n  \
         neuron-mcp --help     show this help\n\n\
         ENV:\n  NEURON_MCP_DB    db file path (default: neuron.db)\n  \
         NEURON_MAX_FACTS  per-scope fact cap\n  NEURON_MCP_LOG=1  log per-call synapse timing to stderr\n\n\
         TOOLS:  recall, recall_value, remember, forget, stats"
    );
}
