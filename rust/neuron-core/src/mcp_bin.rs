//! neuron-mcp: stdio MCP server exposing NeuronDB as LLM memory.
//! Configure your MCP client to launch this binary; it speaks JSON-RPC 2.0 on stdio.
//! Env: NEURON_MCP_DB (db file path), NEURON_MAX_FACTS (per-scope cap).
fn main() {
    if let Err(e) = neuron_core::mcp::serve_stdio() {
        eprintln!("neuron-mcp error: {}", e);
        std::process::exit(1);
    }
}
