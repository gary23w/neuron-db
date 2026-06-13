fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().or_else(|| std::env::var("NEURON_DB").ok()).unwrap_or_else(|| "neurons.db".to_string());
    let port: u16 = a.get(2).and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("NEURON_PORT").ok().and_then(|s| s.parse().ok())).unwrap_or(8088);
    let host = std::env::var("NEURON_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    neuron_core::server::serve(&path, &host, port, 500).expect("serve");
}
