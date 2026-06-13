fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "neurons.db".to_string());
    let port: u16 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8088);
    neuron_core::server::serve(&path, "127.0.0.1", port, 500).expect("serve");
}
