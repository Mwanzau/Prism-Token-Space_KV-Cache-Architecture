use std::sync::atomic::{AtomicBool, Ordering};

pub const BUFFER_SIZE: usize = 128 * 1024 * 1024;

pub struct RingBuffer {
    pub buffer_a: Vec<u8>,
    pub buffer_b: Vec<u8>,
    pub read_index: usize,
    pub write_index: usize,
    pub swap_flag: AtomicBool,
}

impl RingBuffer {
    pub fn new() -> Self {
        let half = BUFFER_SIZE / 2;
        Self {
            buffer_a: vec![0; half],
            buffer_b: vec![0; half],
            read_index: 0,
            write_index: 0,
            swap_flag: AtomicBool::new(false),
        }
    }

    pub fn swap_buffers(&self) {
        self.swap_flag.store(!self.swap_flag.load(Ordering::Relaxed), Ordering::Release);
    }
}
