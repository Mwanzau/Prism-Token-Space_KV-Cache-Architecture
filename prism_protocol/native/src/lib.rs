//! Native Prism Token-Space runtime.

use std::path::Path;
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

pub const BLOCK_SIZE: usize = 4096;

pub struct RingBuffer {
    buffer_a: Vec<u8>,
    buffer_b: Vec<u8>,
    read_pos: usize,
    write_pos: usize,
    max_size: usize,
}

impl RingBuffer {
    pub fn new(max_size: usize) -> Self {
        let allocate = |size| vec![0; size];
        Self {
            buffer_a: allocate(max_size / 2),
            buffer_b: allocate(max_size / 2),
            read_pos: 0,
            write_pos: 0,
            max_size,
        }
    }
}

pub struct LlamaBridge;

impl LlamaBridge {
    pub fn load_tokens(_tokens: &[u32]) {
        // Native llama.cpp evaluation bridge entrypoint placeholder.
    }

    pub fn dump_kv_cache() -> Vec<u8> {
        vec![]
    }

    pub fn restore_kv_cache(_data: &[u8]) {
        // Restore KV cache from serialized bytes.
    }
}

pub struct AsyncReader {
    file: Arc<File>,
    read_offset: u64,
}

impl AsyncReader {
    pub async fn open(path: impl AsRef<Path>) -> tokio::io::Result<Self> {
        let file = File::open(path).await?;
        Ok(Self { file: Arc::new(file), read_offset: 0 })
    }

    pub async fn read_block(&mut self, size: usize) -> tokio::io::Result<Vec<u8>> {
        let mut buffer = vec![0; size];
        self.file.seek(std::io::SeekFrom::Start(self.read_offset)).await?;
        let n = self.file.read(&mut buffer).await?;
        self.read_offset += n as u64;
        buffer.truncate(n);
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn async_reader_reads_first_block() {
        let path = std::env::temp_dir().join("prism_native_test.bin");
        tokio::fs::write(&path, vec![0u8; BLOCK_SIZE]).await.unwrap();
        let mut reader = AsyncReader::open(&path).await.unwrap();
        let chunk = reader.read_block(BLOCK_SIZE).await.unwrap();
        assert_eq!(chunk.len(), BLOCK_SIZE);
    }
}
