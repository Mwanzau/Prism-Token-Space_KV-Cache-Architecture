#include <cstdint>
#include <vector>

extern "C" {

void load_tokens(const uint32_t* tokens, size_t count) {
    // Placeholder for llamacpp C API token injection.
}

std::vector<uint8_t> dump_kv_cache() {
    return {};
}

void restore_kv_cache(const uint8_t* data, size_t size) {
    // Placeholder for KV cache restoration.
}

}
