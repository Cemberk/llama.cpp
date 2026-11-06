#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <map>
#include <string>
#include <memory>
#include <vector>

struct llama_model;

struct llama_moe_cache_options {
    size_t size = 0;      // device bytes for the cache
    std::string trace;    // file the experts selected in cached layers are appended to, empty = off
    std::string profile;  // file of expert usage, read at start (warm start) and written at the end, empty = off
    enum llama_moe_cache_policy policy = LLAMA_MOE_CACHE_POLICY_SLRU;
    bool hybrid = false;  // small ubatches: misses are computed on the CPU instead of uploaded
};

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_moe_cache_options & options);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    // in hybrid mode it also holds a zero expert after the last slot, which the misses read
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // hybrid mode: whether it is on
    bool hybrid() const;

    // hybrid mode: the map of layer il, F32 [3, n_expert] in host memory, per expert:
    // the slot to read on the device (the zero expert for a miss), the expert to compute on the CPU
    // (a miss, or an expert the CPU computes anyway for a hit), and 1 for a miss
    // nullptr if layer il is not cached or the ubatch is too large for hybrid mode
    ggml_tensor * get_hybrid_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    void get_stats(struct llama_moe_cache_stats * stats) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
