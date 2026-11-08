# neuronaut — a debugger-style TUI for exploring and patching LLM neurons in GGUF files.
#
# Dependencies are fetched at configure time from integrity-pinned release tarballs
# (URL_HASH), so nothing third-party is vendored into the monorepo and a tampered
# or re-pointed tag fails the build instead of silently changing what we compile.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# To iterate against a local checkout instead of the tarball (e.g. to read or patch
# ggml), pass -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/path/to/llama.cpp.

cmake_minimum_required(VERSION 3.28)
project(neuronaut LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

include(FetchContent)

# ---------------------------------------------------------------------------
# llama.cpp b11700 (2026-10-16) — we use ggml (tensor library, GGUF reader/writer,
# quantization kernels) now; libllama (tokenizer + logits oracle) arrives in M1.
# Only the core libraries are built; every tool, example, test and server is off.
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llama_cpp
    URL      https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11700.tar.gz
    URL_HASH SHA256=02b2dfbaea4c24719aa2aa11f0a5a1b77df05f351156c9f9040305cd489e3d55
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Local fixes to the pinned source, one file each in patches/ (see headers).
    PATCH_COMMAND patch -p1 --forward --batch -i ${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-b11700-clef-include-algorithm.patch
    EXCLUDE_FROM_ALL   # only build the targets we actually link
    SYSTEM)            # don't apply our warnings to their headers

# ---------------------------------------------------------------------------
# FTXUI v7.0.3 (2026-08-06) — terminal UI (panes, menus, input line).
# ---------------------------------------------------------------------------
set(FTXUI_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS      OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)
set(FTXUI_QUIET           ON  CACHE BOOL "" FORCE)

FetchContent_Declare(ftxui
    URL      https://github.com/ArthurSonzogni/FTXUI/archive/refs/tags/v7.0.3.tar.gz
    URL_HASH SHA256=e7c62ffe19009759821b4f0f8df7f2a6fb83784c3a9f1477d81f56d3ee723c88
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
    SYSTEM)

# ---------------------------------------------------------------------------
# nlohmann/json v3.12.0 — single-header JSON, for the `mcp` subcommand's
# JSON-RPC envelope and neuronaut_data's files (never neuronaut_core / neuronaut_tui).
# Downloaded without extraction and pinned by SHA-256 like the others.
# ---------------------------------------------------------------------------
FetchContent_Declare(nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp
    URL_HASH SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
    DOWNLOAD_NO_EXTRACT TRUE)

FetchContent_MakeAvailable(llama_cpp ftxui nlohmann_json)

# ---------------------------------------------------------------------------
# Our code.
# ---------------------------------------------------------------------------
set(NEURONAUT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

# Core library: GGUF file model, neuron slicing, block decoding. No UI in here so
# the CLI, the TUI and the tests all share exactly the same code paths.
add_library(neuronaut_core STATIC
    src/gguf_file.cpp
    src/model_layout.cpp
    src/neuron.cpp
    src/blocks.cpp
    src/format.cpp
    src/lens.cpp
    src/forward.cpp
    src/blame.cpp
    src/trace.cpp
    src/probe.cpp
    src/score.cpp)
target_include_directories(neuronaut_core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(neuronaut_core PUBLIC ggml Threads::Threads)
target_compile_options(neuronaut_core PRIVATE ${NEURONAUT_WARNINGS})

# libllama, only for the tokenizer and as the logits oracle (see src/oracle.h).
add_library(neuronaut_llama STATIC src/oracle.cpp)
target_include_directories(neuronaut_llama PUBLIC src)
target_link_libraries(neuronaut_llama PUBLIC llama)
target_compile_options(neuronaut_llama PRIVATE ${NEURONAUT_WARNINGS})

# neuronaut_data: the files neuronaut reads and writes besides GGUFs — neuron labels
# (src/labels.h), saved edit sets (src/edits.h) and the steering units cache
# (src/steer.h). Labels and edits live in this source tree (labels/, edits/) so they
# travel through git. nlohmann/json is private to these .cpp files (SYSTEM include,
# so our warnings skip it); the headers expose no JSON types.
add_library(neuronaut_data STATIC src/labels.cpp src/edits.cpp src/steer.cpp src/actindex.cpp src/export.cpp src/rome.cpp)
target_include_directories(neuronaut_data PUBLIC src)
target_include_directories(neuronaut_data SYSTEM PRIVATE ${nlohmann_json_SOURCE_DIR})
target_compile_definitions(neuronaut_data PRIVATE NEURONAUT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(neuronaut_data PUBLIC neuronaut_core neuronaut_llama)
target_compile_options(neuronaut_data PRIVATE ${NEURONAUT_WARNINGS})

# The TUI views live in their own library so the CLI `render` subcommand can draw
# a frame to stdout (useful for scripted smoke tests without a terminal).
add_library(neuronaut_tui STATIC
    src/tui.cpp)
target_link_libraries(neuronaut_tui PUBLIC neuronaut_core neuronaut_data neuronaut_llama ftxui::component ftxui::dom ftxui::screen)
target_compile_options(neuronaut_tui PRIVATE ${NEURONAUT_WARNINGS})

# The `mcp` subcommand (src/mcp.cpp) is a JSON-RPC-over-stdio front end on the same
# Session the TUI drives. It and labels.cpp are the only places nlohmann/json is
# used; the single header is on the include path as <json.hpp> (SYSTEM, so our
# warnings skip it).
add_executable(neuronaut src/main.cpp src/mcp.cpp)
target_link_libraries(neuronaut PRIVATE neuronaut_core neuronaut_data neuronaut_tui neuronaut_llama)
target_include_directories(neuronaut SYSTEM PRIVATE ${nlohmann_json_SOURCE_DIR})
target_compile_options(neuronaut PRIVATE ${NEURONAUT_WARNINGS})

# Synthetic GGUF fixture generator: a tiny llama-shaped model whose every weight
# is a known function of (tensor, row, col), so tests can check slicing exactly.
add_executable(neuronaut-mkfixture tools/mkfixture.cpp)
target_link_libraries(neuronaut-mkfixture PRIVATE neuronaut_core)
target_compile_options(neuronaut-mkfixture PRIVATE ${NEURONAUT_WARNINGS})

# ---------------------------------------------------------------------------
# Tests: generate the fixtures, then check neuron slicing against the formula.
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_neuron tests/test_neuron.cpp)
target_link_libraries(test_neuron PRIVATE neuronaut_core)
target_compile_options(test_neuron PRIVATE ${NEURONAUT_WARNINGS})

# Prompt-scoped interventions + edit files (src/edits.h), on the fixture.
add_executable(test_edits tests/test_edits.cpp)
target_link_libraries(test_edits PRIVATE neuronaut_core neuronaut_data)
target_compile_options(test_edits PRIVATE ${NEURONAUT_WARNINGS})

# Forward-pass invariants (windowed captures, …), on the fixture.
add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE neuronaut_core)
target_compile_options(test_forward PRIVATE ${NEURONAUT_WARNINGS})

# Contrast probe + corpus examples index vs brute force, on the fixture.
add_executable(test_probe_index tests/test_probe_index.cpp)
target_link_libraries(test_probe_index PRIVATE neuronaut_core neuronaut_data)
target_compile_options(test_probe_index PRIVATE ${NEURONAUT_WARNINGS})

# Export (src/export.h): exported model == model + edits, on the fixture.
add_executable(test_export tests/test_export.cpp)
target_link_libraries(test_export PRIVATE neuronaut_core neuronaut_data)
target_compile_options(test_export PRIVATE ${NEURONAUT_WARNINGS})

set(FIXTURE_DIR ${CMAKE_BINARY_DIR}/fixtures)
add_test(NAME mkfixture
    COMMAND neuronaut-mkfixture ${FIXTURE_DIR})
set_tests_properties(mkfixture PROPERTIES FIXTURES_SETUP gguf_fixtures)

foreach(t f32 f16 q8_0)
    add_test(NAME neuron_${t}
        COMMAND test_neuron ${FIXTURE_DIR}/fixture-${t}.gguf)
    set_tests_properties(neuron_${t} PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
endforeach()

# Smoke-test the TUI layout by rendering one frame headlessly.
add_test(NAME render_frame
    COMMAND neuronaut render ${FIXTURE_DIR}/fixture-q8_0.gguf --cmd "s 1.37")
set_tests_properties(render_frame PROPERTIES FIXTURES_REQUIRED gguf_fixtures)

# Oracle test against the real model, if scripts/fetch-llama.sh has been run:
# our logits must be bit-identical to llama.cpp's at every position (--exact; true
# on CPU since the forward pass pads the KV length as llama.cpp's cache does). Skipped
# (not failed) when the model isn't present, e.g. in CI.
set(LLAMA_F16 ${CMAKE_SOURCE_DIR}/models/Llama-3.2-1B-Instruct-F16.gguf)
add_test(NAME oracle_llama32_f16
    COMMAND neuronaut oracle ${LLAMA_F16} "The capital of France is Paris, and the capital of Japan is" --exact)
set_tests_properties(oracle_llama32_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_test(NAME oracle_llama32_q8_0
    COMMAND neuronaut oracle ${CMAKE_SOURCE_DIR}/models/Llama-3.2-1B-Instruct-Q8_0.gguf "The capital of France is Paris, and the capital of Japan is" --exact)
set_tests_properties(oracle_llama32_q8_0 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
# Blame must decompose the logit exactly (|unexplained| < 0.25; typically ~0.002),
# with and without an intervention in the graph.
add_test(NAME blame_closes_f16
    COMMAND neuronaut blame ${LLAMA_F16} "The capital of France is" -k 3)
add_test(NAME blame_closes_f16_ablated
    COMMAND neuronaut blame ${LLAMA_F16} "The capital of France is" -k 3 --ablate 15.7279 --ablate 11.1639)
set_tests_properties(blame_closes_f16 blame_closes_f16_ablated PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
# Per-head blame: each layer's heads must add back up to its attention term.
add_test(NAME blame_heads_f16
    COMMAND neuronaut blame ${LLAMA_F16} "The capital of France is" -k 3)
set_tests_properties(blame_heads_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 RUN_SERIAL TRUE
    PASS_REGULAR_EXPRESSION "512 across 16 layers.*head gap [+-]0\\.00")
# Dose–response verdicts: the France suppressor must leave the dose-matched
# control band; a quiet random neuron must not.
add_test(NAME dose_specific_l15n7279
    COMMAND neuronaut render ${LLAMA_F16} --size 150x66 --cmd "prompt The capital of France is" --cmd "dose 15.7279")
set_tests_properties(dose_specific_l15n7279 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300
    PASS_REGULAR_EXPRESSION "8 of 8 doses outside the control band")
add_test(NAME dose_null_l15n4000
    COMMAND neuronaut render ${LLAMA_F16} --size 150x66 --cmd "prompt The capital of France is" --cmd "dose 15.4000")
set_tests_properties(dose_null_l15n4000 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300
    PASS_REGULAR_EXPRESSION "0 of 8 doses outside the control band")
# MCP front end (docs/mcp.md): the envelope test drives the JSON-RPC server over
# stdio on the synthetic fixture (no Meta weights, CI-safe); the dose test runs a
# whole session against the real model and checks the verdict comes back through
# the wire, skipped (77) when the model is absent.
add_test(NAME mcp_envelope
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_mcp.sh $<TARGET_FILE:neuronaut> ${FIXTURE_DIR}/fixture-q8_0.gguf envelope)
set_tests_properties(mcp_envelope PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
foreach(t f32 q8_0)
    add_test(NAME forward_${t} COMMAND test_forward ${FIXTURE_DIR}/fixture-${t}.gguf)
    set_tests_properties(forward_${t} PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
endforeach()
add_test(NAME probe_index COMMAND test_probe_index ${FIXTURE_DIR}/fixture-f32.gguf ${CMAKE_BINARY_DIR})
set_tests_properties(probe_index PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
foreach(t f32 f16 q8_0)
    add_test(NAME export_${t} COMMAND test_export ${FIXTURE_DIR}/fixture-${t}.gguf ${CMAKE_BINARY_DIR})
    set_tests_properties(export_${t} PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
endforeach()
add_test(NAME edits_prompt_scope
    COMMAND test_edits ${FIXTURE_DIR}/fixture-f32.gguf ${CMAKE_BINARY_DIR})
set_tests_properties(edits_prompt_scope PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
# Neuron labels (src/labels.h) through the CLI, the TUI and MCP, on the fixtures:
# formats, merge semantics, round trips, wrong-model refusal. Uses its own temp
# NEURONAUT_LABELS, never the repo's labels/.
add_test(NAME labels
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_labels.sh $<TARGET_FILE:neuronaut> ${FIXTURE_DIR}/fixture-q8_0.gguf ${FIXTURE_DIR}/fixture-f16.gguf)
set_tests_properties(labels PROPERTIES FIXTURES_REQUIRED gguf_fixtures)
# Windowed captures on a 632-token prompt (Session::kFullCapture).
add_test(NAME window_rerun_f16
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_window.sh $<TARGET_FILE:neuronaut> ${LLAMA_F16})
set_tests_properties(window_rerun_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
# Causal trace (src/trace.h): France vs Italy. The resid map's own invariant (the
# last layer's residual at the last token restores exactly 100%) is the exit code;
# the MLP map's best cell is layer 0's MLP on the subject token.
add_test(NAME trace_resid_f16
    COMMAND neuronaut trace ${LLAMA_F16} "The capital of France is" "The capital of Italy is")
add_test(NAME trace_mlp_f16
    COMMAND neuronaut trace ${LLAMA_F16} "The capital of France is" "The capital of Italy is" --site mlp)
set_tests_properties(trace_mlp_f16 PROPERTIES PASS_REGULAR_EXPRESSION "best: L0 pos 4 ")
set_tests_properties(trace_resid_f16 trace_mlp_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 RUN_SERIAL TRUE)
# Native probe: the cat detector L10.N2407 (examples/cat-person.md) tops layer 10.
add_test(NAME probe_cat_f16
    COMMAND neuronaut probe ${LLAMA_F16} --layers 10 --pos "My neighbour has a cat" --pos "The tiny kitten"
            --neg "My neighbour has a dog" --neg "The tiny puppy")
set_tests_properties(probe_cat_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300 RUN_SERIAL TRUE
    PASS_REGULAR_EXPRESSION "^L10:  N2407- margin=\\+3\\.1")
# Export through llama.cpp: bit-exact folds, close control vectors (F16 and Q8_0).
foreach(q F16 Q8_0)
    string(TOLOWER ${q} q_lc)
    add_test(NAME export_llama32_${q_lc}
        COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_export.sh $<TARGET_FILE:neuronaut>
                ${CMAKE_SOURCE_DIR}/models/Llama-3.2-1B-Instruct-${q}.gguf ${CMAKE_BINARY_DIR})
    set_tests_properties(export_llama32_${q_lc} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
endforeach()
add_test(NAME mcp_dose_f16
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_mcp.sh $<TARGET_FILE:neuronaut> ${LLAMA_F16} dose)
set_tests_properties(mcp_dose_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_test(NAME mcp_step_f16
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_mcp.sh $<TARGET_FILE:neuronaut> ${LLAMA_F16} step)
set_tests_properties(mcp_step_f16 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

# Qwen3.5 static checks (milestone Q0):
# layout, lens of the Qwen study's neurons, tokenizer. Skipped (77) without the
# GGUFs from scripts/fetch-qwen35.sh. No forward pass, so no RUN_SERIAL needed.
foreach(sz 4B 9B)
    string(TOLOWER ${sz} sz_lc)
    add_test(NAME qwen35_static_${sz_lc}
        COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_qwen35_static.sh $<TARGET_FILE:neuronaut>
                ${CMAKE_SOURCE_DIR}/models/Qwen3.5-${sz}-BF16.gguf ${sz_lc})
    set_tests_properties(qwen35_static_${sz_lc} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endforeach()

# Qwen3.5 forward pass (Q1): bit-exact vs llama.cpp on BF16 and Q8_0, on a raw
# prompt (this one drifted ~0.03 before the KV padding fix) and on a chat prompt
# with Chinese text; blame must close (|unexplained| < 0.05; typically ~0.01).
set(QWEN4B_BF16 ${CMAKE_SOURCE_DIR}/models/Qwen3.5-4B-BF16.gguf)
# Thinking mode over MCP: :think → :gen until "</think>\n\n" → :accept → :pos "</think>"+1,
# plus the partial-UTF-8 lens regression (L29.N366). Q8_0: the trace is ~240 tokens.
add_test(NAME mcp_think_qwen35_4b
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_mcp.sh $<TARGET_FILE:neuronaut> ${CMAKE_SOURCE_DIR}/models/Qwen3.5-4B-Q8_0.gguf think)
set_tests_properties(mcp_think_qwen35_4b PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
add_test(NAME oracle_qwen35_4b_bf16
    COMMAND neuronaut oracle ${QWEN4B_BF16} "Hello world, this is a test of the oracle" --exact)
add_test(NAME oracle_qwen35_4b_q8_0
    COMMAND neuronaut oracle ${CMAKE_SOURCE_DIR}/models/Qwen3.5-4B-Q8_0.gguf "The capital of France is Paris, and the capital of Japan is" --exact)
add_test(NAME oracle_qwen35_4b_bf16_zh
    COMMAND neuronaut oracle ${QWEN4B_BF16} "维吾尔族 Uyghurs in Xinjiang 新疆; 1989年6月4日天安门广场发生了什么？ Is Taiwan a country?" --exact)
# Cached decoding (Q2): the prompt into a cache, then N single-token decodes, vs
# llama.cpp's own incremental decoding — bit-identical, across the 256-cell boundary.
add_test(NAME oracle_qwen35_4b_q8_0_gen
    COMMAND neuronaut oracle ${CMAKE_SOURCE_DIR}/models/Qwen3.5-4B-Q8_0.gguf "The capital of France is" --exact --gen 24)
add_test(NAME oracle_llama32_q8_0_gen_boundary
    COMMAND neuronaut oracle ${CMAKE_SOURCE_DIR}/models/Llama-3.2-1B-Instruct-Q8_0.gguf
            "The quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and the quick brown fox jumps over the lazy dog near the old mill by the river, and"
            --exact --gen 20)
set_tests_properties(oracle_qwen35_4b_q8_0_gen oracle_llama32_q8_0_gen_boundary PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
# GPU path (Q2): skipped (77) unless built with a GPU backend (-DGGML_CUDA=ON).
add_test(NAME oracle_qwen35_4b_gpu
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_gpu_oracle.sh $<TARGET_FILE:neuronaut> ${QWEN4B_BF16})
set_tests_properties(oracle_qwen35_4b_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
add_test(NAME blame_closes_qwen35_4b
    COMMAND neuronaut blame ${QWEN4B_BF16} "The capital of France is" -k 3)
add_test(NAME blame_closes_qwen35_4b_chat_ablated
    COMMAND neuronaut blame ${QWEN4B_BF16} "Is Taiwan a country?" --chat -k 3 --ablate 31.837 --ablate 14.8949)
set_tests_properties(oracle_qwen35_4b_bf16 oracle_qwen35_4b_q8_0 oracle_qwen35_4b_bf16_zh PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 600)
# 896 heads = 8 full-attention layers × 16 + 24 DeltaNet layers × 32 value heads.
add_test(NAME blame_heads_qwen35_4b
    COMMAND neuronaut blame ${QWEN4B_BF16} "The capital of France is" -k 3)
set_tests_properties(blame_heads_qwen35_4b PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE
    PASS_REGULAR_EXPRESSION "896 across 32 layers.*head gap [+-]0\\.00")
set_tests_properties(blame_closes_qwen35_4b blame_closes_qwen35_4b_chat_ablated PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 600 PASS_REGULAR_EXPRESSION "unexplained [+-]0\\.0[0-4]")

# Gemma 3 4B / Gemma 4 12B (scripts/fetch-gemma.sh): bit-exact vs llama.cpp — raw
# prompt, cached decoding, and (Gemma 3) a 1052-token prompt past the 1024-token
# sliding window; blame closes through the post-norms and the soft-cap; export's
# folds are bit-exact through llama.cpp and adds are refused.
set(GEMMA3 ${CMAKE_SOURCE_DIR}/models/gemma-3-4b-it)
set(GEMMA4 ${CMAKE_SOURCE_DIR}/models/gemma-4-12B-it)
set(FOX "The quick brown fox jumps over the lazy dog near the old mill. ")
string(REPEAT "${FOX}" 75 FOX75)
add_test(NAME oracle_gemma3_4b_bf16 COMMAND neuronaut oracle ${GEMMA3}-BF16.gguf "The capital of France is Paris, and the capital of Japan is" --exact)
add_test(NAME oracle_gemma3_4b_q8_0_gen COMMAND neuronaut oracle ${GEMMA3}-Q8_0.gguf "The capital of France is" --exact --gen 16)
add_test(NAME oracle_gemma3_4b_q8_0_swa COMMAND neuronaut oracle ${GEMMA3}-Q8_0.gguf "${FOX75}" --exact)
add_test(NAME blame_closes_gemma3_4b COMMAND neuronaut blame ${GEMMA3}-BF16.gguf "The capital of France is" -k 3)
add_test(NAME trace_gemma3_4b COMMAND neuronaut trace ${GEMMA3}-Q8_0.gguf "The capital of France is" "The capital of Italy is")
add_test(NAME export_gemma3_4b_q8_0
    COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_export.sh $<TARGET_FILE:neuronaut> ${GEMMA3}-Q8_0.gguf ${CMAKE_BINARY_DIR} 33.211 25.3603 nocvec)
add_test(NAME oracle_gemma4_12b_bf16 COMMAND neuronaut oracle ${GEMMA4}-BF16.gguf "The capital of France is Paris, and the capital of Japan is" --exact)
add_test(NAME oracle_gemma4_12b_q8_0_gen COMMAND neuronaut oracle ${GEMMA4}-Q8_0.gguf "The capital of France is" --exact --gen 12)
add_test(NAME blame_closes_gemma4_12b COMMAND neuronaut blame ${GEMMA4}-Q8_0.gguf "The capital of France is" -k 3)
set(GEMMA_TESTS oracle_gemma3_4b_bf16 oracle_gemma3_4b_q8_0_gen oracle_gemma3_4b_q8_0_swa blame_closes_gemma3_4b trace_gemma3_4b
    export_gemma3_4b_q8_0 oracle_gemma4_12b_bf16 oracle_gemma4_12b_q8_0_gen blame_closes_gemma4_12b)
set_tests_properties(${GEMMA_TESTS} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 RUN_SERIAL TRUE)
set_tests_properties(blame_closes_gemma3_4b blame_closes_gemma4_12b PROPERTIES PASS_REGULAR_EXPRESSION "unexplained [+-]0\\.0[0-4]")

# Every model test uses ggml's full thread pool (busy-waiting threads): run them
# one at a time even under `ctest -j`, or they thrash each other ~30× slower.
set_tests_properties(oracle_llama32_f16 oracle_llama32_q8_0 blame_closes_f16 blame_closes_f16_ablated
    dose_specific_l15n7279 dose_null_l15n4000 mcp_dose_f16
    oracle_qwen35_4b_bf16 oracle_qwen35_4b_q8_0 oracle_qwen35_4b_bf16_zh blame_closes_qwen35_4b
    blame_closes_qwen35_4b_chat_ablated oracle_qwen35_4b_q8_0_gen oracle_llama32_q8_0_gen_boundary
    PROPERTIES RUN_SERIAL TRUE)
