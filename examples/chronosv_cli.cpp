/*
 * chronosv_cli — small REPL for exercising the ChronosVector C++ API
 * without writing a test harness. Not a supported product; purely a
 * developer / debug tool.
 *
 * Build:
 *   cmake -S . -B build -DCHRONOSV_BUILD_EXAMPLES=ON
 *   cmake --build build --target chronosv_cli
 *
 * Run:
 *   ./build/examples/chronosv_cli
 *
 * Commands (one per line; blank line & lines starting with # are ignored):
 *   create dim=D [cap=C] [metric=cosine|euclidean] [window_ms=N] [interval_ms=N]
 *   preallocate <sensor>
 *   append <sensor> <ts_ms> <v1,v2,...>
 *   query <sensor> <v1,v2,...> [n=N]
 *   range <sensor> <t_start_ms> <t_end_ms> [max=N]
 *   anomaly <sensor> <v1,v2,...> [t=THRESHOLD]
 *   drop <sensor>
 *   list
 *   stats
 *   maintain <window_ms>
 *   evict                       # force one eviction pass (calls maintain with current window)
 *   close
 *   help
 *   quit | exit
 */
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "chronosv/chronos_vector.h"
#include "chronosv/chronos_vector.hpp"

namespace {

/* ------------------------------ tiny parser ------------------------------ */

std::vector<std::string_view> tokenize(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= line.size()) break;
        std::size_t start = i;
        while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        out.emplace_back(line.substr(start, i - start));
    }
    return out;
}

std::optional<std::string_view> kv_value(std::string_view tok, std::string_view key) {
    if (tok.size() <= key.size() + 1) return std::nullopt;
    if (tok.substr(0, key.size()) != key) return std::nullopt;
    if (tok[key.size()] != '=') return std::nullopt;
    return tok.substr(key.size() + 1);
}

std::optional<std::int64_t> parse_i64(std::string_view s) {
    std::int64_t v = 0;
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || p != s.data() + s.size()) return std::nullopt;
    return v;
}
std::optional<std::uint64_t> parse_u64(std::string_view s) {
    std::uint64_t v = 0;
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || p != s.data() + s.size()) return std::nullopt;
    return v;
}
std::optional<float> parse_f32(std::string_view s) {
    // std::from_chars for float requires libc++/libstdc++ recent enough.
    // Apple Clang 21 supports it; fall back to strtof if not.
    float v = 0;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc() || p != s.data() + s.size()) return std::nullopt;
#else
    std::string tmp(s);
    char* end = nullptr;
    v = std::strtof(tmp.c_str(), &end);
    if (!end || *end != '\0') return std::nullopt;
#endif
    return v;
}

std::optional<std::vector<float>> parse_vec(std::string_view s) {
    std::vector<float> out;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        std::string_view field = (j == std::string_view::npos)
                                 ? s.substr(i) : s.substr(i, j - i);
        auto v = parse_f32(field);
        if (!v) return std::nullopt;
        out.push_back(*v);
        if (j == std::string_view::npos) break;
        i = j + 1;
    }
    return out;
}

const char* error_str(chronosv::Error e) {
    return chronosv_error_string(static_cast<chronosv_error_t>(e));
}

void print_uuid(const std::uint8_t u[16]) {
    // 8-4-4-4-12 canonical form.
    std::printf("%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                u[0],u[1],u[2],u[3], u[4],u[5], u[6],u[7],
                u[8],u[9], u[10],u[11],u[12],u[13],u[14],u[15]);
}

/* ------------------------------ CLI state ------------------------------ */

struct CliState {
    std::optional<chronosv::Engine> engine;
    std::uint32_t dim = 0;

    bool require_engine() {
        if (!engine) {
            std::cerr << "no engine — create one first (`create dim=N`)\n";
            return false;
        }
        return true;
    }
};

/* ------------------------------ commands ------------------------------ */

void cmd_help() {
    std::cout <<
"commands:\n"
"  create dim=D [cap=C] [metric=cosine|euclidean] [window_ms=N] [interval_ms=N]\n"
"  preallocate <sensor>\n"
"  append <sensor> <ts_ms> <v1,v2,...>\n"
"  query <sensor> <v1,v2,...> [n=N]              # default n=5\n"
"  range <sensor> <t_start_ms> <t_end_ms> [max=N] # default max=10\n"
"  anomaly <sensor> <v1,v2,...> [t=THRESHOLD]    # default t=0.5\n"
"  drop <sensor>\n"
"  list\n"
"  stats\n"
"  maintain <window_ms>                          # updates window + triggers evict\n"
"  close\n"
"  help | quit | exit\n";
}

void cmd_create(CliState& s, const std::vector<std::string_view>& toks) {
    if (s.engine) { std::cerr << "already have an engine — close first\n"; return; }
    chronosv_config_t cfg{};
    cfg.abi_version = CHRONOSV_ABI_VERSION;

    for (std::size_t i = 1; i < toks.size(); ++i) {
        if (auto v = kv_value(toks[i], "dim")) {
            auto x = parse_u64(*v); if (!x) { std::cerr << "bad dim\n"; return; }
            cfg.dim = static_cast<std::uint32_t>(*x);
        } else if (auto v = kv_value(toks[i], "cap")) {
            auto x = parse_u64(*v); if (!x) { std::cerr << "bad cap\n"; return; }
            cfg.ring_capacity = *x;
        } else if (auto v = kv_value(toks[i], "metric")) {
            if      (*v == "cosine")    cfg.distance_metric = CHRONOSV_METRIC_COSINE;
            else if (*v == "euclidean") cfg.distance_metric = CHRONOSV_METRIC_EUCLIDEAN;
            else { std::cerr << "bad metric (cosine|euclidean)\n"; return; }
        } else if (auto v = kv_value(toks[i], "window_ms")) {
            auto x = parse_i64(*v); if (!x) { std::cerr << "bad window_ms\n"; return; }
            cfg.window_duration_ms = *x;
        } else if (auto v = kv_value(toks[i], "interval_ms")) {
            auto x = parse_i64(*v); if (!x) { std::cerr << "bad interval_ms\n"; return; }
            cfg.eviction_interval_ms = *x;
        } else {
            std::cerr << "unknown arg: " << toks[i] << "\n";
            return;
        }
    }
    if (cfg.dim == 0) { std::cerr << "dim is required\n"; return; }

    auto r = chronosv::Engine::Create(cfg);
    if (!r.has_value()) {
        std::cerr << "create failed: " << error_str(r.error()) << "\n";
        return;
    }
    s.engine.emplace(std::move(*r));
    s.dim = cfg.dim;
    std::cout << "engine created (dim=" << cfg.dim << ")\n";
}

void cmd_preallocate(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() != 2) { std::cerr << "usage: preallocate <sensor>\n"; return; }
    const std::string sid(toks[1]);
    auto err = chronosv_preallocate_sensor(s.engine->raw(), sid.c_str());
    std::cout << (err == CHRONOSV_OK ? "ok\n" : chronosv_error_string(err)) << "\n";
}

void cmd_append(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() != 4) { std::cerr << "usage: append <sensor> <ts_ms> <v1,v2,...>\n"; return; }
    auto ts = parse_i64(toks[2]);
    auto vec = parse_vec(toks[3]);
    if (!ts || !vec) { std::cerr << "bad ts or vector\n"; return; }
    if (vec->size() != s.dim) {
        std::cerr << "vec dim " << vec->size() << " != engine dim " << s.dim << "\n"; return;
    }
    const std::string sid(toks[1]);
    auto err = chronosv_append(s.engine->raw(), sid.c_str(), *ts,
                               vec->data(), vec->size(), nullptr);
    std::cout << chronosv_error_string(err) << "\n";
}

void cmd_query(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() < 3) { std::cerr << "usage: query <sensor> <v1,v2,...> [n=N]\n"; return; }
    int n = 5;
    for (std::size_t i = 3; i < toks.size(); ++i) {
        if (auto v = kv_value(toks[i], "n")) {
            auto x = parse_u64(*v); if (!x) { std::cerr << "bad n\n"; return; }
            n = static_cast<int>(*x);
        }
    }
    auto q = parse_vec(toks[2]);
    if (!q) { std::cerr << "bad query vector\n"; return; }
    if (q->size() != s.dim) {
        std::cerr << "vec dim " << q->size() << " != engine dim " << s.dim << "\n"; return;
    }
    std::vector<std::int64_t> ts(n);
    std::vector<float>        sc(n);
    int count = 0;
    const std::string sid(toks[1]);
    auto err = chronosv_query_nearest_n(s.engine->raw(), sid.c_str(),
                                        q->data(), q->size(), n,
                                        ts.data(), sc.data(), &count);
    if (err < 0) { std::cerr << "error: " << chronosv_error_string(err) << "\n"; return; }
    for (int i = 0; i < count; ++i) {
        std::printf("  ts=%lld  score=%.6f\n",
                    static_cast<long long>(ts[i]), sc[i]);
    }
    if (err > 0) std::cout << "warn: " << chronosv_error_string(err) << "\n";
}

void cmd_range(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() < 4) { std::cerr << "usage: range <sensor> <t_start> <t_end> [max=N]\n"; return; }
    int max = 10;
    for (std::size_t i = 4; i < toks.size(); ++i) {
        if (auto v = kv_value(toks[i], "max")) {
            auto x = parse_u64(*v); if (!x) { std::cerr << "bad max\n"; return; }
            max = static_cast<int>(*x);
        }
    }
    auto t0 = parse_i64(toks[2]);
    auto t1 = parse_i64(toks[3]);
    if (!t0 || !t1) { std::cerr << "bad ts\n"; return; }
    std::vector<std::int64_t> ts(max);
    std::vector<float>        vecs(static_cast<std::size_t>(max) * s.dim);
    int count = 0;
    const std::string sid(toks[1]);
    auto err = chronosv_query_range(s.engine->raw(), sid.c_str(), *t0, *t1,
                                    ts.data(), vecs.data(), max, &count);
    if (err < 0) { std::cerr << "error: " << chronosv_error_string(err) << "\n"; return; }
    for (int i = 0; i < count; ++i) {
        std::printf("  ts=%lld  vec=[", static_cast<long long>(ts[i]));
        for (std::uint32_t d = 0; d < s.dim; ++d) {
            std::printf("%s%.4f", d == 0 ? "" : ",", vecs[i * s.dim + d]);
        }
        std::printf("]\n");
    }
    if (err > 0) std::cout << "warn: " << chronosv_error_string(err) << "\n";
}

void cmd_anomaly(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() < 3) { std::cerr << "usage: anomaly <sensor> <vec> [t=THRESHOLD]\n"; return; }
    float threshold = 0.5f;
    for (std::size_t i = 3; i < toks.size(); ++i) {
        if (auto v = kv_value(toks[i], "t")) {
            auto x = parse_f32(*v); if (!x) { std::cerr << "bad t\n"; return; }
            threshold = *x;
        }
    }
    auto v = parse_vec(toks[2]);
    if (!v || v->size() != s.dim) { std::cerr << "bad vector\n"; return; }
    int is_anom = 0;
    const std::string sid(toks[1]);
    auto err = chronosv_detect_anomaly(s.engine->raw(), sid.c_str(),
                                       v->data(), v->size(), threshold, &is_anom);
    if (err < 0) { std::cerr << "error: " << chronosv_error_string(err) << "\n"; return; }
    std::cout << (is_anom ? "anomaly" : "normal") << "\n";
}

void cmd_drop(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() != 2) { std::cerr << "usage: drop <sensor>\n"; return; }
    const std::string sid(toks[1]);
    auto err = chronosv_drop_sensor(s.engine->raw(), sid.c_str());
    std::cout << chronosv_error_string(err) << "\n";
}

void cmd_list(CliState& s) {
    if (!s.require_engine()) return;
    constexpr std::size_t kMax = 1024;
    char* ids[kMax] = {};
    std::size_t count = 0;
    auto err = chronosv_list_sensors(s.engine->raw(), ids, kMax, &count);
    if (err < 0) { std::cerr << "error: " << chronosv_error_string(err) << "\n"; return; }
    for (std::size_t i = 0; i < count; ++i) {
        std::cout << "  " << ids[i] << "\n";
        std::free(ids[i]);
    }
    std::cout << count << " sensor(s)\n";
}

void cmd_stats(CliState& s) {
    if (!s.require_engine()) return;
    chronosv_stats_t st{};
    if (chronosv_get_stats(s.engine->raw(), &st) != CHRONOSV_OK) {
        std::cerr << "get_stats failed\n"; return;
    }
    std::cout << "uuid:                       ";
    print_uuid(st.uuid);
    std::cout << "\n"
              << "abi_version:                " << st.abi_version              << "\n"
              << "sensor_count:               " << st.sensor_count             << "\n"
              << "sensor_cap:                 " << st.sensor_cap               << "\n"
              << "total_appends:              " << st.total_appends            << "\n"
              << "total_queries:              " << st.total_queries            << "\n"
              << "total_anomaly_checks:       " << st.total_anomaly_checks     << "\n"
              << "total_evictions:            " << st.total_evictions          << "\n"
              << "total_dropped_sensors:      " << st.total_dropped_sensors    << "\n"
              << "total_overwrite_events:     " << st.total_overwrite_events   << "\n"
              << "total_overwritten_entries:  " << st.total_overwritten_entries<< "\n"
              << "hot_bytes:                  " << st.hot_bytes                << "\n";
}

void cmd_maintain(CliState& s, const std::vector<std::string_view>& toks) {
    if (!s.require_engine()) return;
    if (toks.size() != 2) { std::cerr << "usage: maintain <window_ms>\n"; return; }
    auto ms = parse_i64(toks[1]);
    if (!ms) { std::cerr << "bad window_ms\n"; return; }
    auto err = chronosv_maintain_sliding_window(s.engine->raw(), *ms);
    std::cout << chronosv_error_string(err) << "\n";
}

void cmd_close(CliState& s) {
    if (!s.engine) return;
    auto err = chronosv_close(s.engine->raw());
    std::cout << chronosv_error_string(err) << "\n";
}

}  // namespace

int main() {
    CliState s;
    std::cout << "chronosv_cli " << chronosv_version_string()
              << " — type `help` for commands\n";

    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) break;

        // strip trailing whitespace
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') continue;

        auto toks = tokenize(line);
        if (toks.empty()) continue;
        const auto& cmd = toks[0];

        if      (cmd == "help")        cmd_help();
        else if (cmd == "quit" || cmd == "exit") break;
        else if (cmd == "create")      cmd_create(s, toks);
        else if (cmd == "preallocate") cmd_preallocate(s, toks);
        else if (cmd == "append")      cmd_append(s, toks);
        else if (cmd == "query")       cmd_query(s, toks);
        else if (cmd == "range")       cmd_range(s, toks);
        else if (cmd == "anomaly")     cmd_anomaly(s, toks);
        else if (cmd == "drop")        cmd_drop(s, toks);
        else if (cmd == "list")        cmd_list(s);
        else if (cmd == "stats")       cmd_stats(s);
        else if (cmd == "maintain")    cmd_maintain(s, toks);
        else if (cmd == "close")       cmd_close(s);
        else std::cerr << "unknown command: " << cmd << " (type `help`)\n";
    }
    return 0;
}
