#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/nn/embedding.h"
#include "deeptiny/nn/gated_relu.h"
#include "deeptiny/types.h"

namespace {

using Clock = std::chrono::steady_clock;

struct ProfileEntry {
  std::chrono::duration<double> total{};
  uint64_t calls = 0;
};

using ProfileMap = std::unordered_map<std::string, ProfileEntry>;

std::vector<std::string> Tokenize(const std::string& text) {
  std::istringstream stream(text);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

uint64_t ParseIterations(int argc, char** argv) {
  constexpr uint64_t kDefaultIterations = 5000;
  if (argc <= 1) {
    return kDefaultIterations;
  }

  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(argv[1], &end, 10);
  if (end == argv[1] || *end != '\0' || parsed == 0) {
    throw std::runtime_error(
        "Expected optional iterations argument to be a positive integer.");
  }
  return static_cast<uint64_t>(parsed);
}

std::vector<int64_t> EncodeTokens(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, int64_t>& vocabulary) {
  std::vector<int64_t> token_ids;
  token_ids.reserve(tokens.size());

  for (const auto& token : tokens) {
    const auto it = vocabulary.find(token);
    if (it == vocabulary.end()) {
      throw std::runtime_error("Unknown token in input: \"" + token + "\"");
    }
    token_ids.push_back(it->second);
  }
  return token_ids;
}

template <typename Fn>
auto ProfileCall(ProfileMap& profile, const std::string& label, Fn&& fn)
    -> decltype(fn()) {
  const auto start = Clock::now();
  if constexpr (std::is_void_v<decltype(fn())>) {
    fn();
    const auto elapsed = Clock::now() - start;
    auto& entry = profile[label];
    entry.total += elapsed;
    ++entry.calls;
  } else {
    auto result = fn();
    const auto elapsed = Clock::now() - start;
    auto& entry = profile[label];
    entry.total += elapsed;
    ++entry.calls;
    return result;
  }
}

void PrintProfileSummary(const ProfileMap& profile) {
  std::vector<std::pair<std::string, ProfileEntry>> entries(profile.begin(),
                                                            profile.end());
  std::sort(entries.begin(), entries.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.second.total > rhs.second.total;
            });

  double profiled_total_seconds = 0.0;
  for (const auto& entry : entries) {
    profiled_total_seconds += entry.second.total.count();
  }

  std::cout << "\nHotspots (timed model call sites):\n";
  for (const auto& entry : entries) {
    const double total_seconds = entry.second.total.count();
    const double average_seconds =
        total_seconds / static_cast<double>(entry.second.calls);
    const double percent =
        profiled_total_seconds > 0.0
            ? (100.0 * total_seconds) / profiled_total_seconds
            : 0.0;

    std::cout << "  " << std::left << std::setw(28) << entry.first
              << " total=" << std::setw(11) << total_seconds << "s"
              << " avg=" << std::setw(11) << average_seconds << "s"
              << " share=" << percent << "%\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  using deeptiny::FormatShape;

  try {
    constexpr char kInputText[] = "hello world!";
    constexpr uint64_t kEmbeddingDim = 4;
    constexpr uint64_t kHiddenDim = 8;
    constexpr uint64_t kOutputDim = 4;
    constexpr uint64_t kWarmupIterations = 100;

    const uint64_t iterations = ParseIterations(argc, argv);
    const std::vector<std::string> tokens = Tokenize(kInputText);
    if (tokens.size() != 2) {
      throw std::runtime_error(
          "Benchmark expects exactly two tokens from \"hello world!\".");
    }

    const std::unordered_map<std::string, int64_t> vocabulary{{"hello", 0},
                                                              {"world!", 1}};
    const std::vector<int64_t> token_ids = EncodeTokens(tokens, vocabulary);
    const deeptiny::Shape token_shape{1,
                                      static_cast<uint64_t>(token_ids.size())};

    deeptiny::nn::Embedding embedding(vocabulary.size(), kEmbeddingDim);
    deeptiny::nn::GatedReLU model(kEmbeddingDim, kHiddenDim, kOutputDim);

    for (uint64_t i = 0; i < kWarmupIterations; ++i) {
      auto embedded = embedding(token_ids, token_shape);
      auto output = model(embedded);
      auto loss = deeptiny::functional::Reduce(output, {0, 1, 2});
      (void)loss;
    }

    ProfileMap profile;
    deeptiny::Shape output_shape;
    deeptiny::Shape loss_shape;
    uint64_t sink = 0;

    const auto start = Clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
      auto embedded = ProfileCall(profile, "Embedding::operator()", [&] {
        return embedding(token_ids, token_shape);
      });
      auto output = ProfileCall(profile, "GatedReLU::operator()",
                                [&] { return model(embedded); });
      auto loss = ProfileCall(profile, "functional::Reduce()", [&] {
        return deeptiny::functional::Reduce(output, {0, 1, 2});
      });

      sink += loss.numel();
      if (i + 1 == iterations) {
        output_shape = output.shape();
        loss_shape = loss.shape();
      }
    }
    const auto elapsed = std::chrono::duration<double>(Clock::now() - start);
    const double total_seconds = elapsed.count();
    const double seconds_per_iteration =
        total_seconds / static_cast<double>(iterations);

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "transfomer-demo benchmark\n";
    std::cout << "input text: \"" << kInputText << "\"\n";
    std::cout << "tokens: [" << tokens[0] << ", " << tokens[1] << "]\n";
    std::cout << "iterations: " << iterations
              << " (warmup: " << kWarmupIterations << ")\n";
    std::cout << "output shape: " << FormatShape(output_shape) << "\n";
    std::cout << "loss shape: " << FormatShape(loss_shape) << "\n";
    std::cout << "total seconds: " << total_seconds << "\n";
    std::cout << "seconds/iteration: " << seconds_per_iteration << "\n";
    std::cout << "sink: " << sink << "\n";
    PrintProfileSummary(profile);

#ifdef TRANSFOMER_DEMO_GPROF_ENABLED
    std::cout << "\nCompiler-level profiling is enabled (-pg).\n";
    std::cout << "Inspect deeper function hotspots with:\n";
    std::cout << "  gprof ./build/transfomer_benchmark gmon.out | head -n 80\n";
#else
    std::cout << "\nCompiler-level profiling (-pg) is disabled.\n";
#endif

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "benchmark failed: " << ex.what() << "\n";
    return 1;
  }
}
