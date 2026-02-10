#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "deeptiny/autograd.h"
#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"

namespace {

uint64_t ParseMaxNewTokensArg(const std::string& value) {
  size_t parsed = 0;
  const uint64_t parsed_value = std::stoull(value, &parsed);
  if (parsed != value.size()) {
    throw std::runtime_error("Invalid max_new_tokens value: " + value);
  }
  if (parsed_value == 0) {
    throw std::runtime_error("max_new_tokens must be non-zero");
  }
  return parsed_value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    uint64_t max_new_tokens = 64;
    if (argc > 2) {
      throw std::runtime_error(
          "usage: transfomer_generation_benchmark [max_new_tokens]");
    }
    if (argc == 2) {
      max_new_tokens = ParseMaxNewTokensArg(argv[1]);
    }

    const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
    auto model =
        demo::smollm2::CreateSmolLM2_135M_InstructTransformerUninitialized(
            config);

    transfomer_demo::Transformer::GenerationOptions generation_options;
    generation_options.max_new_tokens = max_new_tokens;
    generation_options.temperature = 0.0f;
    generation_options.eos_token_id = std::nullopt;

    const std::vector<int64_t> prompt_tokens = {
        static_cast<int64_t>(config.bos_token_id)};
    std::mt19937 rng(1234U);

    deeptiny::NoGrad guard;
    const auto start = std::chrono::steady_clock::now();
    const std::vector<int64_t> generated =
        model->Generate(prompt_tokens, generation_options, &rng);
    const auto end = std::chrono::steady_clock::now();

    if (generated.size() != generation_options.max_new_tokens) {
      throw std::runtime_error("Benchmark generated token count mismatch.");
    }

    const std::chrono::duration<double, std::milli> elapsed = end - start;
    const double tokens_per_second =
        static_cast<double>(generated.size()) / (elapsed.count() / 1000.0);

    std::cout << "prompt_tokens: " << prompt_tokens.size() << "\n";
    std::cout << "generated_tokens: " << generated.size() << "\n";
    std::cout << "elapsed_ms: " << elapsed.count() << "\n";
    std::cout << "tokens_per_second: " << tokens_per_second << "\n";
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }

  return 0;
}
