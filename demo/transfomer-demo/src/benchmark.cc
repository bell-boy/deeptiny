#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_GPERFTOOLS
#include <gperftools/profiler.h>
#endif
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

namespace {

void PrintUsage() {
  std::cout << "usage:\n";
  std::cout << "  ./build/transfomer_demo_benchmark [model_dir]\n";
}

std::string ReadAllBytes(const std::filesystem::path& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (!fs.good()) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }
  fs.seekg(0, std::ios::end);
  const size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  std::string bytes(size, '\0');
  fs.read(bytes.data(), static_cast<std::streamsize>(size));
  if (!fs.good()) {
    throw std::runtime_error("Failed to read file: " + path.string());
  }
  return bytes;
}

std::filesystem::path ResolveModelDirWithWeights(int argc, char** argv) {
  std::filesystem::path model_dir =
      demo::smollm2::ModelFilesDir(std::filesystem::current_path());
  if (argc == 2) {
    model_dir = argv[1];
  }

  const std::filesystem::path safetensors_path =
      model_dir / "model.safetensors";
  if (!std::filesystem::exists(safetensors_path) ||
      !std::filesystem::is_regular_file(safetensors_path)) {
    const std::filesystem::path downloaded_safetensors =
        demo::smollm2::DownloadSmolLM2_135M_InstructSafetensors(
            std::filesystem::current_path());
    model_dir = downloaded_safetensors.parent_path();
  }

  return model_dir;
}

#ifdef TRANSFOMER_DEMO_HAS_GPERFTOOLS
std::string ResolveProfilePath() {
  const char* profile_path = std::getenv("TRANSFOMER_DEMO_PROFILE_PATH");
  if (profile_path != nullptr && profile_path[0] != '\0') {
    return std::string(profile_path);
  }
  return "transfomer_demo_benchmark.prof";
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 2) {
      PrintUsage();
      return 1;
    }

    if (argc == 2 && std::string(argv[1]) == "--help") {
      PrintUsage();
      return 0;
    }

    const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
    const std::filesystem::path model_dir =
        ResolveModelDirWithWeights(argc, argv);
    auto model = demo::smollm2::CreateSmolLM2_135M_InstructTransformer(
        model_dir, config);

    std::filesystem::path tokenizer_path = model_dir / "tokenizer.json";
    if (!std::filesystem::exists(tokenizer_path) ||
        !std::filesystem::is_regular_file(tokenizer_path)) {
      tokenizer_path =
          demo::smollm2::DownloadSmolLM2_135M_InstructTokenizerJson(
              std::filesystem::current_path());
    }

#ifndef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
    (void)model;
    (void)tokenizer_path;
    throw std::runtime_error(
        "This benchmark requires tokenizers-cpp. Configure with "
        "TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP=ON.");
#else
    const std::string tokenizer_blob = ReadAllBytes(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);
    if (!tokenizer) {
      throw std::runtime_error("Failed to initialize tokenizer from " +
                               tokenizer_path.string());
    }

    const std::vector<int32_t> token_ids = tokenizer->Encode("hello world");
    if (token_ids.empty()) {
      throw std::runtime_error(
          "Tokenizer returned zero tokens for 'hello world'");
    }
    std::vector<int64_t> tokens(token_ids.begin(), token_ids.end());

#ifdef TRANSFOMER_DEMO_HAS_GPERFTOOLS
    bool started_here = false;
    std::string profile_path;
    if (ProfilingIsEnabledForAllThreads() != 0) {
      const char* cpuprofile_path = std::getenv("CPUPROFILE");
      if (cpuprofile_path != nullptr && cpuprofile_path[0] != '\0') {
        profile_path = cpuprofile_path;
      } else {
        profile_path = "<already enabled>";
      }
    } else {
      profile_path = ResolveProfilePath();
      if (ProfilerStart(profile_path.c_str()) == 0) {
        throw std::runtime_error("Failed to start gperftools profiler at " +
                                 profile_path);
      }
      started_here = true;
    }

    const deeptiny::Tensor output = (*model)({tokens});
    if (started_here) {
      ProfilerStop();
    } else {
      ProfilerFlush();
    }

    std::cout << "model_dir: " << model_dir << "\n";
    std::cout << "tokenizer: " << tokenizer_path << "\n";
    std::cout << "profile_output: " << profile_path << "\n";
    std::cout << "input_text: hello world\n";
    std::cout << "input_token_count: " << token_ids.size() << "\n";
    std::cout << "output_numel: " << output.numel() << "\n";
#else
    (void)tokens;
    throw std::runtime_error(
        "This benchmark requires gperftools. Configure with "
        "TRANSFOMER_DEMO_ENABLE_GPERFTOOLS=ON.");
#endif
#endif
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
  return 0;
}
