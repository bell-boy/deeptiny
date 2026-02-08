#include <iostream>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/nn/gated_relu.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "smollm2_135m_instruct_loader.h"

int main(int argc, char** argv) {
  using deeptiny::FormatShape;

  constexpr uint64_t in_dim = 4;
  constexpr uint64_t hidden_dim = 8;
  constexpr uint64_t out_dim = 4;
  const std::vector<float> input_values{
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
      1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
  };
  auto x = deeptiny::Tensor::FromVector(input_values, {2, 3, in_dim},
                                        deeptiny::Device::CPU, true);
  deeptiny::nn::GatedReLU mlp(in_dim, hidden_dim, out_dim);

  auto y = mlp(x);
  auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
  loss.Backward();

  std::cout << "transfomer-demo GatedReLU demo\n";
  std::cout << "input shape: " << FormatShape(x.shape()) << "\n";
  std::cout << "output shape: " << FormatShape(y.shape()) << "\n";
  std::cout << "loss shape: " << FormatShape(loss.shape()) << "\n";
  std::cout << "x.grad available: " << std::boolalpha << x.grad().has_value()
            << "\n";
  std::cout << "gate_proj.weight.grad available: "
            << mlp.gate_proj().weight().grad().has_value() << "\n";

  const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
  const auto specs = demo::smollm2::BuildWeightSpecs(config);

  std::cout << "\nsmollm2-135m-instruct defaults\n";
  std::cout << "hidden_size: " << config.hidden_size << "\n";
  std::cout << "intermediate_size: " << config.intermediate_size << "\n";
  std::cout << "num_hidden_layers: " << config.num_hidden_layers << "\n";
  std::cout << "num_attention_heads: " << config.num_attention_heads << "\n";
  std::cout << "num_key_value_heads: " << config.num_key_value_heads << "\n";
  std::cout << "vocab_size: " << config.vocab_size << "\n";
  std::cout << "expected weight tensors: " << specs.size() << "\n";

  if (argc > 1) {
    const auto load_plan =
        demo::smollm2::LoadSmolLM2_135M_InstructWeights(argv[1], config);
    std::cout << "weight file: " << load_plan.weights_path << "\n";
    std::cout << "sharded checkpoint: " << std::boolalpha
              << load_plan.is_sharded_checkpoint << "\n";
  } else {
    std::cout << "pass a model directory to validate local safetensors files\n";
  }

  return 0;
}
