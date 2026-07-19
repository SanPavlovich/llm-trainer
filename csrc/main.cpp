#include <torch/extension.h>
#include <vector>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rmsnorm_backward(torch::Tensor dy, torch::Tensor x,
                                            torch::Tensor weight, torch::Tensor rrms);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
    m.def("rmsnorm_forward", torch::wrap_pybind_function(rmsnorm_forward), "rmsnorm_forward");
    m.def("rmsnorm_backward", torch::wrap_pybind_function(rmsnorm_backward), "rmsnorm_backward");
}