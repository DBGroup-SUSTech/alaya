#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #include "/home/longxiang/alaya/include/alaya/index/pybind11_test.h"
// #include "../include/alaya/index/pybind11_test.h"

#include "../include/alaya/index/bucket/ivf.h"
#include "../include/alaya/searcher/ivf_searcher.h"
#include "../include/alaya/utils/metric_type.h"

namespace py = pybind11;

inline void get_input_array_shapes(const py::buffer_info &buffer, size_t *rows,
                                   size_t *features) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "Input vector data wrong shape. Number of dimensions %d. Data "
             "must be a 1D or 2D array.",
             buffer.ndim);
  }
  if (buffer.ndim == 2) {
    *rows = buffer.shape[0];
    *features = buffer.shape[1];
  } else {
    *rows = 1;
    *features = buffer.shape[0];
  }
}


struct Ivf {
  std::unique_ptr<alaya::IVF<float>> ivf_ = nullptr;
  std::unique_ptr<alaya::SearcherBase<float>> searcher_ = nullptr;


  Ivf(const std::string& kMetric, int dim, int bukcet_num) :
    ivf_(std::make_unique<alaya::IVF<float>>(dim, kMetric, bukcet_num)) {
    // auto metric = alaya::kMetricMap[kMetric];
    if (kMetric=="L2") {
      searcher_ = std::make_unique<alaya::IvfSearcher<alaya::MetricType::L2>>(ivf_.get());
    } else if (kMetric == "IP") {
      searcher_ = std::make_unique<alaya::IvfSearcher<alaya::MetricType::IP>>(ivf_.get());
    } else if (kMetric == "COS") {
      searcher_ = std::make_unique<alaya::IvfSearcher<alaya::MetricType::COS>>(ivf_.get());
    }
  }

  void Build(py::object input) {
    py::array_t<float, py::array::c_style | py::array::forcecast> data(input);
    auto buffer = data.request();
    size_t rows, cols;
    get_input_array_shapes(buffer, &rows, &cols);
    float *vector_data = (float*)data.data(0);
    ivf_->BuildIndex(rows, vector_data);
  }

  auto Search(py::array_t<float> query, int64_t k) {
    float *query_data = (float*)query.data(0);
    float* distance = new float[k];
    int64_t* result_id = new int64_t[k];
    searcher_->Search(query_data, k, distance, result_id);
    return std::make_tuple(distance, result_id);
  }
};

PYBIND11_MODULE(alaya, m) {
  // py::module m("alaya");
  m.doc() = "pybind11 example plugin";  // optional module docstring

  // m.def("add", &add, "A function which adds two numbers");

  // py::class_<alaya::PybindTest>(m, "PyBindTest")
  //     .def(py::init<int>())
  //     .def("print", &alaya::PybindTest::print);

  // py::enum_<alaya::MetricType>(m, "MetricType")
  //     .value("L2", alaya::MetricType::L2)
  //     .value("IP", alaya::MetricType::IP)
  //     .value("COS", alaya::MetricType::COS)

  py::class_<Ivf>(m, "Ivf")
      .def(py::init<const std::string &, int, int>(),
           py::arg("kMetric"), py::arg("dim"), py::arg("bucket_num"))
      .def("build", &Ivf::Build, py::arg("input"))
      .def("search", &Ivf::Search, py::arg("query"), py::arg("k"));
      // .def("search", [](const py::array_t<float>& query, int64_t k) {
      //   float *query_data = (float*)query.data(0);
      //   float* distance = new float[k];
      //   int64_t* result_id = new int64_t[k];
      //   Ivf->searcher_->Search(query_data, k, distance, result_id);
      //   return std::make_tuple(distance, result_id);
      // }, py::arg("query"), py::arg("k"));
}