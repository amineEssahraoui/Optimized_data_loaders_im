/**
 * @file bind_loader.cpp
 * @brief pybind11 bindings for the C++ shard loader.
 *
 * Exposes the ShardReader, Sample, Batch, and distribution utilities
 * to Python for development and testing.  This module is NOT a
 * production dependency -- it exists solely to enable Python-based
 * round-trip validation of the C++ loader against the Python writer.
 *
 * Numpy arrays are returned via pybind11's numpy integration, allowing
 * direct comparison with the Python shard_reader output.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "shard_reader.h"
#include "distributed.h"
#include "batch.h"

namespace py = pybind11;

/**
 * Convert a std::vector<float> to a numpy array with the given shape.
 * The data is copied into the numpy array (no shared ownership issues).
 */
static py::array_t<float> vec_to_numpy_float(
    const std::vector<float>& data,
    const std::vector<ssize_t>& shape
) {
    py::array_t<float> arr(shape);
    auto buf = arr.mutable_unchecked();
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(float));
    return arr;
}

/**
 * Convert a std::vector<int32_t> to a numpy array with the given shape.
 */
static py::array_t<int32_t> vec_to_numpy_int32(
    const std::vector<int32_t>& data,
    const std::vector<ssize_t>& shape
) {
    py::array_t<int32_t> arr(shape);
    std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(int32_t));
    return arr;
}


PYBIND11_MODULE(vlm_loader_py, m) {
    m.doc() = "pybind11 bindings for the VLM shard loader (dev/test only)";

    // -----------------------------------------------------------------------
    // ShardFileHeader
    // -----------------------------------------------------------------------
    py::class_<vlm::ShardFileHeader>(m, "ShardFileHeader")
        .def_readonly("version", &vlm::ShardFileHeader::version)
        .def_readonly("sample_count", &vlm::ShardFileHeader::sample_count)
        .def_readonly("offset_table_pos", &vlm::ShardFileHeader::offset_table_pos)
        .def_readonly("image_channels", &vlm::ShardFileHeader::image_channels)
        .def_readonly("image_height", &vlm::ShardFileHeader::image_height)
        .def_readonly("image_width", &vlm::ShardFileHeader::image_width)
        .def_readonly("token_length", &vlm::ShardFileHeader::token_length)
    ;

    // -----------------------------------------------------------------------
    // Sample (individual, returned as Python-friendly dict-like object)
    // -----------------------------------------------------------------------
    py::class_<vlm::Sample>(m, "Sample")
        .def_property_readonly("image_tensor", [](const vlm::Sample& s) {
            // Return as numpy array -- we need the header info for shape,
            // but Sample does not carry shape info.  Return as flat array
            // and let the caller reshape.
            return vec_to_numpy_float(s.image_tensor, {static_cast<ssize_t>(s.image_tensor.size())});
        })
        .def_property_readonly("question_ids", [](const vlm::Sample& s) {
            return vec_to_numpy_int32(s.question_ids, {static_cast<ssize_t>(s.question_ids.size())});
        })
        .def_property_readonly("question_mask", [](const vlm::Sample& s) {
            return vec_to_numpy_int32(s.question_mask, {static_cast<ssize_t>(s.question_mask.size())});
        })
        .def_property_readonly("answer_ids", [](const vlm::Sample& s) {
            return vec_to_numpy_int32(s.answer_ids, {static_cast<ssize_t>(s.answer_ids.size())});
        })
        .def_property_readonly("answer_mask", [](const vlm::Sample& s) {
            return vec_to_numpy_int32(s.answer_mask, {static_cast<ssize_t>(s.answer_mask.size())});
        })
        .def_property_readonly("metadata_json", [](const vlm::Sample& s) {
            return s.metadata_json;
        })
    ;

    // -----------------------------------------------------------------------
    // Batch
    // -----------------------------------------------------------------------
    py::class_<vlm::Batch>(m, "Batch")
        .def_readonly("batch_size", &vlm::Batch::batch_size)
        .def_readonly("image_channels", &vlm::Batch::image_channels)
        .def_readonly("image_height", &vlm::Batch::image_height)
        .def_readonly("image_width", &vlm::Batch::image_width)
        .def_readonly("token_length", &vlm::Batch::token_length)
        .def_property_readonly("image_data", [](const vlm::Batch& b) {
            return vec_to_numpy_float(b.image_data, {
                b.batch_size, b.image_channels, b.image_height, b.image_width
            });
        })
        .def_property_readonly("question_ids", [](const vlm::Batch& b) {
            return vec_to_numpy_int32(b.question_ids, {b.batch_size, b.token_length});
        })
        .def_property_readonly("question_mask", [](const vlm::Batch& b) {
            return vec_to_numpy_int32(b.question_mask, {b.batch_size, b.token_length});
        })
        .def_property_readonly("answer_ids", [](const vlm::Batch& b) {
            return vec_to_numpy_int32(b.answer_ids, {b.batch_size, b.token_length});
        })
        .def_property_readonly("answer_mask", [](const vlm::Batch& b) {
            return vec_to_numpy_int32(b.answer_mask, {b.batch_size, b.token_length});
        })
        .def_property_readonly("metadata_json", [](const vlm::Batch& b) {
            return b.metadata_json;
        })
    ;

    // -----------------------------------------------------------------------
    // ShardReader
    // -----------------------------------------------------------------------
    py::class_<vlm::ShardReader>(m, "ShardReader")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Open a shard file and parse its header.")
        .def("header", &vlm::ShardReader::header,
             py::return_value_policy::reference_internal)
        .def("sample_count", &vlm::ShardReader::sample_count)
        .def("read_sample", &vlm::ShardReader::read_sample, py::arg("index"))
        .def("read_batch", &vlm::ShardReader::read_batch, py::arg("indices"))
        .def("read_all", &vlm::ShardReader::read_all)
    ;

    // -----------------------------------------------------------------------
    // Distribution utilities
    // -----------------------------------------------------------------------
    m.def("get_worker_indices", &vlm::get_worker_indices,
          py::arg("total_samples"),
          py::arg("worker_id"),
          py::arg("num_workers"),
          py::arg("strategy") = "contiguous",
          "Compute sample indices for a specific worker.");

    m.def("verify_partition", &vlm::verify_partition,
          py::arg("total_samples"),
          py::arg("num_workers"),
          py::arg("strategy") = "contiguous",
          "Verify that a partitioning covers all samples with no overlaps.");
}
