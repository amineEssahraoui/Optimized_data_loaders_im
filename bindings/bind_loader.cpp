/**
 * @file bind_loader.cpp
 * @brief pybind11 bindings for the C++ shard loader (v1, v2, and async).
 *
 * Exposes the ShardReader, Sample, Batch, distribution utilities,
 * v2 features (NormalizationParams, AspectRatioBucketer), and the
 * Phase 3 AsyncShardLoader with iterator protocol and to_torch().
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
#include "async_loader.h"
#include "gpu_transfer.h"

namespace py = pybind11;

#if defined(_MSC_VER)
using ssize_t = pybind11::ssize_t;
#endif

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

/**
 * Convert a Batch to a Python dict of numpy arrays.
 * Used by both the Batch binding and AsyncShardLoader.to_torch().
 */
static py::dict batch_to_dict(const vlm::Batch& b) {
    py::dict d;

    d["image"] = vec_to_numpy_float(b.image_data, {
        b.batch_size, b.image_channels, b.image_height, b.image_width
    });
    d["question_ids"] = vec_to_numpy_int32(
        b.question_ids, {b.batch_size, b.token_length});
    d["question_mask"] = vec_to_numpy_int32(
        b.question_mask, {b.batch_size, b.token_length});
    d["answer_ids"] = vec_to_numpy_int32(
        b.answer_ids, {b.batch_size, b.token_length});
    d["answer_mask"] = vec_to_numpy_int32(
        b.answer_mask, {b.batch_size, b.token_length});

    // Metadata as list of strings
    py::list meta;
    for (const auto& m : b.metadata_json)
        meta.append(py::str(m));
    d["metadata"] = meta;

    // V2 padding mask (optional)
    if (!b.padding_mask.empty()) {
        d["padding_mask"] = vec_to_numpy_float(b.padding_mask, {
            b.batch_size, static_cast<ssize_t>(1),
            b.image_height, b.image_width
        });
    }

    return d;
}

/**
 * Convert a Batch dict to PyTorch tensors.
 *
 * If torch is available, converts numpy arrays to torch.Tensor.
 * If CUDA is available and a device string like "cuda:0" is passed,
 * tensors are placed on that device.
 *
 * Falls back to numpy dict if torch is not importable.
 */
static py::dict batch_to_torch(
    const vlm::Batch& batch,
    const std::string& device = "cpu")
{
    py::dict numpy_dict = batch_to_dict(batch);

    try {
        py::module_ torch = py::module_::import("torch");
        py::object torch_device = torch.attr("device")(py::str(device));

        py::dict result;

        // Convert each numpy array to a torch tensor
        for (auto item : numpy_dict) {
            py::str key = py::reinterpret_borrow<py::str>(item.first);
            py::object val = py::reinterpret_borrow<py::object>(item.second);

            if (py::isinstance<py::array>(val)) {
                // numpy array → torch tensor → device
                py::object tensor = torch.attr("from_numpy")(val);
                if (device != "cpu") {
                    tensor = tensor.attr("to")(torch_device);
                }
                result[key] = tensor;
            } else {
                // metadata strings pass through
                result[key] = val;
            }
        }

        return result;
    } catch (const py::error_already_set&) {
        // torch not available, return numpy dict
        return numpy_dict;
    }
}


PYBIND11_MODULE(vlm_loader_py, m) {
    m.doc() = "pybind11 bindings for the VLM shard loader (v1 + v2 + async)";

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
        .def_readonly("flags", &vlm::ShardFileHeader::flags)
        .def("is_uint8", &vlm::ShardFileHeader::is_uint8)
        .def("has_per_sample_dims", &vlm::ShardFileHeader::has_per_sample_dims)
        .def("is_lz4_compressed", &vlm::ShardFileHeader::is_lz4_compressed)
        .def("is_zstd_compressed", &vlm::ShardFileHeader::is_zstd_compressed)
    ;

    // -----------------------------------------------------------------------
    // NormalizationParams
    // -----------------------------------------------------------------------
    py::class_<vlm::NormalizationParams>(m, "NormalizationParams")
        .def(py::init<>())
        .def(py::init([](py::list mean, py::list std) {
            vlm::NormalizationParams p;
            for (int i = 0; i < 3 && i < static_cast<int>(py::len(mean)); ++i)
                p.mean[i] = mean[i].cast<float>();
            for (int i = 0; i < 3 && i < static_cast<int>(py::len(std)); ++i)
                p.std[i] = std[i].cast<float>();
            return p;
        }), py::arg("mean"), py::arg("std"),
        "Construct with [mean_r, mean_g, mean_b] and [std_r, std_g, std_b].")
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
        // V2 per-sample dimensions
        .def_readonly("orig_h", &vlm::Sample::orig_h)
        .def_readonly("orig_w", &vlm::Sample::orig_w)
        .def_readonly("actual_h", &vlm::Sample::actual_h)
        .def_readonly("actual_w", &vlm::Sample::actual_w)
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
        // V2 dynamic padding fields
        .def_property_readonly("padding_mask", [](const vlm::Batch& b) -> py::object {
            if (b.padding_mask.empty()) return py::none();
            return vec_to_numpy_float(b.padding_mask, {
                b.batch_size, static_cast<ssize_t>(1),
                b.image_height, b.image_width
            });
        })
        .def_property_readonly("actual_heights", [](const vlm::Batch& b) -> py::object {
            if (b.actual_heights.empty()) return py::none();
            return vec_to_numpy_int32(b.actual_heights,
                {static_cast<ssize_t>(b.actual_heights.size())});
        })
        .def_property_readonly("actual_widths", [](const vlm::Batch& b) -> py::object {
            if (b.actual_widths.empty()) return py::none();
            return vec_to_numpy_int32(b.actual_widths,
                {static_cast<ssize_t>(b.actual_widths.size())});
        })
        // Batch → dict conversion
        .def("to_dict", [](const vlm::Batch& b) { return batch_to_dict(b); },
             "Convert batch to a dict of numpy arrays.")
        .def("to_torch", [](const vlm::Batch& b, const std::string& device) {
             return batch_to_torch(b, device);
        }, py::arg("device") = "cpu",
        "Convert batch to a dict of torch tensors on the specified device.")
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
        .def("set_normalization", &vlm::ShardReader::set_normalization,
             py::arg("params"),
             "Set normalization mean/std for deferred uint8->float32 conversion.")
        .def("read_sample", &vlm::ShardReader::read_sample, py::arg("index"))
        .def("read_batch", &vlm::ShardReader::read_batch, py::arg("indices"))
        .def("read_all", &vlm::ShardReader::read_all)
    ;

    // -----------------------------------------------------------------------
    // AspectRatioBucketer
    // -----------------------------------------------------------------------
    py::class_<vlm::AspectRatioBucketer>(m, "AspectRatioBucketer")
        .def(py::init<int>(), py::arg("num_buckets") = 6)
        .def("add_sample", &vlm::AspectRatioBucketer::add_sample,
             py::arg("index"), py::arg("width"), py::arg("height"))
        .def("get_batches", &vlm::AspectRatioBucketer::get_batches,
             py::arg("batch_size"),
             "Get batch-sized groups of indices from same-AR buckets.")
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

    // ===================================================================
    // Phase 3: Async Loader bindings
    // ===================================================================

    // -----------------------------------------------------------------------
    // LoaderCheckpoint
    // -----------------------------------------------------------------------
    py::class_<vlm::LoaderCheckpoint>(m, "LoaderCheckpoint")
        .def(py::init<>())
        .def_readwrite("epoch", &vlm::LoaderCheckpoint::epoch)
        .def_readwrite("seed", &vlm::LoaderCheckpoint::seed)
        .def("__repr__", [](const vlm::LoaderCheckpoint& cp) {
            return "LoaderCheckpoint(epoch=" + std::to_string(cp.epoch)
                + ", seed=" + std::to_string(cp.seed) + ")";
        })
    ;

    // -----------------------------------------------------------------------
    // AsyncLoaderConfig
    // -----------------------------------------------------------------------
    py::class_<vlm::AsyncLoaderConfig>(m, "AsyncLoaderConfig")
        .def(py::init<>())
        .def_readwrite("shard_paths", &vlm::AsyncLoaderConfig::shard_paths)
        .def_readwrite("batch_size", &vlm::AsyncLoaderConfig::batch_size)
        .def_readwrite("prefetch_depth", &vlm::AsyncLoaderConfig::prefetch_depth)
        .def_readwrite("num_workers", &vlm::AsyncLoaderConfig::num_workers)
        .def_readwrite("seed", &vlm::AsyncLoaderConfig::seed)
        .def_readwrite("shuffle_buffer_size", &vlm::AsyncLoaderConfig::shuffle_buffer_size)
        .def_readwrite("norm", &vlm::AsyncLoaderConfig::norm)
        .def_readwrite("worker_rank", &vlm::AsyncLoaderConfig::worker_rank)
        .def_readwrite("world_size", &vlm::AsyncLoaderConfig::world_size)
        .def_readwrite("partition_strategy", &vlm::AsyncLoaderConfig::partition_strategy)
        .def("__repr__", [](const vlm::AsyncLoaderConfig& c) {
            return "AsyncLoaderConfig(shards=" + std::to_string(c.shard_paths.size())
                + ", batch_size=" + std::to_string(c.batch_size)
                + ", prefetch=" + std::to_string(c.prefetch_depth)
                + ", workers=" + std::to_string(c.num_workers)
                + ", shuffle_buf=" + std::to_string(c.shuffle_buffer_size)
                + ", seed=" + std::to_string(c.seed) + ")";
        })
    ;

    // -----------------------------------------------------------------------
    // AsyncShardLoader
    // -----------------------------------------------------------------------
    py::class_<vlm::AsyncShardLoader>(m, "AsyncShardLoader")
        .def(py::init<vlm::AsyncLoaderConfig>(), py::arg("config"),
             R"doc(
             Construct an async shard loader.

             Memory-maps all shard files, starts worker threads, and begins
             prefetching batches in the background.

             Args:
                 config: AsyncLoaderConfig with shard paths and parameters.
             )doc")

        // Python iterator protocol
        .def("__iter__", [](vlm::AsyncShardLoader& self) -> vlm::AsyncShardLoader& {
            return self;
        }, py::return_value_policy::reference_internal,
        "Return self as the iterator.")

        .def("__next__", [](vlm::AsyncShardLoader& self) -> vlm::Batch {
            if (!self.has_next())
                throw py::stop_iteration();
            return self.next();
        },
        R"doc(
        Get the next prefetched batch.

        Returns:
            Batch: The next batch of samples.

        Raises:
            StopIteration: When all batches in the epoch are exhausted.
        )doc")

        // Explicit next/has_next for non-iterator usage
        .def("next", &vlm::AsyncShardLoader::next,
             "Get the next batch (blocking if prefetch queue is empty).")
        .def("has_next", &vlm::AsyncShardLoader::has_next,
             "Check if more batches remain in the current epoch.")

        // Epoch control
        .def("reset", &vlm::AsyncShardLoader::reset,
             R"doc(
             Reset for a new epoch.

             Drains the prefetch queue, increments the epoch counter,
             re-seeds the shuffle PRNG, and restarts prefetching.
             )doc")

        // Checkpointing
        .def("checkpoint", &vlm::AsyncShardLoader::checkpoint,
             "Save a lightweight checkpoint (epoch + seed).")
        .def("restore", &vlm::AsyncShardLoader::restore,
             py::arg("checkpoint"),
             "Restore from a checkpoint and reset to that epoch's state.")

        // Stats
        .def("total_samples", &vlm::AsyncShardLoader::total_samples,
             "Total samples across all shards (for this worker).")
        .def("epoch", &vlm::AsyncShardLoader::epoch,
             "Current epoch number (0-based).")

        // ── to_torch(): batch → dict of torch tensors ──────────────
        .def("to_torch", [](vlm::AsyncShardLoader& self,
                            const std::string& device) -> py::dict {
            if (!self.has_next())
                throw py::stop_iteration();
            vlm::Batch batch = self.next();
            return batch_to_torch(batch, device);
        }, py::arg("device") = "cpu",
        R"doc(
        Get the next batch as a dict of PyTorch tensors.

        Combines next() + tensor conversion in a single call.
        Falls back to numpy arrays if PyTorch is not installed.

        Args:
            device: Target device string (e.g., "cpu", "cuda:0").

        Returns:
            dict: {
                "image": Tensor [N, C, H, W],
                "question_ids": Tensor [N, T],
                "question_mask": Tensor [N, T],
                "answer_ids": Tensor [N, T],
                "answer_mask": Tensor [N, T],
                "metadata": list[str],
            }

        Raises:
            StopIteration: When all batches in the epoch are exhausted.
        )doc")

        .def("__repr__", [](const vlm::AsyncShardLoader& self) {
            return "AsyncShardLoader(total_samples="
                + std::to_string(self.total_samples())
                + ", epoch=" + std::to_string(self.epoch())
                + ", has_next=" + (self.has_next() ? "True" : "False") + ")";
        })
    ;

    // -----------------------------------------------------------------------
    // Module-level GPU info
    // -----------------------------------------------------------------------
    m.def("has_cuda", []() -> bool {
#ifdef VLM_HAS_CUDA
        return true;
#else
        return false;
#endif
    }, "Check if the loader was compiled with CUDA support.");
}
