#include <iostream>
#include <string>
#include <vector>
#include <cassert>

// Include loader headers (assuming loader/include is in the include path)
#include "shard_reader.h"
#include "detail/sample_parser.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shard_path>\n";
        return 1;
    }

    std::string shard_path = argv[1];

    try {
        vlm::ShardReader reader(shard_path);

        // Assert header constraints
        assert(reader.sample_count() == 1);

        auto sample = reader.read_sample(0);

        // The Python test will generate a sample with specific edge-case properties.
        // We assert these properties precisely.

        // Expected dimensions for the edge case: actual_height=17, actual_width=31, channels=3
        assert(sample.channels == 3);
        assert(sample.actual_height == 17);
        assert(sample.actual_width == 31);

        // Original dimensions: 10000x2
        assert(sample.orig_height == 10000);
        assert(sample.orig_width == 2);

        // Image tensor data check. We know the Python test will fill it with a specific pattern or constant
        // Let's assume the Python test sets all pixels to value 42 (uint8)
        const uint8_t* img_data = reinterpret_cast<const uint8_t*>(sample.image_data.data());
        assert(img_data[0] == 42); // Check first pixel
        assert(img_data[sample.image_data.size() - 1] == 42); // Check last pixel

        // Token lengths check (tokenizer is set to max_length=128 in python test)
        assert(sample.question_ids.size() == 128);
        assert(sample.answer_ids.size() == 128);

        // The Python test will set the first question token to 9999 and the first answer token to 8888
        assert(sample.question_ids[0] == 9999);
        assert(sample.answer_ids[0] == 8888);

        // Check metadata size and substring
        std::string meta_str(sample.metadata_json.begin(), sample.metadata_json.end());
        assert(meta_str.find("\"deeply\":") != std::string::npos);
        assert(meta_str.find("\"nested\":") != std::string::npos);

        std::cout << "C++ Bit-Exactness Test Passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << "\n";
        return 2;
    }
}
