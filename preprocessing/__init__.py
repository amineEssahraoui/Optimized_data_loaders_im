"""
Preprocessing package for the VLM post-training pipeline.

Converts raw HuggingFace datasets into a canonical schema, applies image
and text preprocessing, and writes binary shards that the C++ runtime
loader can read without any Python dependency.
"""
