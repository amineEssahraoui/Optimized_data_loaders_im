"""
Preprocessing package -- Python-side ingestion, normalization, tokenization,
and shard writing for the VLM post-training pipeline.

This package converts raw HuggingFace datasets into a canonical schema,
applies image and text preprocessing, and writes binary shards that the
C++ runtime loader can read without any Python dependency.
"""
