#! /usr/bin/env python3
"""Setup script for vcsgraph package."""

from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "vcsgraph._graph_rs", "crates/graph-py/Cargo.toml", binding=Binding.PyO3
        ),
    ]
)
