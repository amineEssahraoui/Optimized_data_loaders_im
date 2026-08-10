# pybind11 bindings for the VLM loader library.
# This is a development/testing bridge, not a production dependency.

pybind11_add_module(vlm_loader_py
    bind_loader.cpp
)

target_link_libraries(vlm_loader_py PRIVATE vlm_loader)

# Install the Python module alongside the project
install(TARGETS vlm_loader_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
