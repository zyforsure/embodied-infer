cmake_minimum_required(VERSION 3.20)

project(embodied_infer VERSION 0.1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(EMBODIED_INFER_BUILD_EXAMPLES "Build examples" ON)
option(EMBODIED_INFER_BUILD_TESTS "Build tests" ON)

add_library(embodied_infer
    src/action_buffer.cpp
    src/engine.cpp
    src/mock_backend.cpp
    src/processors.cpp
    src/scheduler.cpp
    src/operator_scheduler.cpp
    src/hardware_output.cpp
)
add_library(embodied::infer ALIAS embodied_infer)
set_target_properties(embodied_infer PROPERTIES EXPORT_NAME infer)

target_compile_features(embodied_infer PUBLIC cxx_std_20)
target_include_directories(embodied_infer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(embodied_infer PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(embodied_infer PRIVATE /W4 /permissive-)
else()
    target_compile_options(embodied_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(EMBODIED_INFER_BUILD_EXAMPLES)
    add_executable(embodied-infer-demo examples/demo.cpp)
    target_link_libraries(embodied-infer-demo PRIVATE embodied::infer)
endif()

if(EMBODIED_INFER_BUILD_TESTS)
    include(CTest)
    add_executable(embodied-infer-tests tests/runtime_tests.cpp)
    target_link_libraries(embodied-infer-tests PRIVATE embodied::infer)
    add_test(NAME embodied-infer-tests COMMAND embodied-infer-tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS embodied_infer
    EXPORT embodied_infer_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT embodied_infer_targets
    FILE embodied_infer_targets.cmake
    NAMESPACE embodied::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/embodied_infer
)
configure_package_config_file(
    cmake/embodied_inferConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/embodied_inferConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/embodied_infer
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/embodied_inferConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/embodied_inferConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/embodied_inferConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/embodied_infer
)
