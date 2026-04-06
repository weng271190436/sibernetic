cmake_minimum_required(VERSION 3.20)
project(Sibernetic VERSION 0.9.8 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SIBERNETIC_USE_GRAPHICS "Enable OpenGL graphics" ON)
option(SIBERNETIC_USE_PYTHON "Enable Python integration" ON)
option(SIBERNETIC_USE_METAL "Use Metal instead of OpenCL (Apple only)" OFF)
option(SIBERNETIC_BUILD_TESTS "Build tests" OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform detection
if(APPLE)
    message(STATUS "Building for macOS/Apple Silicon")
    set(SIBERNETIC_PLATFORM "APPLE")
    add_definitions(-D__APPLE__)
    
    # Use Homebrew paths on Apple Silicon
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(HOMEBREW_PREFIX "/opt/homebrew")
    else()
        set(HOMEBREW_PREFIX "/usr/local")
    endif()
    
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
    include_directories("${HOMEBREW_PREFIX}/include")
    link_directories("${HOMEBREW_PREFIX}/lib")
elseif(UNIX)
    message(STATUS "Building for Linux")
    set(SIBERNETIC_PLATFORM "LINUX")
    add_definitions(-D__linux__)
    # Metal not available on Linux
    set(SIBERNETIC_USE_METAL OFF)
endif()

# Source files (common)
set(SIBERNETIC_SOURCES
    src/main.cpp
    src/owHelper.cpp
    src/owPhysicsFluidSimulator.cpp
    src/owWorldSimulation.cpp
    src/owNeuronSimulator.cpp
    src/owSignalSimulator.cpp
    src/owConfigProperty.cpp
    src/owVtkExport.cpp
    src/test/owPhysicTest.cpp
)

# Add solver based on backend choice
if(SIBERNETIC_USE_METAL AND APPLE)
    message(STATUS "Using Metal compute backend")
    list(APPEND SIBERNETIC_SOURCES src/metal/owMetalSolver.cpp)
    add_definitions(-DUSE_METAL)
else()
    message(STATUS "Using OpenCL compute backend")
    list(APPEND SIBERNETIC_SOURCES src/owOpenCLSolver.cpp)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/inc)

# Metal setup (Apple only)
if(SIBERNETIC_USE_METAL AND APPLE)
    # Find metal-cpp headers
    if(EXISTS "${CMAKE_SOURCE_DIR}/metal-cpp")
        include_directories("${CMAKE_SOURCE_DIR}/metal-cpp")
    elseif(EXISTS "/usr/local/include/metal-cpp")
        include_directories("/usr/local/include/metal-cpp")
    elseif(EXISTS "${HOMEBREW_PREFIX}/include/metal-cpp")
        include_directories("${HOMEBREW_PREFIX}/include/metal-cpp")
    else()
        message(FATAL_ERROR "metal-cpp headers not found. Download from https://developer.apple.com/metal/cpp/ and place in ${CMAKE_SOURCE_DIR}/metal-cpp/")
    endif()
    
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    set(COMPUTE_LIBRARY ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
    message(STATUS "Found Metal framework: ${METAL_LIBRARY}")
else()
    # Find OpenCL
    if(APPLE)
        find_library(OPENCL_LIBRARY OpenCL REQUIRED)
        set(OPENCL_INCLUDE_DIR "")
        message(STATUS "Found OpenCL framework: ${OPENCL_LIBRARY}")
    else()
        find_package(OpenCL REQUIRED)
        set(OPENCL_LIBRARY OpenCL::OpenCL)
    endif()
    set(COMPUTE_LIBRARY ${OPENCL_LIBRARY})
endif()

# Find Python 3
if(SIBERNETIC_USE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python ${Python3_VERSION}: ${Python3_EXECUTABLE}")
    message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python libs: ${Python3_LIBRARIES}")
endif()

# Find OpenGL (optional for graphics)
if(SIBERNETIC_USE_GRAPHICS)
    if(APPLE)
        find_library(OPENGL_LIBRARY OpenGL REQUIRED)
        find_library(GLUT_LIBRARY GLUT REQUIRED)
        set(GRAPHICS_LIBRARIES ${OPENGL_LIBRARY} ${GLUT_LIBRARY})
        message(STATUS "Found OpenGL/GLUT frameworks")
    else()
        find_package(OpenGL REQUIRED)
        find_package(GLUT REQUIRED)
        find_package(GLEW REQUIRED)
        set(GRAPHICS_LIBRARIES OpenGL::GL OpenGL::GLU GLUT::GLUT GLEW::GLEW)
    endif()
else()
    add_definitions(-DNO_GRAPHICS)
endif()

# Create executable
add_executable(Sibernetic ${SIBERNETIC_SOURCES})

# Link libraries
target_link_libraries(Sibernetic PRIVATE ${COMPUTE_LIBRARY})

if(SIBERNETIC_USE_PYTHON)
    target_include_directories(Sibernetic PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(Sibernetic PRIVATE ${Python3_LIBRARIES})
    target_compile_definitions(Sibernetic PRIVATE USE_PYTHON)
endif()

if(SIBERNETIC_USE_GRAPHICS)
    target_link_libraries(Sibernetic PRIVATE ${GRAPHICS_LIBRARIES})
else()
    target_compile_definitions(Sibernetic PRIVATE NO_GRAPHICS)
endif()

# macOS-specific settings
if(APPLE)
    target_link_libraries(Sibernetic PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreFoundation"
    )
    
    set_target_properties(Sibernetic PROPERTIES
        MACOSX_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@executable_path/../lib"
    )
    
    # Compile Metal shaders if using Metal
    if(SIBERNETIC_USE_METAL)
        # Compile .metal to .metallib
        add_custom_command(
            OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sphFluid.metallib
            COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/src/metal/sphFluid.metal -o ${CMAKE_BINARY_DIR}/sphFluid.air
            COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/sphFluid.air -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sphFluid.metallib
            DEPENDS ${CMAKE_SOURCE_DIR}/src/metal/sphFluid.metal
            COMMENT "Compiling Metal shaders"
        )
        add_custom_target(MetalShaders ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sphFluid.metallib)
        add_dependencies(Sibernetic MetalShaders)
        
        # Also copy source for runtime compilation fallback
        file(COPY ${CMAKE_SOURCE_DIR}/src/metal DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../src/)
    endif()
endif()

# Linux-specific settings  
if(UNIX AND NOT APPLE)
    target_link_libraries(Sibernetic PRIVATE rt)
endif()

# Copy OpenCL kernels to build directory (for OpenCL backend)
if(NOT SIBERNETIC_USE_METAL)
    file(GLOB CL_KERNELS "src/*.cl")
    foreach(kernel ${CL_KERNELS})
        get_filename_component(kernel_name ${kernel} NAME)
        configure_file(${kernel} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${kernel_name} COPYONLY)
    endforeach()
endif()

# Copy configuration files
file(COPY ${CMAKE_SOURCE_DIR}/configuration DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/..)

# Copy Python files needed at runtime
file(COPY ${CMAKE_SOURCE_DIR}/main_sim.py DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/..)

# Installation
install(TARGETS Sibernetic RUNTIME DESTINATION bin)
install(DIRECTORY configuration/ DESTINATION share/sibernetic/configuration)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Sibernetic Configuration ===")
message(STATUS "Platform:       ${SIBERNETIC_PLATFORM}")
message(STATUS "Compute:        ${SIBERNETIC_USE_METAL}")
if(SIBERNETIC_USE_METAL)
    message(STATUS "                Metal (GPU)")
else()
    message(STATUS "                OpenCL")
endif()
message(STATUS "Graphics:       ${SIBERNETIC_USE_GRAPHICS}")
message(STATUS "Python:         ${SIBERNETIC_USE_PYTHON}")
message(STATUS "C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "================================")
