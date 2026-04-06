cmake_minimum_required(VERSION 3.20)
project(Sibernetic VERSION 0.9.9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(SIBERNETIC_USE_GRAPHICS "Enable OpenGL graphics" ON)
option(SIBERNETIC_USE_PYTHON "Enable Python integration" ON)
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
endif()

# Source files
set(SIBERNETIC_SOURCES
    src/main.cpp
    src/owHelper.cpp
    src/owOpenCLSolver.cpp
    src/owPhysicsFluidSimulator.cpp
    src/owWorldSimulation.cpp
    src/owNeuronSimulator.cpp
    src/owSignalSimulator.cpp
    src/owConfigProperty.cpp
    src/owVtkExport.cpp
    src/test/owPhysicTest.cpp
)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/inc)

# Find OpenCL
if(APPLE)
    # macOS: OpenCL is a framework
    find_library(OPENCL_LIBRARY OpenCL REQUIRED)
    set(OPENCL_INCLUDE_DIR "")
    message(STATUS "Found OpenCL framework: ${OPENCL_LIBRARY}")
else()
    find_package(OpenCL REQUIRED)
    set(OPENCL_LIBRARY OpenCL::OpenCL)
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
target_link_libraries(Sibernetic PRIVATE
    ${OPENCL_LIBRARY}
)

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
    # Link against system frameworks
    target_link_libraries(Sibernetic PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreFoundation"
    )
    
    # Set RPATH for finding libraries
    set_target_properties(Sibernetic PROPERTIES
        MACOSX_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@executable_path/../lib"
    )
endif()

# Linux-specific settings  
if(UNIX AND NOT APPLE)
    target_link_libraries(Sibernetic PRIVATE rt)
endif()

# Copy OpenCL kernels to build directory
file(GLOB CL_KERNELS "src/*.cl")
foreach(kernel ${CL_KERNELS})
    get_filename_component(kernel_name ${kernel} NAME)
    configure_file(${kernel} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${kernel_name} COPYONLY)
endforeach()

# Copy configuration files
file(COPY ${CMAKE_SOURCE_DIR}/configuration DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/..)

# Installation
install(TARGETS Sibernetic RUNTIME DESTINATION bin)
install(FILES ${CL_KERNELS} DESTINATION share/sibernetic/kernels)
install(DIRECTORY configuration/ DESTINATION share/sibernetic/configuration)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Sibernetic Configuration ===")
message(STATUS "Platform: ${SIBERNETIC_PLATFORM}")
message(STATUS "Graphics: ${SIBERNETIC_USE_GRAPHICS}")
message(STATUS "Python:   ${SIBERNETIC_USE_PYTHON}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "================================")
