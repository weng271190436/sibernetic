#include "metal/owMetalRenderer.h"

#ifdef __APPLE__

#include "owConfigProperty.h"
#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>
#include <simd/simd.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

// Uniforms structure matching shader
struct Uniforms {
    simd_float4x4 modelViewProjection;
    simd_float3 offset;
    float scale;
    float pointSize;
    float rho0;
    uint32_t particleCount;
    uint32_t showBoundary;
};

// Matrix helpers
static simd_float4x4 matrix_perspective(float fovyRadians, float aspect, float nearZ, float farZ) {
    float ys = 1 / tanf(fovyRadians * 0.5f);
    float xs = ys / aspect;
    float zs = farZ / (nearZ - farZ);
    return (simd_float4x4){{
        { xs, 0, 0, 0 },
        { 0, ys, 0, 0 },
        { 0, 0, zs, -1 },
        { 0, 0, nearZ * zs, 0 }
    }};
}

static simd_float4x4 matrix_rotation(float radiansX, float radiansY) {
    float cx = cosf(radiansX), sx = sinf(radiansX);
    float cy = cosf(radiansY), sy = sinf(radiansY);
    
    simd_float4x4 rotX = {{
        { 1, 0, 0, 0 },
        { 0, cx, sx, 0 },
        { 0, -sx, cx, 0 },
        { 0, 0, 0, 1 }
    }};
    
    simd_float4x4 rotY = {{
        { cy, 0, -sy, 0 },
        { 0, 1, 0, 0 },
        { sy, 0, cy, 0 },
        { 0, 0, 0, 1 }
    }};
    
    return simd_mul(rotY, rotX);
}

static simd_float4x4 matrix_translation(float x, float y, float z) {
    return (simd_float4x4){{
        { 1, 0, 0, 0 },
        { 0, 1, 0, 0 },
        { 0, 0, 1, 0 },
        { x, y, z, 1 }
    }};
}

owMetalRenderer::owMetalRenderer()
    : device(nullptr), commandQueue(nullptr), library(nullptr), pipelineState(nullptr),
      positionBuffer(nullptr), densityBuffer(nullptr), uniformBuffer(nullptr),
      metalLayer(nullptr), config(nullptr), particleCount(0), maxParticles(0),
      rotationX(0.3f), rotationY(0.0f), zoom(1.0f), showBoundary(false), initialized(false)
{
}

owMetalRenderer::~owMetalRenderer() {
    if (positionBuffer) positionBuffer->release();
    if (densityBuffer) densityBuffer->release();
    if (uniformBuffer) uniformBuffer->release();
    if (pipelineState) pipelineState->release();
    if (library) library->release();
    if (commandQueue) commandQueue->release();
    // Note: device is shared, don't release
}

bool owMetalRenderer::init(void* nsWindow, owConfigProperty* cfg) {
    config = cfg;
    
    // Get default Metal device
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal renderer: Failed to get Metal device" << std::endl;
        return false;
    }
    
    std::cout << "Metal renderer using: " << device->name()->utf8String() << std::endl;
    
    commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        std::cerr << "Metal renderer: Failed to create command queue" << std::endl;
        return false;
    }
    
    createPipeline();
    createBuffers(200000);  // Support up to 200k particles
    
    initialized = true;
    std::cout << "Metal renderer initialized" << std::endl;
    return true;
}

void owMetalRenderer::createPipeline() {
    // Load shader source from file
    std::string shaderPath = "./src/metal/particleRenderer.metal";
    std::ifstream file(shaderPath);
    if (!file.is_open()) {
        // Try build directory
        shaderPath = "./build/src/metal/particleRenderer.metal";
        file.open(shaderPath);
    }
    
    std::string shaderSource;
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        shaderSource = buffer.str();
        file.close();
    } else {
        std::cerr << "Metal renderer: Could not load shader file" << std::endl;
        return;
    }
    
    NS::Error* error = nullptr;
    library = device->newLibrary(NS::String::string(shaderSource.c_str(), NS::UTF8StringEncoding), nullptr, &error);
    
    if (!library) {
        std::cerr << "Metal renderer: Failed to compile shaders" << std::endl;
        if (error) {
            std::cerr << error->localizedDescription()->utf8String() << std::endl;
        }
        return;
    }
    
    MTL::Function* vertexFunction = library->newFunction(NS::String::string("particleVertex", NS::UTF8StringEncoding));
    MTL::Function* fragmentFunction = library->newFunction(NS::String::string("particleFragment", NS::UTF8StringEncoding));
    
    if (!vertexFunction || !fragmentFunction) {
        std::cerr << "Metal renderer: Failed to load shader functions" << std::endl;
        return;
    }
    
    MTL::RenderPipelineDescriptor* pipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipelineDesc->setVertexFunction(vertexFunction);
    pipelineDesc->setFragmentFunction(fragmentFunction);
    pipelineDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    pipelineDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    
    pipelineState = device->newRenderPipelineState(pipelineDesc, &error);
    
    if (!pipelineState) {
        std::cerr << "Metal renderer: Failed to create pipeline state" << std::endl;
        if (error) {
            std::cerr << error->localizedDescription()->utf8String() << std::endl;
        }
    }
    
    vertexFunction->release();
    fragmentFunction->release();
    pipelineDesc->release();
    
    std::cout << "Metal renderer pipeline created" << std::endl;
}

void owMetalRenderer::createBuffers(unsigned int maxCount) {
    maxParticles = maxCount;
    
    positionBuffer = device->newBuffer(maxParticles * 4 * sizeof(float), MTL::ResourceStorageModeShared);
    densityBuffer = device->newBuffer(maxParticles * sizeof(float), MTL::ResourceStorageModeShared);
    uniformBuffer = device->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
    
    std::cout << "Metal renderer buffers created for " << maxParticles << " particles" << std::endl;
}

void owMetalRenderer::updateParticles(const float* positions, const float* densities, unsigned int count) {
    particleCount = count;
    if (count > maxParticles) count = maxParticles;
    
    memcpy(positionBuffer->contents(), positions, count * 4 * sizeof(float));
    if (densities) {
        memcpy(densityBuffer->contents(), densities, count * sizeof(float));
    }
}

void owMetalRenderer::render() {
    if (!initialized || !pipelineState || !metalLayer) return;
    
    // Get drawable
    CA::MetalDrawable* drawable = metalLayer->nextDrawable();
    if (!drawable) return;
    
    // Update uniforms
    Uniforms* uniforms = (Uniforms*)uniformBuffer->contents();
    
    float aspect = 1.0f;  // Assuming square window for now
    simd_float4x4 projection = matrix_perspective(M_PI / 4, aspect, 0.1f, 1000.0f);
    simd_float4x4 view = matrix_translation(0, 0, -zoom * 100.0f);
    simd_float4x4 rotation = matrix_rotation(rotationX, rotationY);
    
    uniforms->modelViewProjection = simd_mul(projection, simd_mul(view, rotation));
    uniforms->offset = simd_make_float3(config->xmax / 2, config->ymax / 2, config->zmax / 2);
    uniforms->scale = 1.0f;
    uniforms->pointSize = 2.0f;
    uniforms->rho0 = config->getConst("rho0");
    uniforms->particleCount = particleCount;
    uniforms->showBoundary = showBoundary ? 1 : 0;
    
    // Create command buffer
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    
    // Create render pass
    MTL::RenderPassDescriptor* renderPass = MTL::RenderPassDescriptor::alloc()->init();
    renderPass->colorAttachments()->object(0)->setTexture(drawable->texture());
    renderPass->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    renderPass->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(0.1, 0.1, 0.2, 1.0));
    renderPass->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(renderPass);
    encoder->setRenderPipelineState(pipelineState);
    encoder->setVertexBuffer(positionBuffer, 0, 0);
    encoder->setVertexBuffer(densityBuffer, 0, 1);
    encoder->setVertexBuffer(uniformBuffer, 0, 2);
    
    // Draw all particles as points in a single call
    encoder->drawPrimitives(MTL::PrimitiveTypePoint, NS::UInteger(0), NS::UInteger(particleCount));
    
    encoder->endEncoding();
    
    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();
    
    renderPass->release();
}

void owMetalRenderer::setView(float rotX, float rotY, float z) {
    rotationX = rotX;
    rotationY = rotY;
    zoom = z;
}

void owMetalRenderer::setShowBoundary(bool show) {
    showBoundary = show;
}

#endif // __APPLE__
