#ifndef OW_METAL_RENDERER_H
#define OW_METAL_RENDERER_H

#ifdef __APPLE__

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalLayer.hpp>

class owConfigProperty;

class owMetalRenderer {
public:
    owMetalRenderer();
    ~owMetalRenderer();
    
    // Initialize with window and config
    bool init(void* nsWindow, owConfigProperty* config);
    
    // Update particle data (call before render)
    void updateParticles(const float* positions, const float* densities, unsigned int count);
    
    // Render frame
    void render();
    
    // Set camera/view
    void setView(float rotX, float rotY, float zoom);
    
    // Toggle boundary particle visibility
    void setShowBoundary(bool show);
    
    // Check if initialized
    bool isInitialized() const { return initialized; }
    
private:
    void createPipeline();
    void createBuffers(unsigned int maxParticles);
    
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* library;
    MTL::RenderPipelineState* pipelineState;
    
    MTL::Buffer* positionBuffer;
    MTL::Buffer* densityBuffer;
    MTL::Buffer* uniformBuffer;
    
    CA::MetalLayer* metalLayer;
    
    owConfigProperty* config;
    unsigned int particleCount;
    unsigned int maxParticles;
    
    float rotationX;
    float rotationY;
    float zoom;
    bool showBoundary;
    bool initialized;
};

#endif // __APPLE__
#endif // OW_METAL_RENDERER_H
