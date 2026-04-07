#ifndef OW_FAST_RENDERER_H
#define OW_FAST_RENDERER_H

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

class owConfigProperty;

/**
 * Fast particle renderer using Vertex Buffer Objects (VBOs)
 * Renders all particles in a single draw call instead of per-particle glBegin/glEnd
 */
class owFastRenderer {
public:
    owFastRenderer();
    ~owFastRenderer();
    
    /**
     * Initialize with max particle count
     */
    void init(unsigned int maxParticles);
    
    /**
     * Update particle data for rendering
     * @param positions Array of float4 (x, y, z, type) for each particle
     * @param densities Array of densities (can be null)
     * @param count Number of particles
     * @param config Config for bounds and rho0
     */
    void updateParticles(const float* positions, const float* densities, 
                         unsigned int count, owConfigProperty* config);
    
    /**
     * Render all particles
     * @param scale Display scale factor
     * @param showBoundary Whether to show boundary particles
     */
    void render(float scale, bool showBoundary = false);
    
    /**
     * Check if renderer is available
     */
    bool isAvailable() const { return vboSupported; }
    
private:
    void computeColors(const float* positions, const float* densities,
                       unsigned int count, owConfigProperty* config);
    
    GLuint positionVBO;    // Vertex buffer for positions
    GLuint colorVBO;       // Vertex buffer for colors
    
    float* vertexData;     // CPU-side vertex data (transformed positions)
    float* colorData;      // CPU-side color data
    
    unsigned int maxParticles;
    unsigned int particleCount;
    bool vboSupported;
    bool initialized;
};

#endif // OW_FAST_RENDERER_H
