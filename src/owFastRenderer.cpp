#include "owFastRenderer.h"
#include "owConfigProperty.h"
#include "owPhysicsConstant.h"
#include <cstring>
#include <iostream>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#else
#include <GL/gl.h>
#include <GL/glext.h>
#endif

owFastRenderer::owFastRenderer()
    : positionVBO(0), colorVBO(0), vertexData(nullptr), colorData(nullptr),
      maxParticles(0), particleCount(0), vboSupported(true), initialized(false)
{
}

owFastRenderer::~owFastRenderer() {
    if (positionVBO) glDeleteBuffers(1, &positionVBO);
    if (colorVBO) glDeleteBuffers(1, &colorVBO);
    delete[] vertexData;
    delete[] colorData;
}

void owFastRenderer::init(unsigned int maxCount) {
    maxParticles = maxCount;
    
    // Allocate CPU-side buffers
    vertexData = new float[maxParticles * 3];  // x, y, z per particle
    colorData = new float[maxParticles * 4];   // r, g, b, a per particle
    
    // Create VBOs
    glGenBuffers(1, &positionVBO);
    glGenBuffers(1, &colorVBO);
    
    // Initialize position VBO
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Initialize color VBO
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, maxParticles * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    initialized = true;
    std::cout << "[FastRenderer] Initialized with VBOs for " << maxParticles << " particles" << std::endl;
}

void owFastRenderer::computeColors(const float* positions, const float* densities,
                                    unsigned int count, owConfigProperty* config) {
    float rho0 = config->getConst("rho0");
    
    for (unsigned int i = 0; i < count; i++) {
        int particleType = (int)positions[i * 4 + 3];
        float r = 0, g = 0, b = 1, a = 1;  // Default blue
        
        if (particleType == BOUNDARY_PARTICLE) {
            // Boundary - gray, semi-transparent
            r = 0.5f; g = 0.5f; b = 0.5f; a = 0.3f;
        } else if (particleType == ELASTIC_PARTICLE || 
                   (particleType > 2 && particleType < 3)) {
            // Worm body - dark green
            r = 0.1f; g = 0.5f; b = 0.1f; a = 1.0f;
        } else {
            // Liquid - color by density
            if (densities) {
                float rho = densities[i];
                float dc = 100.0f * (rho - rho0) / rho0;
                
                if (dc < 0) dc = 0;
                if (dc > 4) dc = 4;
                
                // Blue -> Cyan -> Green -> Yellow -> Red
                if (dc < 1) {
                    r = 0; g = dc; b = 1;
                } else if (dc < 2) {
                    r = 0; g = 1; b = 2 - dc;
                } else if (dc < 3) {
                    r = dc - 2; g = 1; b = 0;
                } else {
                    r = 1; g = 4 - dc; b = 0;
                }
            }
            a = 0.8f;
        }
        
        colorData[i * 4 + 0] = r;
        colorData[i * 4 + 1] = g;
        colorData[i * 4 + 2] = b;
        colorData[i * 4 + 3] = a;
    }
}

void owFastRenderer::updateParticles(const float* positions, const float* densities,
                                      unsigned int count, owConfigProperty* config) {
    if (!initialized) return;
    
    particleCount = count;
    if (count > maxParticles) count = maxParticles;
    
    float offsetX = config->xmax / 2;
    float offsetY = config->ymax / 2;
    float offsetZ = config->zmax / 2;
    
    // Transform positions to centered coordinates
    unsigned int visibleCount = 0;
    for (unsigned int i = 0; i < count; i++) {
        int particleType = (int)positions[i * 4 + 3];
        
        // Skip boundary particles (optional - you can add a flag)
        // if (particleType == BOUNDARY_PARTICLE) continue;
        
        vertexData[visibleCount * 3 + 0] = positions[i * 4 + 0] - offsetX;
        vertexData[visibleCount * 3 + 1] = positions[i * 4 + 1] - offsetY;
        vertexData[visibleCount * 3 + 2] = positions[i * 4 + 2] - offsetZ;
        visibleCount++;
    }
    
    // Compute colors
    computeColors(positions, densities, count, config);
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * 3 * sizeof(float), vertexData);
    
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * 4 * sizeof(float), colorData);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void owFastRenderer::render(float scale, bool showBoundary) {
    if (!initialized || particleCount == 0) return;
    
    glPushMatrix();
    glScalef(scale, scale, scale);
    
    glPointSize(2.0f);
    
    // Enable vertex arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    
    // Bind position VBO
    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    // Bind color VBO
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glColorPointer(4, GL_FLOAT, 0, 0);
    
    // Draw all particles in ONE call!
    glDrawArrays(GL_POINTS, 0, particleCount);
    
    // Cleanup
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glPopMatrix();
}
