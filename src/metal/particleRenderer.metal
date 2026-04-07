#include <metal_stdlib>
using namespace metal;

// Vertex shader input
struct VertexIn {
    float4 position [[attribute(0)]];  // xyz = position, w = particle type
};

// Vertex shader output / Fragment shader input
struct VertexOut {
    float4 position [[position]];
    float4 color;
    float pointSize [[point_size]];
};

// Uniforms
struct Uniforms {
    float4x4 modelViewProjection;
    float3 offset;       // Center offset (xmax/2, ymax/2, zmax/2)
    float scale;         // Display scale
    float pointSize;     // Base point size
    float rho0;          // Rest density for coloring
    uint particleCount;
    uint showBoundary;   // 0 = hide boundary particles
};

// Constants for particle types
constant int BOUNDARY_PARTICLE = 3;
constant int LIQUID_PARTICLE = 1;
constant int ELASTIC_PARTICLE = 2;

vertex VertexOut particleVertex(
    uint vertexId [[vertex_id]],
    device float4* positions [[buffer(0)]],
    device float* densities [[buffer(1)]],
    constant Uniforms& uniforms [[buffer(2)]]
) {
    VertexOut out;
    
    float4 pos = positions[vertexId];
    int particleType = int(pos.w);
    
    // Skip boundary particles if not showing them
    if (particleType == BOUNDARY_PARTICLE && uniforms.showBoundary == 0) {
        out.position = float4(0, 0, -1000, 1);  // Off screen
        out.pointSize = 0;
        return out;
    }
    
    // Transform position
    float3 worldPos = (pos.xyz - uniforms.offset) * uniforms.scale;
    out.position = uniforms.modelViewProjection * float4(worldPos, 1.0);
    
    // Color based on particle type and density
    if (particleType == ELASTIC_PARTICLE || (particleType > 2 && particleType < 3)) {
        // Worm body - green/dark color
        out.color = float4(0.1, 0.6, 0.1, 1.0);  // Green
    } else if (particleType == LIQUID_PARTICLE) {
        // Water - blue with density-based coloring
        float rho = densities[vertexId];
        float dc = 100.0f * (rho - uniforms.rho0) / uniforms.rho0;
        dc = clamp(dc, 0.0f, 1.0f);
        
        // Gradient: blue -> cyan -> green -> yellow -> red
        if (dc < 0.25) {
            out.color = float4(0, dc * 4, 1, 0.7);  // blue to cyan
        } else if (dc < 0.5) {
            out.color = float4(0, 1, 1 - (dc - 0.25) * 4, 0.7);  // cyan to green
        } else if (dc < 0.75) {
            out.color = float4((dc - 0.5) * 4, 1, 0, 0.7);  // green to yellow
        } else {
            out.color = float4(1, 1 - (dc - 0.75) * 4, 0, 0.7);  // yellow to red
        }
    } else {
        // Boundary or other - gray (usually hidden)
        out.color = float4(0.5, 0.5, 0.5, 0.3);
    }
    
    out.pointSize = uniforms.pointSize;
    
    return out;
}

fragment float4 particleFragment(VertexOut in [[stage_in]]) {
    // Simple circle shape for points
    float2 pointCoord = float2(0.5) - in.position.xy / in.position.w;
    // For point sprites, we just return the color
    return in.color;
}
