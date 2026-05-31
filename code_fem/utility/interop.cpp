// Expose CUDA–OpenGL interop functions to Python

// Pipeline:
// Python/Warp -> pybind11 bindings -> C++ CUDA/OpenGL interop -> OpenGL VBO GPU memory
// Give me the GPU address of an OpenGL buffer so I can write to it



// expose C++ functions to Python
// call CUDA/OpenGL code from Python
#include <pybind11/pybind11.h> // lets C++ functions be called from Python

// CUDA runtime API
// provide basic CUDA functions (cudaMemcpy, cudaMalloc, cudaDeviceSynchronize,etc.)
#include <cuda_runtime.h> 

// CUDA <-> OpenGL interoperability API
// After installing CUDA, you already get:
// /usr/local/cuda/include/cuda_gl_interop.h
// contains like:
// cudaGraphicsGLRegisterBuffer
// cudaGraphicsMapResources
// cudaGraphicsResourceGetMappedPointer
// cudaGraphicsUnmapResources
#include <cuda_gl_interop.h> // CUDA/OpenGL interoperability API provided by the CUDA Toolkit

// OpenGL API
// provides GLuint, glBindBuffer, glBufferData, etc.
#include <GL/gl.h> //OpenGL types like GLuint

// C++ error handling
#include <stdexcept> // for throwing errors




namespace py = pybind11;

// Error macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    }


// global resource
// This is the handle to OpenGL VBO inside CUDA    
cudaGraphicsResource* resource = nullptr;


// Register OpenGL VBO with CUDA
void register_vbo(unsigned int vbo) {

    // clean old registration
    if (resource != nullptr) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
    }

    // Register OpenGL VBO with CUDA
    // vbo = OpenGL buffer ID, CUDA now knows about it
    // flag: cudaGraphicsMapFlagsWriteDiscard, meaning it will overwrite everything
    // good for rendering,  avoids sync overhead
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &resource,
        vbo,
        cudaGraphicsMapFlagsWriteDiscard
    ));
}

// cleanup when done
void unregister_vbo() {
    if (resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
    }
}



// OpenGL VBO -> CUDA pointer
// MUST call unmap_vbo() after using the pointer!!
uintptr_t map_vbo() {

    if (!resource) {
        throw std::runtime_error("VBO not registered");
    }
    //map resource, locks the buffer for CUDA access
    // cudaError_t cudaGraphicsUnmapResources(
    //     int count,
    //     cudaGraphicsResource_t *resources,
    //     cudaStream_t stream
    // );
    // count: 
    // 1, One graphics resource,  
    // 2, TWO graphics resource, maybe one VBO for positions, one VBO for colors

    // cudaStream_t stream = 0: Use the default CUDA stream.
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));


    void* ptr = nullptr;
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &size, resource));

    if (!ptr || size == 0) {
        throw std::runtime_error("Failed to map VBO");
    }
    
    // Convert device pointer to integer so it can be passed to Python.
    // This does NOT transfer memory. 
    // it only passes the address.
    return reinterpret_cast<uintptr_t>(ptr);
}




void unmap_vbo() {
    if (!resource) return;

    // CUDA releases the buffer -> OpenGL can use it again
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));
}



// gpu copy
// Used for device-to-device copy into OpenGL VBO (GPU-copy strategy)
void memcpy_to_vbo(uintptr_t dst, uintptr_t src, size_t size) {
    CUDA_CHECK(cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToDevice));
}


// Python binding
// create a Python module named cuda_gl_interop, and call the module object m
PYBIND11_MODULE(cuda_gl_interop, m) {
    // Exposed functions:
    // Add a Python-visible function to this module.
    m.def("register_vbo", &register_vbo);
    m.def("map_vbo", &map_vbo);
    m.def("unmap_vbo", &unmap_vbo);
    m.def("unregister_vbo", &unregister_vbo);
    m.def("memcpy_to_vbo", &memcpy_to_vbo);
}



// Notes:

// register_vbo: Tell CUDA that this OpenGL buffer exists, 
// This creates the long-term connection between: OpenGL VBO and CUDA runtime
// registering is expensive and persistent.
// done ONCE.


// Mapped resource ownership:
// Before map, ownership is OpenGL
// After map, ownership is CUDA
// After unmap, ownership is OpenGL

// map_vbo: Temporarily give CUDA access to the GPU memory
// CUDA may now access the buffer memory directly.
// This temporarily locks/transfers ownership to CUDA.
// Usually done EVERY FRAME or EVERY UPDATE.
