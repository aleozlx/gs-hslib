#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include "vector_types.h"
#include "driver_types.h"

namespace heterogeneous_superpixel {
typedef int16_t SuperpixelIdx;
typedef uint32_t SuperpixelSize;
typedef std::unordered_map<SuperpixelIdx, std::vector<SuperpixelIdx>> ConnectivityMap;

// Divide and round up.
template <typename SIZE>
SIZE gdiv(SIZE a, SIZE b) {
	return (a + b - 1) / b;
}

// A POD structure for 2-D size representation, suitable for both host and device.
struct Size2D {
	typedef unsigned int Precision;
	unsigned int width;
	unsigned int height;
	operator dim3() const;
	size_t elems(unsigned char elem_stride = 1) const;
	dim3 operator*(unsigned int z) const;
	dim3 operator/(const dim3& tile_size) const;
	static Size2D From(const dim3& size);

#ifdef __CUDACC__
	__host__ __device__
#endif
	size_t addr(unsigned int x, unsigned int y, unsigned char elem_stride) const {
		unsigned int stride = elem_stride;
		const size_t offset_x = static_cast<size_t>(x * stride);
		stride *= width;
		return static_cast<size_t>(y * stride) + offset_x;
	}

#ifdef __CUDACC__
	__forceinline__
#endif
#ifdef __CUDACC__
	__host__ __device__
#endif
	size_t addr(unsigned int x, unsigned int y) const {
		return static_cast<size_t>(y * width) + x;
	}

#ifdef __CUDACC__
	__forceinline__
#endif
#ifdef __CUDACC__
	__host__ __device__
#endif
	bool boundary_check(unsigned int x, unsigned int y) const {
		return x < width && y < height;
	}
};

enum ColorMode {
	COLOR_RGB = 0,
	COLOR_LAB
};

// (POD) Properties of a superpixel.
struct SuperpixelAttr {
	SuperpixelIdx id;
	SuperpixelSize num_pixels;
	float2 center;
	float4 prototype_color;
	struct {
		float2 min;
		float2 max;
	} roi;
	friend std::ostream& operator<<(std::ostream& os, const SuperpixelAttr& s);
	void ToJson(char* buffer, size_t n) const;
};

typedef std::function<float(const SuperpixelAttr&, const SuperpixelAttr&)> SuperpixelDistanceFn;

// Generic tensor shape.
struct Shape {
	typedef int Precision;
	Precision d[4];
	Shape() : d() {}
	Shape(std::initializer_list<Precision> shape_src);
	explicit Shape(const Size2D& s);
	explicit operator Size2D() const;
	template <typename T>
	static Shape From(const T* shape_src, size_t n) {
		Shape shape_dst;
		const size_t num_cpy = (std::min)(std::size(shape_dst.d), n);
		for (int j = 0; j < num_cpy; ++j) {
			shape_dst.d[j] = static_cast<Precision>(shape_src[j]);
		}
		return shape_dst;
	}
	template <typename T>
	void To(T* shape_dst, size_t n) const {
		const size_t num_cpy = std::min(std::size(d), n);
		for (int j = 0; j < num_cpy; ++j) {
			shape_dst[j] = static_cast<T>(d[j]);
		}
	}
};

// An interface for shape inference enabled whatever.
class IShape {
public:
	virtual ~IShape() {};
	virtual unsigned int GetNumInputDims() const = 0;
	virtual unsigned int GetNumOutputDims() const = 0;
	virtual Shape GetInputShape(unsigned int) const = 0;
	virtual Shape GetOutputShape(unsigned int) const = 0;
	virtual void ComputeShapes(const std::vector<Shape>&) = 0;
};

// An abstract class for shape inference.
template <unsigned int NUM_I = 1, unsigned int NUM_O = 1>
class OpTopology : public IShape {
public:
	Shape input_shapes[NUM_I];
	Shape output_shapes[NUM_O];
	OpTopology() : input_shapes(), output_shapes() {}
	virtual ~OpTopology() {};
	unsigned int GetNumInputDims() const override {
		return NUM_I;
	}
	unsigned int GetNumOutputDims() const override {
		return NUM_O;
	}
	Shape GetInputShape(unsigned int i) const override {
		return i < NUM_I ? input_shapes[i] : Shape();
	}
	Shape GetOutputShape(unsigned int i) const override {
		return i < NUM_O ? output_shapes[i] : Shape();
	}
};

struct DeviceMemoryAllocationRequest {
	size_t size_in_elems;
	size_t size_in_bytes;
	DeviceMemoryAllocationRequest() : size_in_elems(0), size_in_bytes(0) {}
};

class OpBase;

// An interface for custom memory management.
//   It is only as dangerous as pointers :) You are not forced to use it.
class IDeviceMemory {
public:
	enum StorageSection {
		Input = 0, Output, Scratch, Local
	};
protected:
	virtual void* GetDevicePointer(StorageSection, unsigned int) const = 0;
public:
	template <typename T>
	inline T* GetInput(unsigned int i) const {
		return reinterpret_cast<T*>(GetDevicePointer(Input, i));
	}
	template <typename T>
	inline T* GetScratch(unsigned int i) const {
		return reinterpret_cast<T*>(GetDevicePointer(Scratch, i));
	}
	template <typename T>
	inline T* GetOutput(unsigned int i) const {
		return reinterpret_cast<T*>(GetDevicePointer(Output, i));
	}
	static void ApplyOpUnchecked(const OpBase& op, const IDeviceMemory& mem);
};

// Everything an operation could ask for, in the form of an open ABI.
class OpBase {
protected:
	std::vector<DeviceMemoryAllocationRequest> dynamic_shared_memory;
	// Caution: A deliberate SFINAE-adjacent pitfall. Call the functor with the correct memory.
	virtual void operator()(const IDeviceMemory&) const = 0;
	friend void IDeviceMemory::ApplyOpUnchecked(const OpBase& op, const IDeviceMemory& mem);
public:
	std::vector<DeviceMemoryAllocationRequest> input_allocations;
	std::vector<DeviceMemoryAllocationRequest> output_allocations;
	std::vector<DeviceMemoryAllocationRequest> scratch_allocations;
	cudaStream_t stream_id;
	OpBase(size_t num_inputs, size_t num_outputs, size_t num_scratch_spaces = 0) : stream_id(0) {
		input_allocations.resize(num_inputs);
		output_allocations.resize(num_outputs);
		scratch_allocations.resize(num_scratch_spaces);
	}
	explicit OpBase(const IShape* s, size_t num_scratch_spaces = 0) : stream_id(0) {
		input_allocations.resize(s->GetNumInputDims());
		output_allocations.resize(s->GetNumOutputDims());
		scratch_allocations.resize(num_scratch_spaces);
	}
	virtual ~OpBase() {}
	// Call this function to know all that it would take to run this operation.
	virtual void ComputeAllocationSizes() = 0;
	virtual size_t ComputeDynamicSharedMemorySize() {
		size_t sz = 0;
		for (const auto &a : dynamic_shared_memory) {
			sz += a.size_in_bytes;
		}
		return sz;
	}
	DeviceMemoryAllocationRequest* GetAllocationRequest(IDeviceMemory::StorageSection sect, unsigned int i) const {
		const std::vector<DeviceMemoryAllocationRequest>* storage_map[] = { &input_allocations, &output_allocations, &scratch_allocations, nullptr };
		const auto allocations = storage_map[sect];
		if (allocations && i < allocations->size()) {
			return const_cast<DeviceMemoryAllocationRequest*>(&(*allocations)[i]);
		}
		else return nullptr;
	}
};

class ICUDAGraphNode {
protected:
	cudaGraphNode_t cuda_node;
public:
	ICUDAGraphNode() : cuda_node(nullptr) {}
	cudaGraphNode_t GetCUDAGraphNode() const {
		return cuda_node;
	}
	virtual cudaGraphNode_t MakeCUDAGraphNode(cudaGraph_t) = 0;
};

class ConvertOp_uchar3_float4 : public OpTopology<>, public OpBase, public ICUDAGraphNode {
public:
	Size2D image_size;
	ColorMode color_mode;
	static const unsigned int _blksz = 32;
	cudaKernelNodeParams cuda_kernel_node_params;
	ConvertOp_uchar3_float4(unsigned int width, unsigned int height, ColorMode color_mode = COLOR_LAB) :
		OpBase(this), image_size{width, height}, color_mode(color_mode) {}
	virtual ~ConvertOp_uchar3_float4() {}
	void operator()(const unsigned char* im_input, float* im_output) const;
	void operator()(const IDeviceMemory& mem) const override {
		this->operator()(mem.GetInput<unsigned char>(0), mem.GetOutput<float>(0));
	}
	void ComputeAllocationSizes() override;
	void ComputeShapes(const std::vector<Shape>& input_shapes = std::vector<Shape>()) override {
		if (input_shapes.size() == 0) {
			this->input_shapes[0] = static_cast<Shape>(image_size);
			this->output_shapes[0] = this->input_shapes[0];
		}
		else {
			this->input_shapes[0] = input_shapes[0];
			this->output_shapes[0] = this->input_shapes[0];
			image_size = static_cast<Size2D>(this->input_shapes[0]);
		}
	}
	cudaGraphNode_t MakeCUDAGraphNode(cudaGraph_t) override;
};

struct SegmentationParams_gSLIC {
	Size2D image_size;
	SuperpixelSize target_superpixel_size;
	float compactness;
	ColorMode color_mode;
	SegmentationParams_gSLIC(unsigned int width, unsigned int height, SuperpixelSize target_superpixel_size, float compactness=0.6, ColorMode color_mode=COLOR_LAB) :
		image_size{width, height}, target_superpixel_size(target_superpixel_size), compactness(compactness), color_mode(color_mode)
	{}
};

class SegmentationOp_gSLIC : public OpTopology<1, 2>, public OpBase {
public:
	SegmentationParams_gSLIC params;
	unsigned int num_iterations;
	static const unsigned int _blksz = 16;
	explicit SegmentationOp_gSLIC(const SegmentationParams_gSLIC& params) :
		OpBase(this, 2), params(params), num_iterations(5) {}
	virtual ~SegmentationOp_gSLIC() {}
	void operator()(const float* im_input, SuperpixelAttr* im_output, SuperpixelIdx* sid_output, SuperpixelAttr* scratch_clustering_reduction, SuperpixelIdx* scratch_local_connectivity) const;
	void operator()(const IDeviceMemory& mem) const override {
		this->operator()(mem.GetInput<float>(0), mem.GetOutput<SuperpixelAttr>(0), mem.GetOutput<SuperpixelIdx>(1), mem.GetScratch<SuperpixelAttr>(0), mem.GetScratch<SuperpixelIdx>(1));
	}
	void ComputeAllocationSizes() override;
	void ComputeShapes(const std::vector<Shape>& input_shapes = std::vector<Shape>()) override;
	unsigned int GetNumSuperpixels(const IDeviceMemory& mem, bool exclude_empty_superpixel = true) const;
	void GetSuperpixels(const IDeviceMemory& mem, std::vector<SuperpixelAttr>& out, bool exclude_empty_superpixel = true) const;
};

struct ConnectivityParams {
	Size2D image_size;
	unsigned int num_superpixels;
	explicit ConnectivityParams(const SegmentationParams_gSLIC& params, unsigned int num_superpixels = 0) :
		image_size(params.image_size), num_superpixels(num_superpixels) {}
	ConnectivityParams(unsigned int width, unsigned int height, unsigned int num_superpixels = 0) :
		image_size{width, height}, num_superpixels(num_superpixels) {}
};

class ConnectivityOp : public OpBase {
public:
	ConnectivityParams params;
	ConnectivityOp(const ConnectivityParams& params, unsigned int block_size = 32) :
		OpBase(2, 1), params(params), blksz(block_size) {}
	virtual ~ConnectivityOp() {}
	void operator()(const SuperpixelAttr* im_output, const SuperpixelIdx* sid_output, unsigned char* connectivity_matrix) const;
	void operator()(const IDeviceMemory& mem) const override {
		this->operator()(mem.GetInput<SuperpixelAttr>(0), mem.GetInput<SuperpixelIdx>(1), mem.GetOutput<unsigned char>(0));
	}
	void ComputeAllocationSizes() override;
	static size_t GetNumEdges(const ConnectivityMap& connectivity_map);
	void GetConnectivityMap(const IDeviceMemory& mem, ConnectivityMap& connectivity_map, SuperpixelDistanceFn dist = {}, size_t top_n = INT_MAX) const;
	void GetMST(const IDeviceMemory& mem, ConnectivityMap& connectivity_map, SuperpixelDistanceFn dist, ConnectivityMap& return_map) const;
protected:
	unsigned int blksz;
};

struct FeatureExtractionFactorsParams {
	unsigned int num_superpixels;
	Size2D input_size;
	Size2D output_size;
	Size2D output_tensor;
	FeatureExtractionFactorsParams(unsigned int num_superpixels, const Size2D& input_size, const Size2D& output_size) :
		num_superpixels(num_superpixels),
		input_size(input_size),
		output_size(output_size),
		output_tensor{
			static_cast<unsigned int>(output_size.elems()),
			num_superpixels
		}
	{}
};

class FeatureExtractionFactorsOp : public OpBase {
public:
	FeatureExtractionFactorsParams params;
	explicit FeatureExtractionFactorsOp(const FeatureExtractionFactorsParams& params) :
		OpBase(1, 1), params(params) {}
	virtual ~FeatureExtractionFactorsOp() {}
	void operator()(const SuperpixelIdx* im_sid_input, float* output_tensor) const;
	void operator()(const IDeviceMemory& mem) const override {
		this->operator()(mem.GetInput<SuperpixelIdx>(0), mem.GetOutput<float>(0));
	}
	void ComputeAllocationSizes() override {
		auto& im_sid_input = input_allocations[0];
		im_sid_input.size_in_elems = params.input_size.elems();
		im_sid_input.size_in_bytes = im_sid_input.size_in_elems * sizeof(SuperpixelIdx);

		auto& output_tensor = output_allocations[0];
		output_tensor.size_in_elems = params.output_size.elems() * params.num_superpixels;
		output_tensor.size_in_bytes = output_tensor.size_in_elems * sizeof(float);
	}
};

struct BoundaryAnnotationParams {
	Size2D image_size;
	uchar3 color;
	float alpha;
	bool inplace;
	BoundaryAnnotationParams(uchar3 color, unsigned int width = 0, unsigned int height = 0) :
		image_size { width, height }, color(color), alpha(1.f), inplace(false)
	{}
};

class BoundaryAnnotationOp : public OpTopology<2, 1>, public OpBase {
public:
	BoundaryAnnotationParams params;
	explicit BoundaryAnnotationOp(const BoundaryAnnotationParams& params) :
		OpBase(this), params(params)
	{}
	virtual ~BoundaryAnnotationOp() {}
	void operator()(const unsigned char* im_input, const SuperpixelIdx* im_sid, unsigned char* im_output) const;
	void operator()(const IDeviceMemory& mem) const override {
		this->operator()(mem.GetInput<unsigned char>(0), mem.GetInput<SuperpixelIdx>(1), mem.GetOutput<unsigned char>(0));
	}
	void ComputeAllocationSizes() override;
	void ComputeShapes(const std::vector<Shape>& input_shapes = std::vector<Shape>()) override;
};

struct EdgeAnnotationParams {
	Size2D image_size;
	unsigned int num_edges;
	uchar3 color_vertex;
	uchar3 color_edge;
	float w2alpha;
	bool inplace;
	EdgeAnnotationParams(uchar3 color, unsigned int width = 0, unsigned int height = 0) :
		image_size { width, height }, num_edges(0), color_vertex(color), color_edge(color), w2alpha(1.f), inplace(false)
	{}
};

struct WeightedEdge {
	float w;
	float2 p1, p2;
};

class EdgeAnnotationOp : public OpTopology<2, 1>, public OpBase {
public:
	EdgeAnnotationParams params;
	explicit EdgeAnnotationOp(const EdgeAnnotationParams& params) :
		OpBase(this), params(params)
	{}
	virtual ~EdgeAnnotationOp() {}
	void operator()(const unsigned char* im_input, const WeightedEdge* edges, unsigned char* im_output) const;
	void operator()(const IDeviceMemory& mem) const override {
		this->operator()(mem.GetInput<unsigned char>(0), mem.GetInput<WeightedEdge>(1), mem.GetOutput<unsigned char>(0));
	}
	void ComputeAllocationSizes() override;
	void ComputeShapes(const std::vector<Shape>& input_shapes = std::vector<Shape>()) override;
};

} // namespace heterogeneous_superpixel
