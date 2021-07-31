#pragma once
#include <algorithm>
#include <cstdio>
#include <deque>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>
#include "cuda_runtime.h"
#include "heterogeneous_superpixel.hpp"

namespace heterogeneous_superpixel::optional {
#define cudaCheck(ret_code) gpuCheck((ret_code), __FILE__, __LINE__)
inline cudaError_t gpuCheck(cudaError_t ret_code, const char* file, int line) {
	if (ret_code != cudaSuccess) {
		fprintf(stderr, "gpuCheck: %s %s %d\n", cudaGetErrorString(ret_code), file, line);
	}
	return ret_code;
}

class IDependency {
	IDependency(IDependency&) = delete;
	IDependency(IDependency&&) = delete;
public:
	IDependency* predecessor;
	std::list<IDependency*> successors;
	IDependency() : predecessor(nullptr) {}
	virtual ~IDependency() {}

	virtual void Link(IDependency* predecessor) {
		this->predecessor = predecessor;
		DancingLinkRestore();
	}

	void DancingLinkDrop() {
		if (predecessor) {
			auto it_this = std::find(predecessor->successors.begin(), predecessor->successors.end(), this);
			if (it_this != predecessor->successors.end()) {
				predecessor->successors.erase(it_this);
			}
			for (auto successor : successors) {
				predecessor->successors.push_back(successor);
			}
		}
		for (auto successor : successors) {
			successor->predecessor = this->predecessor;
		}
	}

	void DancingLinkRestore() {
		if (predecessor) {
			for (auto successor : successors) {
				auto it_successor = std::find(predecessor->successors.begin(), predecessor->successors.end(), successor);
				if (it_successor != predecessor->successors.end()) {
					predecessor->successors.erase(it_successor);
				}
			}
			predecessor->successors.push_back(this);
		}
		for (auto successor : successors) {
			successor->predecessor = this;
		}
	}
};

struct DeviceMemoryAllocation : public DeviceMemoryAllocationRequest, public IDependency {
	void* device_ptr;
	// Mark as owned if the memory is actually managed using this data structure.
	bool is_owned;
	// Mark as dynamic if the allocation requirement is unknown at build.
	bool is_dynamic;
	// Mask as unused if caused by optional or in-place data access.
	bool is_unused;

	DeviceMemoryAllocation(const DeviceMemoryAllocationRequest& req) :
		DeviceMemoryAllocationRequest(req),
		device_ptr(nullptr),
		is_owned(true),
		is_dynamic(false),
		is_unused(false)
	{}

	void Borrow(void* device_ptr, bool is_owned = false) {
		if (device_ptr) {
			this->device_ptr = device_ptr;
		}
		this->is_owned = is_owned;
	}

	void Link(IDependency* _predecessor) override {
		auto a = dynamic_cast<DeviceMemoryAllocation*>(_predecessor);
		if (a) {
			this->Link(a);
		}
	}

	void Link(DeviceMemoryAllocation* predecessor, bool is_owned = false) {
		IDependency::Link(predecessor);
		TryBorrowing(is_owned);
	}

	void TryBorrowing(bool is_owned = false) {
		auto a = dynamic_cast<DeviceMemoryAllocation*>(this->predecessor);
		if (a) {
			Borrow(a->device_ptr, is_owned);
		}
	}
};

class LambdaOp : public OpBase {
protected:
	std::function<void(const IDeviceMemory& mem)> op_func;
public:
	explicit LambdaOp(std::function<void(const IDeviceMemory& mem)> functor) : OpBase(0, 0, 0), op_func(functor) {}
	virtual ~LambdaOp() {}
	void operator()(const IDeviceMemory& mem) const override {
		this->op_func(mem);
	}
	void ComputeAllocationSizes() override {
		// This is just a bare minimum for lambdas. Set {input,output,scratch}_allocations manually.
		// For a more fully featured operation, should consider deriving a new one from OpBase.
	}
};

class UnmanagedOp : public OpBase {
protected:
	std::function<void()> op_func;
public:
	UnmanagedOp(std::function<void()> functor) : OpBase(0, 0, 0), op_func(functor) {}
	virtual ~UnmanagedOp() {}
	void operator()(const IDeviceMemory& mem) const override {
		this->op_func();
	}
	void ComputeAllocationSizes() override {
		// This is a completely unmanaged operation.
		// The only use case to even capture this is to facilitate interaction with graph optimization mechanisms.
	}
};

class ManagedDeviceMemory : public IDeviceMemory {
protected:
	OpBase& op;
	const std::type_info& op_type;
	bool is_from_lambda;
	std::deque<DeviceMemoryAllocation> input_allocations;
	std::deque<DeviceMemoryAllocation> output_allocations;
	std::deque<DeviceMemoryAllocation> scratch_allocations;
	ManagedDeviceMemory(ManagedDeviceMemory&) = delete;
	ManagedDeviceMemory(ManagedDeviceMemory&&) = delete;
	void InitAllocations(OpBase& op) {
		for (auto &_a : op.input_allocations) {
			this->input_allocations.emplace_back(_a);
		}
		for (auto &_a : op.scratch_allocations) {
			this->scratch_allocations.emplace_back(_a);
		}
		for (auto &_a : op.output_allocations) {
			this->output_allocations.emplace_back(_a);
		}
	}
public:
	explicit ManagedDeviceMemory(OpBase& op) : op(op), op_type(typeid(op)), is_from_lambda(false) {
		op.ComputeAllocationSizes();
		InitAllocations(op);
	}

	explicit ManagedDeviceMemory(std::function<void(const IDeviceMemory& mem)> functor) : op(*new LambdaOp(functor)), op_type(typeid(functor)), is_from_lambda(true) {
		InitAllocations(op);
	}

	explicit ManagedDeviceMemory(std::function<void()> functor) : op(*new UnmanagedOp(functor)), op_type(typeid(functor)), is_from_lambda(true) {
	}

	~ManagedDeviceMemory() {
		for (auto sect : {Input, Scratch, Output}) {
			for (auto &a : SelectStorage(sect)) {
				if (a.is_owned) {
					cudaCheck(cudaFree(a.device_ptr));
				}
			}
		}
		if (is_from_lambda) {
			delete &op;
		}
	}

	ManagedDeviceMemory& mark_as_dynamic(StorageSection sect, unsigned int i, bool v = true) {
		auto& allocations = SelectStorage(sect);
		if (i < allocations.size()) {
			allocations[i].is_dynamic = v;
		}
		return *this;
	}

	ManagedDeviceMemory& mark_as_dependent(StorageSection sect_dst, unsigned int di, ManagedDeviceMemory& mem, StorageSection sect_src, unsigned int si /*DeviceMemoryAllocation* dep*/) {
		auto& allocations = SelectStorage(sect_dst);
		if (di < allocations.size()) {
			auto dep = mem(sect_src, si);
			if (dep) {
				allocations[di].Link(dep);
			}
			else {
				// remove dependency
				throw std::invalid_argument("Removing dependency is not implemented");
			}
		}
		return *this;
	}

	ManagedDeviceMemory& set_raw_pointer(StorageSection sect_dst, unsigned int di, void *p) {
		auto& allocations = SelectStorage(sect_dst);
		if (di < allocations.size()) {
			allocations[di].device_ptr = p;
			allocations[di].is_owned = false;
		}
		return *this;
	}

	ManagedDeviceMemory& allocate() {
		for (auto sect : {Input, Scratch, Output}) {
			for (auto &a : SelectStorage(sect)) {
				if (a.is_owned && !a.is_dynamic) {
					std::cerr << "Allocating " << a.size_in_bytes << " Bytes as " << (sect != Scratch ? "I/O" : "scratch") << " space for " << op_type.name() << std::endl;
					cudaCheck(_allocate(&a.device_ptr, a.size_in_bytes, sect != Scratch));
				}
			}
		}
		return *this;
	}

	void Reallocate(StorageSection sect, unsigned int i) {
		auto& allocations = SelectStorage(sect);
		if (i < allocations.size()) {
			auto &a = allocations[i];
			if (a.is_owned) {
				cudaCheck(cudaFree(a.device_ptr));
				op.ComputeAllocationSizes();
				const auto new_reqest = op.GetAllocationRequest(sect, i);
				static_cast<DeviceMemoryAllocationRequest&>(a) = *new_reqest;
				cudaCheck(_allocate(&a.device_ptr, a.size_in_bytes, sect != Scratch));
			}
		}
	}

	DeviceMemoryAllocation* operator()(StorageSection sect, unsigned int i) const {
		auto& allocations = SelectStorage(sect);
		if (i < allocations.size()) {
			return const_cast<DeviceMemoryAllocation*>(&allocations[i]);
		}
		else return nullptr;
	}

	void ApplyOp() const {
		IDeviceMemory::ApplyOpUnchecked(op, *this);
	}

protected:
	void* GetDevicePointer(StorageSection sect, unsigned int i) const override {
		auto& allocations = SelectStorage(sect);
		if (i < allocations.size()) {
			return allocations[i].device_ptr;
		}
		else return nullptr;
	}

	static cudaError_t _allocate(void** ptr_to_device_ptr, size_t size_in_bytes, bool managed) {
		// TODO generalize allocator
		if (managed) {
			return cudaMallocManaged(ptr_to_device_ptr, size_in_bytes);
		}
		else {
			return cudaMalloc(ptr_to_device_ptr, size_in_bytes);
		}
	}

private:
	std::deque<DeviceMemoryAllocation>& SelectStorage(StorageSection sect) const {
		switch(sect) {
		case Input:
			return const_cast<std::deque<DeviceMemoryAllocation>&>(input_allocations);
		case Output:
			return const_cast<std::deque<DeviceMemoryAllocation>&>(output_allocations);
		case Scratch:
			return const_cast<std::deque<DeviceMemoryAllocation>&>(scratch_allocations);
		default:
			throw std::invalid_argument("Unsupported StorageSection");
		}
	}
};

// class OpNodeInterface {
// protected:
// 	heterogeneous_superpixel::OpBase& op;

// 	explicit OpNodeInterface(heterogeneous_superpixel::OpBase& op) : op(op) {
// 	}

// public:

// };

// template <typename Op>
// class OpNode : public OpNodeInterface {
// public:
// 	OpNode(Op& op) :
// 		OpNodeInterface(op)
// 	{

// 	}
// };

#if 0
class OpNodeInterface {
public:
	std::unique_ptr<heterogeneous_superpixel::OpBase> op;

	explicit OpNodeInterface(std::unique_ptr<heterogeneous_superpixel::OpBase> op) : op(std::move(op)) {
	}
};

struct OpConnector {
	size_t op_idx;
	size_t alloc_idx;
};

class OpGraph {
public:
	std::vector<OpNodeInterface> nodes;
	std::list<std::pair<OpConnector, OpConnector>> edges;

	// void GenerateDeviceMemoryRequirement() {
	// 	for (auto& i : nodes) {
	// 		i.op->ComputeAllocationSizes();
	// 	}
	// }

	void Link(size_t src_op_idx, size_t src_alloc_idx, size_t dst_op_idx, size_t dst_alloc_idx) {
		edges.push_back(std::make_pair(OpConnector{src_op_idx, src_alloc_idx}, OpConnector{dst_op_idx, dst_alloc_idx}));
	}

	int Link(size_t src_op_idx, size_t dst_op_idx) {
		if (src_op_idx < nodes.size() && dst_op_idx < nodes.size()) {
			const auto& src = nodes[src_op_idx].op->output_allocations;
			const auto& dst = nodes[dst_op_idx].op->input_allocations;
			if (src.size() == dst.size()) {
				for (size_t j = 0; j < dst.size(); ++j) {
					Link(src_op_idx, j, dst_op_idx, j);
				}
				return static_cast<int>(dst.size());
			}
			else return 0;
		}
		else return 0;
	}

protected:
};
#endif

} // namespace heterogeneous_superpixel::optional
