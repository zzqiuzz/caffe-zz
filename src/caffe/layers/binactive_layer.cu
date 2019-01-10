#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/binactive_layer.hpp"

namespace caffe{
	
template <typename Dtype>
__global__ void BinActiveForward(const int n, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index,n){
		out[index] = in[index] > 0 ? Dtype(1) : Dtype(-1);	
	}
}
template <typename Dtype>
void BinActiveLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
	
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	BinActiveForward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
				count,bottom_data,top_data);
	CUDA_POST_KERNEL_CHECK; 
}
template <typename Dtype>
__global__ void BinActiveBackward(const int n,const Dtype* in_diff,const Dtype* in_data,Dtype* out_diff)
{
	CUDA_KERNEL_LOOP(index,n){
		out_diff[index] = in_diff[index];
		if(in_data[index] >= Dtype(1) || in_data[index] <= Dtype(-1))
			out_diff[index] = 0;
	}	
}
template <typename Dtype>
void BinActiveLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
	if(propagate_down[0]){
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		BinActiveBackward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,top_diff,bottom_data,bottom_diff);
		CUDA_POST_KERNEL_CHECK; } 
}
	
INSTANTIATE_LAYER_GPU_FUNCS(BinActiveLayer);
	
	
}
