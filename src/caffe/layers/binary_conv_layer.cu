#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {
#define sign(x) ((x)>=0?1:-1)
#define clamp(x) ((x) < -1 ? -1 : (x) >1 ? 1 : (x))
template <typename Dtype>
__global__ void BinaryGpu_binarize(const int n, const int num, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, n){//n:numbers of filters. 
		Dtype sum = 0;         //num: numbers of filters' elements.
		Dtype mean = 0;
		for (int coor = 0; coor < num; coor++){
			sum += std::abs(in[index*num + coor]) / Dtype(num);
<<<<<<< HEAD
			mean += in[index*num + coor];
		}
		for (int coor = 0; coor < num; coor++){
			out[index*num + coor] = sign(clamp(in[index*num + coor]))*sum;
=======
			mean += in[index*num + coor] / Dtype(num);
		}
		 for (int coor = 0; coor < num; coor++){
			 out[index*num + coor] = sign(clamp(in[index*num + coor]-mean))*sum; 
>>>>>>> dev
		}
		 
	}
}
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	 
	const int num = this->blobs_[0]->num();
	const int div = this->blobs_[0]->count() / num;
	BinaryGpu_binarize<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
		num, div, this->blobs_[0]->gpu_data(), this->W_b->mutable_gpu_data());
	
	caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), W_buffer->mutable_gpu_data());
	
	caffe_copy(this->blobs_[0]->count(), W_b->gpu_data(), this->blobs_[0]->mutable_gpu_data());

	//normal conv operations,directly copied from conv_layer.cpp
	const Dtype* weight = this->blobs_[0]->gpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
				top_data + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
			}
		}
	}
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->gpu_diff();
		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
			Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
			}
		}
		if (this->param_propagate_down_[0] || propagate_down[i]) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
						top_diff + n * this->top_dim_, weight_diff);
				}
				// gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) {
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
						bottom_diff + n * this->bottom_dim_);
				}
			}
		}
	}
	caffe_copy(this->blobs_[0]->count(), W_buffer->gpu_data(), this->blobs_[0]->mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionLayer);

}  // namespace caffe
