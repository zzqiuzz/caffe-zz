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
			mean += in[index*num + coor] / Dtype(num);
		}
		 for (int coor = 0; coor < num; coor++){
			 out[index*num + coor] = sign(clamp(in[index*num + coor]-mean))*sum; 
		}
		 
	}
}
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::gpuMeanClampBinarizeConvParam(const shared_ptr<Blob<Dtype> > weights,
	const shared_ptr<Blob<Dtype> > wb){
	const int num = weights->num();
	const int div = weights->count() / num; 
	BinaryGpu_binarize<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
		num, div, weights->gpu_data(), wb->mutable_gpu_data());
}
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	//TODO:
	//convert float weights to binary 
	gpuMeanClampBinarizeConvParam(this->blobs_[0], W_b);
	/*const Dtype* cpu_d = this->blobs_[0]->cpu_data();
	const Dtype* gpu_d = this->blobs_[0]->gpu_data();
	for (int i = 0; i < 5; i++){
		std::cout << "cpu data: " << cpu_d[i];
		std::cout << " gpu data:" << gpu_d[i] << std::endl;
	}*/
	//store float weights to W_buffer
	copyGpuFromTo(this->blobs_[0], W_buffer);
	//reinitialize blob_ with binarized weights W_b.
	copyGpuFromTo(W_b, this->blobs_[0]);
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
	copyGpuFromTo(W_buffer, this->blobs_[0]);
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionLayer);

}  // namespace caffe
