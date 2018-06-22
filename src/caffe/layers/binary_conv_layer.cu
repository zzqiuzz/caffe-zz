#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {
#define sign(x) ((x)>=0?1:-1)
#define clamp(x) ((x) < -1 ? -1 : (x) >1 ? 1 : (x))
template <typename Dtype>
__global__ void BinaryGpu_binarize(const int num, const int weight_col, const Dtype* alpha,const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, num){
		int n = index / weight_col; 
		const Dtype binarycode = in[index] >= 0 ? 1 : -1; 
		out[index] = binarycode*alpha[n];

		/*for (int coor = 0; coor < weight_col; coor++){
			out[index*weight_col + coor] = sign(in[index*weight_col + coor]) * alpha[index];
		}*/
	}
}
template <typename Dtype>
__global__ void Gradient_adder(const int num,const int weight_dim,const Dtype* weight,Dtype* weight_diff,const Dtype* alpha){
	CUDA_KERNEL_LOOP(index, num){ 
		const int n = index / weight_dim;
		Dtype multiplier = 0;
		if (abs(weight[index]) <= 1)
		{
			multiplier = 1;
			multiplier *= alpha[n];
		}
		multiplier += Dtype(1) / weight_dim;
		multiplier *= (1 - 1./weight_dim);
		multiplier *= weight_dim;
		weight_diff[index] *= multiplier; 
	}
}
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	 
	//const int num = this->blobs_[0]->num();
	const int num = this->num_output_;
	const int div = this->blobs_[0]->count() / num;
	const int N = this->blobs_[0]->count();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* binaryweight = this->W_b.mutable_gpu_data();
	caffe_copy<Dtype>(N, weight, binaryweight);
	//calculate mean_.
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num, div, 1. / div, weight, weight_sum_multiplier.cpu_data(), 0.,
		mean_.mutable_cpu_data()); 
	//extract mean.
	const Dtype* mean_data=mean_.cpu_data();
	for(int i=0;i<num;++i){
		caffe_gpu_add_scalar<Dtype>(div, *(mean_data + i), this->blobs_[0]->mutable_gpu_data() + i*div);
	}
	//clamp weights
	this->blobs_[0]->clip_data();
	//calculate alphas_.
	for (int n = 0; n < num; n++){
		caffe_gpu_asum<Dtype>(div, weight + n*div, alphas_.mutable_cpu_data() + n); 
		alphas_.mutable_cpu_data()[n] /= div; 
	}
	//binarize weights.
	BinaryGpu_binarize<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (
		N, div, this->alphas_.gpu_data(), weight, binaryweight);
	

	//normal conv operations,directly copied from conv_layer.cpp
	//const Dtype* weight = this->blobs_[0]->gpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, binaryweight,
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
	//const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* binaryweight = W_b.gpu_data();
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
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, binaryweight,
						bottom_diff + n * this->bottom_dim_);
				}
			}
			//
			const Dtype* weight = this->blobs_[0]->gpu_data();
			const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
			const int n = this->blobs_[0]->count();

			Gradient_adder<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >
				(n, weight_dim, weight, weight_diff, alphas_.gpu_data());
			//
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionLayer);

}  // namespace caffe
