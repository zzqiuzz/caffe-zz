#include <vector>

#include "caffe/layers/ternary_conv_layer.hpp"

namespace caffe {
 
template <typename Dtype>
__global__ void ternarize_data(const int num,const Dtype delta, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, num){
		const Dtype data = in[index];
		out[index] = (data>delta) - (data<-delta);
	}
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	Dtype delta_scaler = 0.7;
	const int N = this->blobs_[0]->count();
	Dtype* ternaryweight = this->W_t.mutable_gpu_data();
	this->delta_ = delta_scaler * this->blobs_[0]->asum_data() / N;
	caffe_copy<Dtype>(N, this->blobs_[0]->gpu_data(), ternaryweight);
	ternarize_data<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N,this->delta_,this->blobs_[0]->gpu_data(),
		ternaryweight); 
	Dtype* temp1 = new Dtype(0);
	caffe_gpu_dot(N, this->W_t.gpu_data(), this->blobs_[0]->gpu_data(), temp1);
	Dtype* temp2 = new Dtype(0);
	caffe_gpu_dot(N, this->W_t.gpu_data(), this->W_t.gpu_data(), temp2); 
	this->alpha_ = (*temp1) / ((*temp2) + 1e-6);  
	caffe_gpu_scale(N, this->alpha_, this->W_t.gpu_data(), ternaryweight);
	delete temp1;
	delete temp2;
	//normal conv operations,directly copied from conv_layer.cpp 
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, ternaryweight,
				top_data + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->gpu_data();
				this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
			}
		}
	}
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* ternaryweight = W_t.gpu_data();
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
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, ternaryweight,
						bottom_diff + n * this->bottom_dim_);
				}
			}
			
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryConvolutionLayer);

}  // namespace caffe
