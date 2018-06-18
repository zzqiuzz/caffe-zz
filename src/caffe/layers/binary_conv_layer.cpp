#include <vector>
#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe{
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
	const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
	
	alphas_.Reshape(this->num_output_,1,1,1);
	mean_.Reshape(this->num_output_, 1, 1, 1);
	W_b.Reshape(this->blobs_[0]->shape());
	weight_sum_multiplier.Reshape(weight_dim, 1, 1, 1); 
	caffe_set(this->num_output_, Dtype(1), weight_sum_multiplier.mutable_cpu_data()); 
	caffe_set(this->num_output_, Dtype(1), alphas_.mutable_cpu_data());
	caffe_set(this->num_output_, Dtype(1), mean_.mutable_cpu_data());
	
	 
}
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
	const int* kernel_shape_data = this->kernel_shape_.cpu_data();
	const int* stride_data = this->stride_.cpu_data();
	const int* pad_data = this->pad_.cpu_data();
	const int* dilation_data = this->dilation_.cpu_data();
	this->output_shape_.clear();
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		// i + 1 to skip channel axis
		const int input_dim = this->input_shape(i + 1);
		const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
		const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
			/ stride_data[i] + 1;
		this->output_shape_.push_back(output_dim);
	}
	 
}  
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){  
	const Dtype* weight = this->blobs_[0]->cpu_data(); 
	caffe_copy(W_b.count(), weight, W_b.mutable_cpu_data());
	const int num = this->num_output_;
	const int N = this->blobs_[0]->count();
	const int weight_dim = N / num;
	//binarize weights.
	caffe_abs(N, weight, W_b.mutable_cpu_data());
	const Dtype* binaryweight = W_b.cpu_data();
	caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, weight_dim, 1. / weight_dim, binaryweight, weight_sum_multiplier.cpu_data(), 0.,
		alphas_.mutable_cpu_data());
	for (int i = 0; i < N; i++){
		int n = i / weight_dim;
		Dtype binary_code = (weight[i] >= 0) ? 1 : -1;
		W_b.mutable_cpu_data()[i] = binary_code*alphas_.cpu_data()[n];
	}
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top[i]->mutable_cpu_data();
		for (int n = 0; n < this->num_; ++n) {
			this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, binaryweight,
				top_data + n * this->top_dim_);
			if (this->bias_term_) {
				const Dtype* bias = this->blobs_[1]->cpu_data();
				this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
			}
		}
	}
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& top){
	//const Dtype* weight = this->blobs_[0]->cpu_data();
	const Dtype* binaryweight = W_b.cpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
			Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
			}
		}
		if (this->param_propagate_down_[0] || propagate_down[i]) {
			for (int n = 0; n < this->num_; ++n) {
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
						top_diff + n * this->top_dim_, weight_diff);
				} 
				// gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) {
					this->backward_cpu_gemm(top_diff + n * this->top_dim_, binaryweight,
						bottom_diff + n * this->bottom_dim_);
				}
			}
			//
			const Dtype* weight = this->blobs_[0]->cpu_data();
			const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
			for (int i = 0; i < this->blobs_[0]->count(); i++){
				const int n = i / weight_dim;
				Dtype multiplier = 0;
				if (abs(weight[i]) >= 1)
					multiplier = 0;
				else
				{
					multiplier = 1;
					multiplier *= alphas_.cpu_data()[n];
				}
				multiplier += Dtype(1) / this->blobs_[0]->count();
				weight_diff[i] *= multiplier;
			}
		}
	} 
}

#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);
REGISTER_LAYER_CLASS(BinaryConvolution);
}