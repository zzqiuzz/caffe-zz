#include <vector>
#include "caffe/layers/ternary_conv_layer.hpp"

namespace caffe{
template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
	this->delta_ = 0;
	this->alpha_ = 0;
	this->W_t.Reshape(this->blobs_[0]->shape());
	 
}
template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::compute_output_shape() {
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
void TernaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){  
	//Not Implemented. 
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& top){
	//Not Impelemented. 
}

#ifdef CPU_ONLY
STUB_GPU(TernaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(TernaryConvolutionLayer);
REGISTER_LAYER_CLASS(TernaryConvolution);
}